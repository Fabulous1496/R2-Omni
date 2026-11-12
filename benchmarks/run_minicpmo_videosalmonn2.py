"""Video-SALMONN 2 runner for MiniCPM-o-4.5 and released baselines."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from r2omni.minicpm import MiniCPMGenerationAdapter
from r2omni.common.paths import project_root, resolve_workspace, resolve_third_party
from r2omni.sspa.integration.qwen_sspa_runtime import cache_tensor_storage_bytes


def _media_loaders(workspace: Path, helper_path: Path | None = None) -> tuple[Any, Any]:
    helper = helper_path or os.environ.get("MINICPM_MEDIA_HELPER")
    if helper:
        source = Path(helper).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"MiniCPM media helper not found: {source}")
        spec = importlib.util.spec_from_file_location("minicpmo_media_utils", source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load MiniCPM media helper from {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.get_video_frame_audio_segments, module.get_audio_segments
    try:
        from minicpmo.utils import get_audio_segments, get_video_frame_audio_segments

        return get_video_frame_audio_segments, get_audio_segments
    except ImportError as exc:
        raise ImportError(
            "Install minicpmo-utils==1.0.6 in this environment, or supply "
            "--media-helper /path/to/minicpmo/utils.py (MINICPM_MEDIA_HELPER)."
        ) from exc


def _question(row: dict[str, Any]) -> str:
    if "question" in row:
        return str(row["question"])
    return next(
        str(turn["value"]).replace("<image>", "").strip()
        for turn in row["conversations"]
        if str(turn.get("from", "")).lower() == "human"
    )


def _load_media(loader: Any, video: Path) -> tuple[Any, Any, Any, bool]:
    kwargs = {
        "stack_frames": 1,
        "use_ffmpeg": False,
        "adjust_audio_length": True,
        "use_audio": True,
    }
    try:
        frames, audio, stacked = loader(str(video), **kwargs)
        return frames, audio, stacked, False
    except Exception:
        # video/36.mp4 contains malformed H.264 packets that Decord cannot read.
        # Repair only the temporary job-local copy and keep the benchmark intact.
        with tempfile.TemporaryDirectory(prefix="minicpm-media-") as directory:
            repaired = Path(directory) / video.name
            subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-y", "-i", str(video),
                    "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                    "-preset", "ultrafast", "-crf", "18", "-c:a", "copy",
                    str(repaired),
                ],
                check=True,
            )
            frames, audio, stacked = loader(str(repaired), **kwargs)
            return frames, audio, stacked, True


def _load_fixed_media(
    audio_loader: Any,
    video: Path,
    num_frames: int,
) -> tuple[Any, Any, None, bool]:
    """Uniformly sample an exact frame count and align audio by timestamp."""

    import numpy as np
    from decord import VideoReader, cpu
    from PIL import Image

    def load(path: Path) -> tuple[Any, Any, None]:
        reader = VideoReader(str(path), ctx=cpu(0))
        fps = float(reader.get_avg_fps())
        indices = np.linspace(0, len(reader) - 1, num_frames, dtype=np.int64)
        timestamps = [float(index) / fps for index in indices]
        frames = [
            Image.fromarray(frame.astype("uint8")).convert("RGB")
            for frame in reader.get_batch(indices).asnumpy()
        ]
        audio_chunks = audio_loader(
            list(range(math.ceil(len(reader) / fps))),
            len(reader) / fps,
            str(path),
            adjust_length=True,
            use_ffmpeg=False,
        )
        # MiniCPM is trained with one-second audio chunks. Keep that official
        # audio granularity while inserting the additional uniformly sampled
        # video frames at their timestamps.
        audio = [None] * num_frames
        timeline = np.asarray(timestamps)
        for second, chunk in enumerate(audio_chunks):
            position = int(np.abs(timeline - second).argmin())
            audio[position] = chunk
        return frames, audio, None

    try:
        frames, audio, stacked = load(video)
        return frames, audio, stacked, False
    except Exception:
        with tempfile.TemporaryDirectory(prefix="minicpm-media-") as directory:
            repaired = Path(directory) / video.name
            subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-y", "-i", str(video),
                    "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                    "-preset", "ultrafast", "-crf", "18", "-c:a", "copy",
                    str(repaired),
                ],
                check=True,
            )
            frames, audio, stacked = load(repaired)
        return frames, audio, stacked, True


def _load_fps_media(
    audio_loader: Any,
    video: Path,
    video_fps: float,
    max_video_frames: int,
) -> tuple[Any, Any, None, bool, float]:
    """Sample the complete clip at fixed FPS with uniform capped coverage."""

    import numpy as np
    from decord import VideoReader, cpu
    from PIL import Image

    def load(path: Path) -> tuple[Any, Any, None, float]:
        reader = VideoReader(str(path), ctx=cpu(0))
        source_fps = float(reader.get_avg_fps())
        duration = len(reader) / source_fps
        requested = min(int(duration * video_fps) // 2 * 2, max_video_frames)
        requested = min(max(requested, 2), len(reader))
        indices = np.linspace(0, len(reader) - 1, requested).round().astype(np.int64)
        indices = np.unique(indices)
        frame_times = indices.astype(np.float64) / source_fps
        frames = [
            Image.fromarray(frame.astype("uint8")).convert("RGB")
            for frame in reader.get_batch(indices).asnumpy()
        ]
        audio_chunks = audio_loader(
            list(range(math.ceil(duration))),
            duration,
            str(path),
            adjust_length=True,
            use_ffmpeg=False,
        ) or []
        audio = [None] * len(frames)
        for second, chunk in enumerate(audio_chunks):
            position = int(np.abs(frame_times - second).argmin())
            audio[position] = chunk
        return frames, audio, None, duration

    try:
        frames, audio, stacked, duration = load(video)
        return frames, audio, stacked, False, duration
    except Exception:
        with tempfile.TemporaryDirectory(prefix="minicpm-media-") as directory:
            repaired = Path(directory) / video.name
            subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-y", "-i", str(video),
                    "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                    "-preset", "ultrafast", "-crf", "18", "-c:a", "copy",
                    str(repaired),
                ],
                check=True,
            )
            frames, audio, stacked, duration = load(repaired)
            return frames, audio, stacked, True, duration


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _install_method(
    model: Any,
    method: str,
    workspace: Path,
    max_new_tokens: int,
    *,
    r2_matched_fp_until_layer: int | None = None,
    r2_stc_config: Path | None = None,
    r2_shallow_v_bits: float = 2.1,
    r2_deep_v_bits: float = 2.0,
    r2_key_bits: int = 4,
    r2_group_size: int = 128,
    r2_layer_group_size: int = 4,
    kv_k_bits: int | None = None,
    kv_v_bits: int | None = None,
    kv_group_size: int | None = None,
    kv_residual_length: int | None = None,
    kv_fp_prefix_layers: int | None = None,
    kv_readout_backend: str | None = None,
    kivi_implementation: str | None = None,
    ott_sink_num: int | None = None,
    ott_sink_max_size: int | None = None,
    oscar_num_bits: int = 2,
    oscar_group_size: int = 32,
    oscar_kv_rotation: str = "hadamard",
    oscar_kv_norm: str = "1",
    token_target_compression: float = 5.0,
) -> tuple[MiniCPMGenerationAdapter | None, Any | None]:
    if method == "bf16":
        return None, None
    if method in {"omniselect", "omnizip", "seats"}:
        from r2omni.minicpm.token_pruning import MiniCPMTokenPruner

        pruner = MiniCPMTokenPruner(
            method,
            workspace,
            next(model.parameters()).device,
            target_compression=token_target_compression,
        )
        bridge = MiniCPMGenerationAdapter(model, token_pruner=pruner).install()
        return bridge, pruner
    if method in {"kivi-om", "ott-om"}:
        if method == "kivi-om":
            from baselines.kivi_om import install_kivi_om

            source_root = resolve_third_party("KIVI", workspace)
            native = install_kivi_om(
                model,
                source_root=source_root,
                implementation=kivi_implementation or "official",
                k_bits=2 if kv_k_bits is None else kv_k_bits,
                v_bits=2 if kv_v_bits is None else kv_v_bits,
                group_size=128 if kv_group_size is None else kv_group_size,
                residual_length=384 if kv_residual_length is None else kv_residual_length,
                fp_prefix_layers=0 if kv_fp_prefix_layers is None else kv_fp_prefix_layers,
                readout_backend=kv_readout_backend or "native_packed",
            )
        else:
            from baselines.ott_om import install_ott_om

            source_root = resolve_third_party("OTT", workspace)
            native = install_ott_om(
                model,
                source_root=source_root,
                k_bits=2 if kv_k_bits is None else kv_k_bits,
                v_bits=2 if kv_v_bits is None else kv_v_bits,
                group_size=128 if kv_group_size is None else kv_group_size,
                residual_length=256 if kv_residual_length is None else kv_residual_length,
                fp_prefix_layers=0 if kv_fp_prefix_layers is None else kv_fp_prefix_layers,
                readout_backend=kv_readout_backend or "native_packed",
                sink_num=3 if ott_sink_num is None else ott_sink_num,
                sink_max_size=32 if ott_sink_max_size is None else ott_sink_max_size,
            )
        bridge = MiniCPMGenerationAdapter(
            model, cache_factory=native.new_cache
        ).install()
        return bridge, native
    if method == "oscar":
        from baselines.oscar_om import install_oscar_om

        source_root = resolve_third_party("OScaR-KV-Quant", workspace)
        extension_root = source_root
        cache_factory = install_oscar_om(
            model,
            source_root=source_root,
            extension_root=extension_root,
            num_bits=oscar_num_bits,
            group_size=oscar_group_size,
            kv_rotation=oscar_kv_rotation,
            kv_norm=oscar_kv_norm,
        )
        bridge = MiniCPMGenerationAdapter(
            model, cache_factory=cache_factory
        ).install()
        return bridge, None
    if method == "r2omni":
        from r2omni.sspa.integration.qwen_sspa_attention import (
            QwenSSPAAttentionManager,
        )
        from r2omni.sspa.integration.sspa_config import SSPAConfig
        from r2omni.sspa.policy.layer_strategy import (
            LayerStrategyConfig,
            build_layer_strategy,
        )
        from r2omni.stc.pre_llm_units import load_config

        strategy = build_layer_strategy(
            LayerStrategyConfig(
                mode="block",
                num_layers=len(model.llm.model.layers),
                block_start_layer=0,
                block_group_size=r2_layer_group_size,
            )
        )
        manager = QwenSSPAAttentionManager(
            model,
            strategy,
            SSPAConfig(
                fp_append_capacity=max_new_tokens + 4,
                value_int2_group_size=r2_group_size,
                shallow_v_payload_bit_ceiling=r2_shallow_v_bits,
                deep_v_payload_bit_ceiling=r2_deep_v_bits,
            ),
            kernel_backend="triton_mixed_fused_splitk",
            cache_variant=(
                "v_fourq_k_int2_compact"
                if r2_key_bits == 2
                else "v_fourq_k_int4_compact"
            ),
            fallback_on_error=False,
            validate_kernel_output=False,
            key_quantization_superblock=r2_group_size,
            key_hadamard=True,
            value_hadamard=True,
            matched_fp_until_layer=r2_matched_fp_until_layer,
        ).install()
        bridge = MiniCPMGenerationAdapter(
            model,
            sspa_manager=manager,
            stc_config=load_config(
                r2_stc_config or project_root() / "configs/stc.yaml"
            ),
        ).install()
        return bridge, manager
    raise ValueError(f"unsupported method: {method}")


def _cache_metadata(
    method: str,
    bridge: MiniCPMGenerationAdapter | None,
    runtime: Any | None,
) -> dict[str, Any]:
    if method in {"omniselect", "omnizip", "seats"}:
        return {
            "backend": "transformers_bf16_dynamic_cache",
            "token_pruning": (
                bridge.last_token_pruning_metadata if bridge is not None else None
            ),
        }
    if method == "r2omni":
        return runtime.get_stats()
    if method in {"kivi-om", "ott-om"}:
        if bridge is None or bridge.last_cache is None:
            return {"backend": "native_packed", "status": "cache_not_created"}
        return runtime.storage_metadata(bridge.last_cache)
    if method == "oscar":
        cache = bridge.last_cache
        return {
            "backend": "official_oscar_qwen3_cuda_packed",
            "offline_v_hadamard": bool(
                getattr(bridge.model, "_r2omni_oscar_offline_v_hadamard", False)
            ),
            "logical_sequence_length": (
                int(cache.get_seq_length()) if cache is not None else None
            ),
            "packed_sequence_length": (
                int(getattr(cache, "_r2omni_packed_length", 0))
                if cache is not None
                else None
            ),
            "residual_sequence_length": (
                int(getattr(cache, "_r2omni_residual_length", 0))
                if cache is not None
                else None
            ),
            "actual_cache_storage_bytes": (
                _tensor_bytes(vars(cache)) if cache is not None else None
            ),
            "allocated_persistent_cache_bytes": (
                cache_tensor_storage_bytes(cache) if cache is not None else None
            ),
        }
    return {"backend": "transformers_bf16_dynamic_cache"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=resolve_workspace())
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("videosalmonn2", "longvale", "omnidcbench"),
        default="videosalmonn2",
    )
    parser.add_argument(
        "--method",
        choices=(
            "bf16", "r2omni", "oscar", "kivi-om", "ott-om",
            "omniselect", "omnizip", "seats",
        ),
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--video-frames", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--max-video-frames", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--r2-matched-fp-until-layer", type=int, default=None)
    parser.add_argument("--r2-stc-config", type=Path, default=None)
    parser.add_argument("--r2-shallow-v-bits", type=float, default=2.1)
    parser.add_argument("--r2-deep-v-bits", type=float, default=2.0)
    parser.add_argument("--r2-key-bits", type=int, choices=(2, 4), default=4)
    parser.add_argument("--r2-group-size", type=int, choices=(32, 128), default=128)
    parser.add_argument("--r2-layer-group-size", type=int, default=4)
    parser.add_argument("--media-helper", type=Path)
    parser.add_argument("--kv-k-bits", type=int, choices=(2, 4))
    parser.add_argument("--kv-v-bits", type=int, choices=(2, 4))
    parser.add_argument("--kv-group-size", type=int)
    parser.add_argument("--kv-residual-length", type=int)
    parser.add_argument("--kv-fp-prefix-layers", type=int)
    parser.add_argument("--kv-readout-backend", choices=("native_packed", "dequant_fa2_reference", "dequant_sdpa_reference"))
    parser.add_argument("--kivi-implementation", choices=("legacy", "official"))
    parser.add_argument("--ott-sink-num", type=int)
    parser.add_argument("--ott-sink-max-size", type=int)
    parser.add_argument("--oscar-num-bits", type=int, choices=(2, 4), default=2)
    parser.add_argument("--oscar-group-size", type=int, default=32)
    parser.add_argument(
        "--oscar-kv-rotation", choices=("none", "hadamard"), default="hadamard"
    )
    parser.add_argument("--oscar-kv-norm", choices=("0", "1"), default="1")
    parser.add_argument("--token-target-compression", type=float, default=5.0)
    args = parser.parse_args()
    args.workspace = resolve_workspace(args.workspace)
    get_video_frame_audio_segments, get_audio_segments = _media_loaders(args.workspace, args.media_helper)

    if args.dataset == "videosalmonn2":
        dataset_root = args.workspace / "Benchmarks/videoSALMONN2"
        rows = json.loads((dataset_root / "video_salmonn2_test.json").read_text())
    elif args.dataset == "longvale":
        from r2omni.common.longvale import DENSE_CAPTION_PROMPT

        dataset_root = args.workspace / "Benchmarks/LongVALE/videos_eval/video_test_1171"
        annotations = json.loads(
            (args.workspace / "Benchmarks/LongVALE/longvale-annotations-eval.json").read_text()
        )
        rows = [
            {
                **annotation,
                "video_id": str(video_id),
                "video": f"{video_id}.mp4",
                "question": DENSE_CAPTION_PROMPT,
            }
            for video_id, annotation in annotations.items()
        ]
    else:
        from r2omni.common.omnidcbench import load_omnidcbench_rows

        dataset_root = args.workspace / "Benchmarks/OmniDCBench"
        rows = load_omnidcbench_rows(dataset_root / "ours_gt_file.json")
        rows = [
            {
                **row,
                "video_id": str(row["clip_path"]),
                "video": str(row["clip_path"]),
            }
            for row in rows
        ]
    if args.dataset == "omnidcbench":
        if args.video_fps is None or args.video_fps <= 0:
            parser.error("OmniDCBench requires a positive --video-fps")
        if args.max_video_frames is None or args.max_video_frames <= 0:
            parser.error("OmniDCBench requires a positive --max-video-frames")
        if args.video_frames is not None:
            parser.error("OmniDCBench fixed-FPS evaluation cannot use --video-frames")
    elif args.video_fps is not None or args.max_video_frames is not None:
        parser.error(
            "--video-fps/--max-video-frames are currently specific to OmniDCBench"
        )
    rows = rows[args.start_index :]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions.jsonl"

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        init_vision=True,
        init_audio=True,
        init_tts=False,
    ).eval().cuda()
    attention = {
        "llm_prefill": str(model.llm.config._attn_implementation),
        "vision_prefill": str(model.vpm.config._attn_implementation),
        "audio_prefill": str(model.apm.config._attn_implementation),
    }
    expected = {
        "llm_prefill": "flash_attention_2",
        "vision_prefill": "flash_attention_2",
        "audio_prefill": "sdpa",
    }
    if attention != expected:
        raise RuntimeError(f"unexpected MiniCPM attention configuration: {attention}")
    bridge, runtime = _install_method(
        model,
        args.method,
        args.workspace,
        args.max_new_tokens,
        r2_matched_fp_until_layer=args.r2_matched_fp_until_layer,
        r2_stc_config=args.r2_stc_config,
        r2_shallow_v_bits=args.r2_shallow_v_bits,
        r2_deep_v_bits=args.r2_deep_v_bits,
        r2_key_bits=args.r2_key_bits,
        r2_group_size=args.r2_group_size,
        r2_layer_group_size=args.r2_layer_group_size,
        kv_k_bits=args.kv_k_bits,
        kv_v_bits=args.kv_v_bits,
        kv_group_size=args.kv_group_size,
        kv_residual_length=args.kv_residual_length,
        kv_fp_prefix_layers=args.kv_fp_prefix_layers,
        kv_readout_backend=args.kv_readout_backend,
        kivi_implementation=args.kivi_implementation,
        ott_sink_num=args.ott_sink_num,
        ott_sink_max_size=args.ott_sink_max_size,
        oscar_num_bits=args.oscar_num_bits,
        oscar_group_size=args.oscar_group_size,
        oscar_kv_rotation=args.oscar_kv_rotation,
        oscar_kv_norm=args.oscar_kv_norm,
        token_target_compression=args.token_target_compression,
    )

    records = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with predictions.open("w", encoding="utf-8") as writer:
        for index, row in enumerate(rows):
            try:
                model.reset_session()
                observed_duration = None
                if args.video_fps is not None:
                    (
                        frames,
                        audio_segments,
                        stacked_frames,
                        repaired_media,
                        observed_duration,
                    ) = _load_fps_media(
                        get_audio_segments,
                        dataset_root / row["video"],
                        args.video_fps,
                        args.max_video_frames,
                    )
                    sampling_strategy = "fixed_fps_capped_uniform_audio_aligned"
                elif args.video_frames is None:
                    frames, audio_segments, stacked_frames, repaired_media = _load_media(
                        get_video_frame_audio_segments,
                        dataset_root / row["video"],
                    )
                    sampling_strategy = "minicpmo_native_audio_video_alignment"
                else:
                    frames, audio_segments, stacked_frames, repaired_media = _load_fixed_media(
                        get_audio_segments,
                        dataset_root / row["video"],
                        args.video_frames,
                    )
                    sampling_strategy = "uniform_fixed_frames_audio_aligned"
                content = []
                for position, frame in enumerate(frames):
                    content.append(frame)
                    if audio_segments[position] is not None:
                        content.append(audio_segments[position])
                    if stacked_frames is not None and stacked_frames[position] is not None:
                        content.append(stacked_frames[position])
                content.append(_question(row))
                messages = [
                    model.get_sys_prompt(ref_audio=None, mode="omni", language="en"),
                    {"role": "user", "content": content},
                ]
                if runtime is not None and hasattr(runtime, "set_media"):
                    runtime.set_media(_question(row), frames, audio_segments)
                answer = model.chat(
                    msgs=messages,
                    max_new_tokens=args.max_new_tokens,
                    max_inp_length=32768,
                    do_sample=False,
                    num_beams=1,
                    use_image_id=False,
                    max_slice_nums=1,
                    use_tts_template=True,
                    enable_thinking=False,
                    omni_mode=True,
                    generate_audio=False,
                )
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError("empty MiniCPM response")
                sample_id = str(row.get("video_id") or Path(str(row["video"])).name)
                record = {
                    "dataset": args.dataset,
                    "sample_id": sample_id,
                    "video": row["video"],
                    "question": _question(row),
                    "response": answer.strip(),
                    "prediction": answer.strip(),
                    "dataset_metadata": (
                        {"events": row.get("events", [])}
                        if args.dataset == "videosalmonn2"
                        else ({
                            "video_id": row["video_id"],
                            "duration": row.get("duration"),
                            "timestamps": row.get("timestamps"),
                            "sentences": row.get("sentences"),
                        } if args.dataset == "longvale" else {
                            "clip_path": row.get("clip_path"),
                            "orig_video_path": row.get("orig_video_path"),
                            "duration": row.get("duration"),
                            "seg_num": row.get("seg_num"),
                            "official_fps": float(args.video_fps),
                            "official_max_frames": args.max_video_frames,
                        })
                    ),
                    "media_sampling": {
                        "strategy": sampling_strategy,
                        "video_frames": len(frames),
                        "video_fps": args.video_fps,
                        "duration": observed_duration,
                        "max_video_frames": args.max_video_frames,
                        "repaired_media": repaired_media,
                    },
                    "cache": _cache_metadata(args.method, bridge, runtime),
                    "stc": None if bridge is None else bridge.last_stc_metadata,
                    "error": None,
                }
            except Exception as exc:
                record = {
                    "sample_id": str(row.get("video_id") or Path(str(row["video"])).name),
                    "video": row.get("video"),
                    "question": _question(row),
                    "response": "",
                    "prediction": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            records.append(record)
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.flush()
            print(
                f"sample={index + 1}/{len(rows)} method={args.method} "
                f"error={record['error']}",
                flush=True,
            )

    failures = sum(bool(record["error"]) for record in records)
    result = {
        "status": "success" if failures == 0 else "failed",
        "model_family": "MiniCPM-o-4.5",
        "method": args.method,
        "dataset": {
            "longvale": "LongVALE",
            "videosalmonn2": "VideoSALMONN2",
            "omnidcbench": "OmniDCBench",
        }[args.dataset],
        "model_path": str(args.model.resolve()),
        "samples": len(records),
        "successful_count": len(records) - failures,
        "failed_count": failures,
        "attention": attention,
        "processor_cache": False,
        "video_fps": args.video_fps,
        "max_video_frames": args.max_video_frames,
        "max_new_tokens": args.max_new_tokens,
        "decode_backend": _cache_metadata(args.method, bridge, runtime).get(
            "backend", args.method
        ),
        "run_seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"{failures} samples failed")


if __name__ == "__main__":
    main()
