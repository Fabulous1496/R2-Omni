#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import random
import traceback
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from benchmarks.interface import benchmark_names, get_benchmark
from r2omni.common.paths import project_root, resolve_workspace, resolve_third_party
from r2omni.sspa.integration.qwen_sspa_attention import QwenSSPAAttentionManager
from r2omni.sspa.integration.sspa_config import SSPAConfig
from r2omni.sspa.integration.qwen_sspa_runtime import (
    calculate_actual_cache_storage_breakdown,
)
from r2omni.sspa.integration.qwen_r2omni_decode import r2omni_greedy_decode
from r2omni.sspa.integration.qwen_r2omni_decode import official_model_generate_decode
from r2omni.sspa.integration.qwen_modality_utils import infer_qwen_omni_modality_vector
from r2omni.sspa.integration.qwen_omni_greedy_decode import _decode_text
from r2omni.sspa.policy.layer_strategy import (
    LayerStrategyConfig,
    build_layer_strategy,
    summarize_layer_strategy,
)


# Dataset-specific generation ceilings from the official evaluation paths.
# MCQ tasks still stop at EOS (normally after a single option), so a large
# ceiling does not force extra decode work.
OFFICIAL_MAX_NEW_TOKENS = {
    "worldsense": 2048,
    "video_mme": 2048,
    "daily_omni": 10,
    "video_salmonn2": 1024,
    "longvale": 1024,
    "omnidcbench": 8192,
    "avsd": 20,
}


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _quality_metadata(value: Any) -> Any:
    """Strip performance counters from auxiliary policy metadata."""
    if isinstance(value, dict):
        return {
            str(key): _quality_metadata(item)
            for key, item in value.items()
            if not (
                "latency" in str(key).lower()
                or str(key).lower().endswith(("_ms", "_sec"))
                or str(key).lower() in {"timing", "timings"}
            )
        }
    if isinstance(value, (list, tuple)):
        return [_quality_metadata(item) for item in value]
    return value


def _baseline_reference(
    baseline: dict[Any, dict[str, Any]], sample_id: Any
) -> dict[str, Any] | None:
    if sample_id in baseline:
        return baseline[sample_id]
    text_id = str(sample_id)
    if text_id in baseline:
        return baseline[text_id]
    try:
        integer_id = int(sample_id)
    except (TypeError, ValueError):
        return None
    return baseline.get(integer_id)


def extract_first_generated_choice(
    processor: Any, generated_ids: torch.Tensor
) -> tuple[str, str]:
    """Decode and grade only the first generated token of an MCQ response."""
    from r2omni.common.worldsense import extract_choice

    if not torch.is_tensor(generated_ids) or generated_ids.ndim != 2:
        raise ValueError("generated_ids must have shape [1,T]")
    if int(generated_ids.shape[0]) != 1 or int(generated_ids.shape[1]) < 1:
        raise ValueError("generated_ids must contain at least one batch-1 token")
    first_token_text = _decode_text(processor, generated_ids[:, :1])
    return extract_choice(first_token_text), first_token_text


def aggregate_accuracy(records: list[dict[str, Any]], baseline: dict[Any, dict[str, Any]] | None = None) -> dict[str, Any]:
    valid = [record for record in records if not record.get("error")]
    total = len(records)
    successful = len(valid)
    correct = sum(bool(record.get("correct")) for record in valid)
    empty = sum(not str(record.get("response", "")).strip() for record in valid)
    invalid = sum(bool(str(record.get("response", "")).strip()) and not record.get("prediction") for record in valid)
    match_total = 0
    match_count = 0
    baseline_correct = 0
    candidate_correct = 0
    baseline_accuracy = None
    candidate_accuracy = None
    if baseline:
        for record in valid:
            reference = _baseline_reference(baseline, record["sample_id"])
            if reference is not None and not reference.get("error"):
                match_total += 1
                match_count += record.get("prediction") == reference.get("prediction")
                baseline_correct += int(bool(reference.get("correct")))
                candidate_correct += int(bool(record.get("correct")))
        baseline_accuracy = baseline_correct / match_total if match_total else None
        candidate_accuracy = candidate_correct / match_total if match_total else None
    # In an evaluation, a failed sample is not allowed to disappear from the
    # accuracy denominator.  Keep successful_count explicit for operational
    # diagnostics while treating failures as incorrect for the headline metric.
    accuracy = correct / max(total, 1)
    return {
        "correct": correct,
        "total": total,
        "successful_count": successful,
        "accuracy": accuracy,
        "accuracy_drop_vs_baseline": (
            None if baseline_accuracy is None or candidate_accuracy is None
            else baseline_accuracy - candidate_accuracy
        ),
        "answer_match_rate_vs_baseline": None if not match_total else match_count / match_total,
        "invalid_output_count": invalid,
        "empty_output_count": empty,
        "failed_count": total - successful,
    }


def _average(records: list[dict[str, Any]], path: tuple[str, ...], default: float = 0.0) -> float:
    values: list[float] = []
    for record in records:
        if record.get("error"):
            continue
        value: Any = record
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            values.append(float(value))
    return sum(values) / max(len(values), 1) if values else default


def _kv_geometry(thinker: Any) -> dict[str, int]:
    """Return the text decoder geometry needed for a BF16 KV reference."""
    config = getattr(thinker, "config", None)
    text_config = getattr(config, "text_config", None) or config
    num_layers = int(getattr(text_config, "num_hidden_layers"))
    num_kv_heads = int(getattr(text_config, "num_key_value_heads"))
    hidden_size = int(getattr(text_config, "hidden_size"))
    num_attention_heads = int(getattr(text_config, "num_attention_heads"))
    head_dim = int(
        getattr(text_config, "head_dim", hidden_size // num_attention_heads)
    )
    return {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "bf16_bytes_per_token": (
            num_layers * num_kv_heads * head_dim * 2 * 2
        ),
    }


def _kv_budget_record(
    *,
    thinker: Any,
    processor_input_tokens: int,
    decode: dict[str, Any],
    stc_stats: dict[str, Any],
    manager_budget_stats: dict[str, Any],
    final_cache: Any,
    active_manager: bool,
) -> dict[str, Any]:
    """Measure the persistent KV cache at the end of generation."""
    geometry = _kv_geometry(thinker)
    prefill_original_tokens = int(
        stc_stats.get("original_seq_len", processor_input_tokens)
    )
    prefill_stored_tokens = int(
        decode.get(
            "initial_effective_attention_length",
            decode.get("original_kv_tokens", prefill_original_tokens),
        )
    )
    generated_tokens = int(decode["generated_ids"].shape[-1])
    # Generation caches contain the prompt and all generated tokens except the
    # final token whose logits were returned.  Normalize every method against
    # the BF16 bytes for that same uncompressed logical sequence.
    final_reference_tokens = prefill_original_tokens + max(generated_tokens - 1, 0)
    reference_bf16_bytes = (
        final_reference_tokens * geometry["bf16_bytes_per_token"]
    )

    if active_manager:
        storage = dict(manager_budget_stats.get("cache_storage") or {})
    else:
        storage = calculate_actual_cache_storage_breakdown(final_cache)
    active_bytes = int(
        storage.get(
            "active_persistent_cache_bytes",
            storage.get("persistent_cache_bytes", 0),
        )
    )
    if active_bytes <= 0:
        raise RuntimeError("persistent cache is empty")

    active_ratio = active_bytes / max(reference_bf16_bytes, 1)
    allocated_bytes = int(storage.get("allocated_persistent_cache_bytes", active_bytes))
    # Keep the paper's active ratio and separately disclose reachable backing
    # storage. In particular, a residual slice may retain a larger allocation.
    return {
        "measurement_point": "end_of_generation_cache",
        "prefill_original_tokens": prefill_original_tokens,
        "prefill_stored_tokens": prefill_stored_tokens,
        "prefill_token_keep_ratio": (
            prefill_stored_tokens / max(prefill_original_tokens, 1)
        ),
        "generated_tokens": generated_tokens,
        "final_reference_tokens": final_reference_tokens,
        "final_stored_logical_tokens": int(
            decode.get("effective_attention_length", prefill_stored_tokens)
        ),
        "reference_bf16_cache_bytes": reference_bf16_bytes,
        "persistent_cache_bytes": active_bytes,
        "active_persistent_cache_bytes": active_bytes,
        "allocated_persistent_cache_bytes": allocated_bytes,
        "allocated_physical_compression_factor_vs_bf16": reference_bf16_bytes / max(allocated_bytes, 1),
        "allocated_storage_scope": "reachable cache backing/capacity; excludes model, working buffers and allocator reserved pool",
        "physical_ratio_vs_bf16": active_ratio,
        "physical_compression_factor_vs_bf16": 1.0 / max(active_ratio, 1e-12),
        "physical_equivalent_kv_bits_including_metadata": 16.0 * active_ratio,
        "compression_accounting": "active-v1",
        "geometry": geometry,
        "storage_breakdown": storage,
    }


def _aggregate_kv_budget(records: list[dict[str, Any]]) -> dict[str, Any]:
    measurements = [
        record["kv_cache_budget"]
        for record in records
        if not record.get("error") and record.get("kv_cache_budget")
    ]
    if not measurements:
        return {"samples": 0}

    original_tokens = sum(int(item["prefill_original_tokens"]) for item in measurements)
    stored_tokens = sum(int(item["prefill_stored_tokens"]) for item in measurements)
    reference_bytes = sum(
        int(item["reference_bf16_cache_bytes"]) for item in measurements
    )
    active_bytes = sum(
        int(item["persistent_cache_bytes"]) for item in measurements
    )
    active_ratio = active_bytes / max(reference_bytes, 1)
    allocated_bytes = sum(int(item.get("allocated_persistent_cache_bytes", item["persistent_cache_bytes"])) for item in measurements)
    return {
        "samples": len(measurements),
        "measurement_point": "end_of_generation_cache",
        "prefill_original_tokens_total": original_tokens,
        "prefill_stored_tokens_total": stored_tokens,
        "prefill_token_keep_ratio": stored_tokens / max(original_tokens, 1),
        "generated_tokens_total": sum(int(item["generated_tokens"]) for item in measurements),
        "reference_bf16_cache_bytes_total": reference_bytes,
        "persistent_cache_bytes_total": active_bytes,
        "active_persistent_cache_bytes_total": active_bytes,
        "allocated_persistent_cache_bytes_total": allocated_bytes,
        "allocated_physical_compression_factor_vs_bf16": reference_bytes / max(allocated_bytes, 1),
        "physical_ratio_vs_bf16": active_ratio,
        "physical_compression_factor_vs_bf16": 1.0 / max(active_ratio, 1e-12),
        "physical_equivalent_kv_bits_including_metadata": 16.0 * active_ratio,
        "compression_accounting": "active-v1",
        "per_sample_persistent_cache_bytes_min": min(
            int(item["persistent_cache_bytes"]) for item in measurements
        ),
        "per_sample_persistent_cache_bytes_max": max(
            int(item["persistent_cache_bytes"]) for item in measurements
        ),
        "geometry": measurements[0]["geometry"],
    }


def _load_baseline(path: str | None) -> dict[Any, dict[str, Any]] | None:
    if not path:
        return None
    source = Path(path)
    if source.is_dir():
        source = source / "predictions.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"baseline predictions not found: {source}")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {record["sample_id"]: record for record in records}


def _load_dataset_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    adapter = get_benchmark(args.dataset)
    rows = adapter.load_rows(Path(args.ann).resolve())
    video_root = Path(args.video_root).resolve()
    available = [
        row
        for row in rows
        if Path(adapter.resolve_video(video_root, row)).is_file()
    ]
    stats = {
        "annotation_rows": len(rows),
        "available_media_rows": len(available),
        "missing_media_rows": len(rows) - len(available),
    }
    if adapter.audio_root_relative is not None or adapter.name == "daily_omni":
        audio_root = Path(args.audio_root).resolve() if args.audio_root else None
        available_audio = [
            row
            for row in rows
            if (adapter.resolve_audio(video_root, row, audio_root) or Path()).is_file()
        ]
        stats.update({
            "available_audio_rows": len(available_audio),
            "missing_audio_rows": len(rows) - len(available_audio),
        })
    return (available if args.available_media_only else rows), stats


def resolve_dataset_video_path(
    video_root: Path, row: dict[str, Any], dataset: str = "worldsense"
) -> str:
    return get_benchmark(dataset).resolve_video(video_root, row)


def _sample_id(row: dict[str, Any], fallback: int) -> Any:
    return row.get("index", fallback)


def resolve_text_attention(args: argparse.Namespace) -> str:
    """Resolve the text attention used by the native prefill path.

    ``official_generate`` remains the upstream SDPA implementation unless an
    experiment explicitly opts into FlashAttention-2. Other runners may select
    a native prefill implementation while retaining their custom cache/decode
    manager after the first forward.
    """
    value = str(getattr(args, "text_prefill_attention", "sdpa"))
    if value not in {"sdpa", "eager", "flash_attention_2"}:
        raise ValueError(f"unsupported text prefill attention: {value}")
    if getattr(args, "backend", None) == "official_generate":
        allow_fa2 = bool(
            getattr(args, "allow_official_flash_attention_2", False)
        )
        return (
            "flash_attention_2"
            if allow_fa2 and value == "flash_attention_2"
            else "sdpa"
        )
    return value


def _native_kv_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path | None]:
    workspace = resolve_workspace()
    if args.kv_cache_baseline == "kivi-om":
        if args.kivi_implementation == "legacy":
            source_root = (
                Path(args.kivi_legacy_root).resolve()
                if args.kivi_legacy_root
                else resolve_third_party("KIVI", workspace)
            )
            return source_root, None
        source_root = (
            Path(args.kivi_official_root).resolve()
            if args.kivi_official_root
            else resolve_third_party("KIVI", workspace)
        )
        extension_root = (
            Path(args.kivi_native_extension_root).resolve()
            if args.kivi_native_extension_root
            else source_root / "quant"
        )
        return source_root, extension_root
    if args.kv_cache_baseline == "ott-om":
        source_root = (
            Path(args.ott_official_root).resolve()
            if args.ott_official_root
            else resolve_third_party("OTT", workspace)
        )
        extension_root = (
            Path(args.ott_native_extension_root).resolve()
            if args.ott_native_extension_root
            else None
        )
        return source_root, extension_root
    raise ValueError(f"unsupported native KV baseline: {args.kv_cache_baseline}")


def _dataset_metadata(dataset: str, row: dict[str, Any]) -> dict[str, Any]:
    return get_benchmark(dataset).record_metadata(row)


def _load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoConfig, Qwen2_5OmniProcessor
    from r2omni.common.worldsense import force_text_decoder_eager, force_vision_segmented_sdpa

    from transformers import Qwen2_5OmniForConditionalGeneration

    model_path = Path(args.model_path).resolve()
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config.enable_talker = False
    config.enable_audio_output = False
    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    config.dtype = model_dtype
    if hasattr(config, "thinker_config"):
        config.thinker_config.dtype = model_dtype
    if hasattr(config, "talker_config"):
        config.talker_config.torch_dtype = torch.float32
    resolved_attention = resolve_text_attention(args)
    load_attention = (
        "flash_attention_2"
        if resolved_attention == "flash_attention_2"
        else "sdpa"
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        dtype=model_dtype,
        device_map="auto",
        attn_implementation=load_attention,
        local_files_only=True,
    )
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    if resolved_attention == "flash_attention_2":
        # flash-attn's standalone vision rotary helper launches sequence blocks
        # on CUDA grid-Y and fails once ceil(seq/8) exceeds 65,535.  Keep the
        # attention itself on FA2 and use an equivalent chunked torch rotary
        # only for those unusually large fixed-frame vision sequences.
        model._long_vision_rotary_fallback_calls = 0
        for module in model.modules():
            if type(module).__name__ != "Qwen2_5OmniVisionFlashAttention2":
                continue
            original_rotary = module._apply_rotary_pos_emb_flashatt

            def safe_vision_rotary(
                _self: Any,
                tensor: torch.Tensor,
                freqs: torch.Tensor,
                _original: Any = original_rotary,
            ) -> torch.Tensor:
                sequence_length = int(tensor.shape[1])
                if sequence_length <= 8 * 65535:
                    return _original(tensor, freqs)
                model._long_vision_rotary_fallback_calls += 1
                output = torch.empty_like(tensor)
                rotary_dim = int(freqs.shape[-1]) * 2
                chunk_size = 65536
                for start in range(0, sequence_length, chunk_size):
                    end = min(sequence_length, start + chunk_size)
                    value = tensor[:, start:end].float()
                    half = rotary_dim // 2
                    first = value[..., :half]
                    second = value[..., half:rotary_dim]
                    cos = freqs[start:end].cos()[None, :, None, :]
                    sin = freqs[start:end].sin()[None, :, None, :]
                    rotated = torch.cat(
                        (first * cos - second * sin, first * sin + second * cos),
                        dim=-1,
                    )
                    if rotary_dim < int(value.shape[-1]):
                        rotated = torch.cat(
                            (rotated, value[..., rotary_dim:]), dim=-1
                        )
                    output[:, start:end].copy_(rotated.to(tensor.dtype))
                return output

            module._apply_rotary_pos_emb_flashatt = MethodType(
                safe_vision_rotary, module
            )
    model.eval()
    # The upstream vision SDPA materializes a per-video block mask and attempts
    # a >300 GiB allocation at 128 frames on WorldSense. Keep the mathematically
    # equivalent segmented vision call for every backend; official_generate
    # otherwise retains the upstream text attention and GenerationMixin path.
    if resolved_attention != "flash_attention_2":
        force_vision_segmented_sdpa(model)
    if resolved_attention == "eager":
        force_text_decoder_eager(model, args.eager_query_chunk_size)
    if args.kv_cache_baseline != "none":
        from baselines.common.qwen_packed_adapter import (
            NativeKVAdapter,
            install_native_qwen_adapter,
        )

        baseline_root, extension_root = _native_kv_paths(args)
        native_adapter = NativeKVAdapter(
            method=args.kv_cache_baseline.removesuffix("-om"),
            baseline_root=baseline_root,
            native_extension_root=extension_root,
            kivi_implementation=args.kivi_implementation,
            k_bits=args.kv_k_bits,
            v_bits=args.kv_v_bits,
            group_size=args.kv_group_size,
            residual_length=args.kv_residual_length,
            fp_prefix_layers=args.kv_fp_prefix_layers,
            readout_backend=args.kv_readout_backend,
            sink_num=args.ott_sink_num,
            sink_max_size=args.ott_sink_max_size,
        )
        install_native_qwen_adapter(model, native_adapter)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, local_files_only=True)
    return model, processor


def _build_inputs(
    processor: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    media_cache: dict[str, Any] | None = None,
    prepared_media: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    from r2omni.common.worldsense import (
        build_processor_inputs,
    )
    media = prepared_media or _media_request_spec(row, args)
    messages = media["messages"]
    effective_audio_in_video = bool(media["use_audio_in_video"])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    cache_key = media["cache_key"]
    if media_cache is not None and media_cache.get("key") == cache_key:
        audios, images, videos, video_kwargs = media_cache["value"]
    else:
        decoded = (
            media
            if "value" in media
            else _decode_media_spec(media)
        )
        # A small number of official video clips legitimately contain no
        # embedded audio stream.  The decoder marks those requests as
        # video-only; propagate that effective setting to processor/generate.
        effective_audio_in_video = bool(decoded["use_audio_in_video"])
        audios, images, videos, video_kwargs = decoded["value"]
        if media_cache is not None:
            media_cache.clear()
            media_cache.update({
                "key": cache_key,
                "value": (audios, images, videos, video_kwargs),
            })
    inputs = build_processor_inputs(
        processor, text, audios, images, videos, video_kwargs, effective_audio_in_video
    )
    metadata = {
        "use_audio_in_video": effective_audio_in_video,
        "media_sampling": {
            "decoded_video_frames": [
                int(video[0].shape[0] if isinstance(video, tuple) else video.shape[0])
                for video in (videos or [])
            ],
            "sampled_video_fps": [
                float(value) for value in video_kwargs.get("fps", [])
            ],
            "frame_count_source": "decoded_video_tensor",
        },
    }
    return inputs, metadata


def _media_request_spec(
    row: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Build media metadata without decoding video/audio payloads."""
    dataset_root = Path(args.video_root).resolve()
    audio_root = Path(args.audio_root).resolve() if args.audio_root else None
    include_audio = not bool(getattr(args, "disable_audio", False))
    adapter = get_benchmark(args.dataset)
    messages = adapter.build_message(
        row,
        dataset_root,
        args.nframes,
        audio_root,
        include_audio=include_audio,
        video_resolution_policy=args.video_resolution_policy,
        video_fps=args.video_fps,
        max_video_frames=args.max_video_frames,
    )
    audio_path = (
        adapter.resolve_audio(dataset_root, row, audio_root)
        if include_audio
        else None
    )
    # video-SALMONN 2 stores the benchmark audio track inside each MP4.  Its
    # official task is explicitly audio-visual, so enable extraction unless an
    # experiment deliberately requests the visual-only control.
    effective_audio_in_video = bool(
        audio_path is None
        and (
            (adapter.embedded_audio and include_audio)
            or args.use_audio_in_video
        )
    )
    return {
        "messages": messages,
        "use_audio_in_video": effective_audio_in_video,
        "cache_key": (
            resolve_dataset_video_path(dataset_root, row, args.dataset),
            str(audio_path or ""),
            None if args.video_fps is not None else int(args.nframes),
            args.video_fps,
            args.max_video_frames,
            str(args.video_resolution_policy),
            effective_audio_in_video,
        ),
    }


def _decode_media_spec(spec: dict[str, Any]) -> dict[str, Any]:
    from qwen_omni_utils import process_mm_info

    try:
        value = process_mm_info(
            spec["messages"],
            use_audio_in_video=bool(spec["use_audio_in_video"]),
            return_video_kwargs=True,
        )
        return {**spec, "value": value}
    except AssertionError as exc:
        # OmniDCBench includes two valid clips without an audio stream.  Qwen's
        # media helper raises before decoding when audio extraction is forced;
        # retry those clips as video-only instead of dropping the sample.
        if not (
            bool(spec["use_audio_in_video"])
            and "Video must has audio track" in str(exc)
        ):
            raise
        value = process_mm_info(
            spec["messages"],
            use_audio_in_video=False,
            return_video_kwargs=True,
        )
        return {
            **spec,
            "use_audio_in_video": False,
            "audio_fallback": "no_embedded_audio_stream",
            "value": value,
        }


def _prepare_media_payload(
    row: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Decode one request's media without touching the shared processor/model."""
    return _decode_media_spec(_media_request_spec(row, args))


def _iter_prefetched_media(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    workers: int,
):
    """Yield bounded, in-order media futures while GPU work consumes requests."""
    if workers <= 0:
        for row in rows:
            yield row, None
        return
    pending: deque[tuple[dict[str, Any], Future]] = deque()
    source = iter(rows)
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="omni-media",
    ) as executor:
        for _ in range(min(len(rows), workers + 1)):
            row = next(source)
            pending.append((row, executor.submit(_prepare_media_payload, row, args)))
        while pending:
            row, future = pending.popleft()
            payload = future.result()
            try:
                next_row = next(source)
            except StopIteration:
                pass
            else:
                pending.append(
                    (
                        next_row,
                        executor.submit(_prepare_media_payload, next_row, args),
                    )
                )
            yield row, payload


def _strategy(args: argparse.Namespace, num_layers: int) -> dict[int, dict[str, Any]]:
    return build_layer_strategy(LayerStrategyConfig(
        mode="block",
        num_layers=num_layers,
        block_start_layer=args.block_start_layer,
        block_group_size=args.block_group_size,
        block_end_layer=args.block_end_layer,
    ))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def select_rows(
    rows: list[dict[str, Any]],
    *,
    start_idx: int,
    limit: int | None,
    sample_seed: int | None,
) -> list[dict[str, Any]]:
    selected = rows[start_idx:]
    if limit is None:
        return selected
    if sample_seed is None:
        return selected[:limit]
    return random.Random(sample_seed).sample(selected, k=min(limit, len(selected)))


def run(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_rows, media_stats = _load_dataset_rows(args)
    rows = select_rows(
        dataset_rows,
        start_idx=args.start_idx,
        limit=args.limit,
        sample_seed=None if args.sequential_selection else args.sample_seed,
    )
    if args.open_question:
        rows = [
            {**row, "question": args.open_question, "open_ended": True}
            for row in rows
        ]
    selected_sample_ids = [_sample_id(row, position) for position, row in enumerate(rows)]
    model, processor = _load_model(args)
    thinker = getattr(model, "thinker", model)
    device = next(thinker.parameters()).device
    num_layers = len(thinker.model.layers)
    stc_state: Any | None = None
    if args.stc_config:
        from r2omni.stc import install_stc
        from r2omni.stc.pre_llm_units import load_config as load_stc_config

        stc_state = install_stc(
            model,
            config=load_stc_config(args.stc_config),
            verbose=args.verbose,
        )
    manager: Any | None = None
    strategy_summary = {"mode": "baseline", "active_scope": "none", "enabled_layers": []}
    if args.backend == "r2omni":
        layer_strategy = _strategy(args, num_layers)
        strategy_summary = summarize_layer_strategy(layer_strategy)
        manager = QwenSSPAAttentionManager(
            model,
            layer_strategy,
            SSPAConfig(
                fp_append_capacity=args.max_new_tokens + 4,
                attention_weight=args.attention_weight,
                query_anchor_count=args.query_anchor_count,
                query_anchor_reduction=args.query_anchor_reduction,
                query_anchor_ema_decay=args.query_anchor_ema_decay,
                value_int2_group_size=args.value_int2_group_size,
                residual_allocation_rounds=args.residual_allocation_rounds,
                k_conditioned_iterations=args.k_conditioned_iterations,
                residual_query_cvar_fraction=args.residual_query_cvar_fraction,
                residual_transition_weight=args.residual_transition_weight,
                output_priority_weight=args.output_priority_weight,
                semantic_priority_weight=args.semantic_priority_weight,
                video_precision_weight=args.video_precision_weight,
                audio_precision_weight=args.audio_precision_weight,
                recent_fp_window=args.recent_fp_window,
                page_size=args.page_size,
                precision_sigmoid_steepness=args.precision_sigmoid_steepness,
                shallow_v_payload_bit_ceiling=(
                    args.shallow_v_payload_bit_ceiling
                ),
                deep_v_payload_bit_ceiling=args.deep_v_payload_bit_ceiling,
            ),
            kernel_backend="triton_mixed_fused_splitk",
            cache_variant=(
                "v_fourq_k_int2_compact"
                if args.r2_key_bits == 2
                else "v_fourq_k_int4_compact"
            ),
            fallback_on_error=False,
            verbose=args.verbose,
            validate_kernel_output=False,
            validate_policy_state=False,
            matched_fp_until_layer=args.matched_fp_until_layer,
            key_quantization_superblock=32,
            key_hadamard=True,
            value_hadamard=True,
            use_tensor_cores_qk=True,
            use_tensor_cores_pv=False,
            splitk_hierarchical_reduce=True,
            splitk_hierarchical_group_blocks=64,
        )
        manager.bind_stc_state(stc_state)
        manager.install()
    records: list[dict[str, Any]] = []
    predictions_path = output_dir / "predictions.jsonl"
    media_cache: dict[str, Any] | None = (
        {}
        if args.media_cache and args.media_prefetch_workers <= 0
        else None
    )
    try:
        with predictions_path.open("w", encoding="utf-8") as writer:
            for row_index, (row, prepared_media) in enumerate(
                _iter_prefetched_media(rows, args, args.media_prefetch_workers)
            ):
                sample_id = _sample_id(row, args.start_idx + row_index)
                record: dict[str, Any] = {
                    "sample_id": sample_id,
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "dataset": args.dataset,
                    "dataset_metadata": _dataset_metadata(args.dataset, row),
                    "backend": args.backend,
                    "strategy_name": "R2Omni" if manager is not None else args.backend,
                }
                try:
                    inputs, input_metadata = _build_inputs(
                        processor,
                        row,
                        args,
                        media_cache,
                        prepared_media=prepared_media,
                    )
                    processor_input_tokens = int(inputs["input_ids"].shape[-1])
                    if stc_state is not None:
                        from r2omni.common.worldsense import get_video_grid_thw

                        stc_state.prepare(
                            input_ids=inputs["input_ids"],
                            video_grid_thw=get_video_grid_thw(inputs),
                            video_merge_size=processor.video_processor.merge_size,
                        )
                    modality, modality_metadata = infer_qwen_omni_modality_vector(
                        inputs,
                        tokenizer=processor.tokenizer,
                        processor=processor,
                        model_config=thinker.config,
                        fallback_text_protect=True,
                    )
                    if manager is not None:
                        manager.prepare(modality, modality_metadata=modality_metadata, sample_id=sample_id)
                    if args.backend == "official_generate":
                        decode = official_model_generate_decode(
                            model, processor, dict(inputs),
                            max_new_tokens=args.max_new_tokens,
                            use_audio_in_video=input_metadata["use_audio_in_video"],
                            device=device, dtype=torch.bfloat16,
                            stop_on_eos=args.stop_on_eos,
                        )
                    else:
                        decode = r2omni_greedy_decode(
                            model,
                            processor,
                            dict(inputs),
                            max_new_tokens=args.max_new_tokens,
                            backend=args.backend,
                            use_audio_in_video=input_metadata["use_audio_in_video"],
                            device=device,
                            dtype=torch.bfloat16,
                            stc_state=stc_state,
                            prefill_logits_only=args.prefill_logits_only,
                            stop_on_eos=args.stop_on_eos,
                        )
                    manager_budget_stats = _quality_metadata(
                        manager.get_summary_stats() if manager is not None else {}
                    )
                    active_stats = (
                        manager.get_quality_stats() if manager is not None else {}
                    )
                    stc_stats = (
                        _quality_metadata(
                            dict(stc_state.last_metadata.get("summary", {}))
                        )
                        if stc_state is not None
                        else {}
                    )
                    final_cache = decode.pop("final_past_key_values")
                    kv_cache_budget = _kv_budget_record(
                        thinker=thinker,
                        processor_input_tokens=processor_input_tokens,
                        decode=decode,
                        stc_stats=stc_stats,
                        manager_budget_stats=manager_budget_stats,
                        final_cache=final_cache,
                        active_manager=manager is not None,
                    )
                    response = str(decode["output_text"])
                    prediction_mode = get_benchmark(args.dataset).prediction_mode
                    if prediction_mode == "text":
                        prediction = response
                        choice_token_text = response
                    elif prediction_mode == "full_choice":
                        # Match the official Daily-Omni evaluator: accept the
                        # first standalone A/B/C/D from the complete response.
                        from r2omni.common.worldsense import extract_choice

                        prediction = extract_choice(response)
                        choice_token_text = response
                    else:
                        prediction, choice_token_text = extract_first_generated_choice(
                            processor, decode["generated_ids"]
                        )
                    record.update({
                        "generated_token_ids": decode["generated_ids"],
                        "prediction": prediction,
                        "choice_token_text": choice_token_text,
                        "response": response,
                        "correct": (
                            None
                            if prediction_mode == "text"
                            else prediction == row.get("answer")
                        ),
                        "active_stats": active_stats,
                        "stc_stats": stc_stats,
                        "kv_cache_budget": _quality_metadata(kv_cache_budget),
                        "media_sampling": _quality_metadata(
                            input_metadata.get("media_sampling", {})
                        ),
                        "compact_metadata": _quality_metadata(
                            decode.get("compact_metadata", {})
                        ),
                        "error": None,
                    })
                    del final_cache, decode, inputs
                except Exception as exc:
                    record.update({
                        "prediction": "",
                        "response": "",
                        "correct": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    if args.traceback:
                        record["traceback"] = traceback.format_exc()
                    if args.raise_errors:
                        writer.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
                        writer.flush()
                        raise
                records.append(record)
                writer.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
                writer.flush()
                print(f"sample={sample_id} prediction={record.get('prediction')!r} correct={record.get('correct')} error={record.get('error')}")
    finally:
        if manager is not None:
            manager.uninstall()

    baseline = _load_baseline(args.baseline_predictions)
    accuracy = aggregate_accuracy(records, baseline)
    dataset_metrics = get_benchmark(args.dataset).aggregate(records)
    valid = [record for record in records if not record.get("error")]
    summary = {
        "backend": args.backend,
        "dataset": args.dataset,
        "model_path": str(Path(args.model_path).resolve()),
        "model_dtype": args.model_dtype,
        "dataset_media": media_stats,
        "dataset_metrics": dataset_metrics,
        "media_cache_enabled": args.media_cache,
        "layer_strategy": strategy_summary["mode"] if manager is not None else "none",
        "strategy_name": "R2Omni" if manager is not None else args.backend,
        "precision_mode": "sspa" if args.backend == "r2omni" else "none",
        "cache_variant": (
            f"int{args.r2_key_bits}_k_tiered_v"
            if args.backend == "r2omni"
            else "none"
        ),
        "matched_fp_until_layer": (
            args.matched_fp_until_layer
            if manager is not None and args.matched_fp_until_layer >= 0
            else None
        ),
        "key_quantization_superblock": (
            32 if manager is not None else None
        ),
        "key_hadamard": bool(manager is not None),
        "value_hadamard": bool(manager is not None),
        "precision_schedule": "sigmoid" if manager is not None else None,
        "precision_sigmoid_steepness": (
            args.precision_sigmoid_steepness if manager is not None else None
        ),
        "shallow_v_payload_bit_ceiling": (
            args.shallow_v_payload_bit_ceiling if manager is not None else None
        ),
        "deep_v_payload_bit_ceiling": (
            args.deep_v_payload_bit_ceiling if manager is not None else None
        ),
        "sspa_kernel_backend": (
            "triton_mixed_fused_splitk" if manager is not None else "none"
        ),
        "sspa_tensor_core_mode": "qk" if manager is not None else "none",
        "splitk_hierarchical_reduce": bool(manager is not None),
        "splitk_hierarchical_group_blocks": (
            64 if manager is not None else None
        ),
        "block_start_layer": (
            args.block_start_layer
            if manager is not None and strategy_summary["mode"] == "block"
            else None
        ),
        "block_end_layer": (
            args.block_end_layer
            if manager is not None and strategy_summary["mode"] == "block"
            else None
        ),
        "block_group_size": (
            args.block_group_size
            if manager is not None and strategy_summary["mode"] == "block"
            else None
        ),
        "page_size": args.page_size if manager is not None else None,
        "generated_cache_policy": (
            "chunked_recompress" if manager is not None else "none"
        ),
        "generated_value_precision": (
            "int2" if manager is not None else "none"
        ),
        "recent_fp_window": args.recent_fp_window if manager is not None else None,
        "recompress_chunk_size": (
            manager.assignment_config.recompress_chunk_size
            if manager is not None
            else None
        ),
        "prefill_logits_only": bool(
            args.prefill_logits_only
            and args.backend != "official_generate"
        ),
        "media_prefetch_workers": args.media_prefetch_workers,
        "processor_input_cache_dir": None,
        "processor_input_cache_hits": 0,
        "processor_input_cache_misses": 0,
        "attention_weight": args.attention_weight if args.backend == "r2omni" else None,
        "query_anchor_count": (
            args.query_anchor_count if args.backend == "r2omni" else None
        ),
        "query_anchor_reduction": (
            args.query_anchor_reduction if args.backend == "r2omni" else None
        ),
        "query_anchor_ema_decay": (
            args.query_anchor_ema_decay if args.backend == "r2omni" else None
        ),
        "value_int2_group_size": (
            args.value_int2_group_size if args.backend == "r2omni" else None
        ),
        "value_distortion_metric": (
            "k_conditioned_readout" if args.backend == "r2omni" else None
        ),
        "precision_allocator": (
            "residual_rate_distortion" if args.backend == "r2omni" else None
        ),
        "residual_allocation_rounds": (
            args.residual_allocation_rounds
            if args.backend == "r2omni"
            else None
        ),
        "k_conditioned_iterations": (
            args.k_conditioned_iterations
            if args.backend == "r2omni"
            else None
        ),
        "residual_query_objective": "cvar" if args.backend == "r2omni" else None,
        "residual_query_cvar_fraction": (
            args.residual_query_cvar_fraction
            if args.backend == "r2omni"
            else None
        ),
        "residual_transition_weight": (
            args.residual_transition_weight
            if args.backend == "r2omni"
            else None
        ),
        "output_priority_weight": (
            args.output_priority_weight if args.backend == "r2omni" else None
        ),
        "semantic_priority_weight": (
            args.semantic_priority_weight if args.backend == "r2omni" else None
        ),
        "video_precision_weight": (
            args.video_precision_weight
            if args.backend == "r2omni"
            else None
        ),
        "audio_precision_weight": (
            args.audio_precision_weight
            if args.backend == "r2omni"
            else None
        ),
        "stc_enabled": stc_state is not None,
        "stc_config": args.stc_config,
        "kv_cache_baseline": args.kv_cache_baseline,
        "native_kv_config": (
            {
                "k_bits": args.kv_k_bits,
                "v_bits": args.kv_v_bits,
                "group_size": args.kv_group_size,
                "residual_length": args.kv_residual_length,
                "readout_backend": args.kv_readout_backend,
                **(
                    {"implementation": f"{args.kivi_implementation}_qwen_adapter"}
                    if args.kv_cache_baseline == "kivi-om"
                    else {"implementation": "official_qwen_adapter"}
                ),
                **(
                    {"fp_prefix_layers": args.kv_fp_prefix_layers}
                    if args.kv_fp_prefix_layers
                    else {}
                ),
                "sink_num": args.ott_sink_num if args.kv_cache_baseline == "ott-om" else 0,
                "sink_max_size": (
                    args.ott_sink_max_size if args.kv_cache_baseline == "ott-om" else 0
                ),
            }
            if args.kv_cache_baseline != "none"
            else None
        ),
        "native_kv_source": (
            {"root": str(_native_kv_paths(args)[0])}
            if args.kv_cache_baseline != "none"
            else None
        ),
        "decode_driver": (
            "transformers_model_generate"
            if args.backend == "official_generate"
            else "controlled_greedy_loop"
        ),
        "audio_attention": (
            "transformers_flash_attention_2"
            if resolve_text_attention(args) == "flash_attention_2"
            else "transformers_sdpa"
        ),
        "vision_attention": (
            "transformers_flash_attention_2"
            if resolve_text_attention(args) == "flash_attention_2"
            else "segmented_pytorch_sdpa"
        ),
        "text_attention": (
            {
                "sdpa": "transformers_sdpa",
                "eager": "bounded_chunked_eager",
                "flash_attention_2": "transformers_flash_attention_2",
            }[resolve_text_attention(args)]
        ),
        "model_attention_implementation": getattr(
            thinker.config, "_attn_implementation", None
        ),
        "long_vision_rotary_fallback_calls": int(
            getattr(model, "_long_vision_rotary_fallback_calls", 0)
        ),
        "oscar_prefill_runtime": getattr(
            model, "_oscar_prefill_attention", None
        ),
        "oscar_decode_runtime": getattr(
            model, "_oscar_decode_attention", None
        ),
        "official_flash_attention_2_opt_in": bool(
            args.backend == "official_generate"
            and args.allow_official_flash_attention_2
            and resolve_text_attention(args) == "flash_attention_2"
        ),
        "active_scope": strategy_summary["active_scope"],
        **accuracy,
        "fallback_count": sum(int(record.get("active_stats", {}).get("fallback_count", 0)) for record in valid),
        "full_dequant_tensor_count": sum(
            int(record.get("active_stats", {}).get("full_dequant_tensor_count", 0)) for record in valid
        ),
        "fallback_layers": sorted({layer for record in valid for layer in record.get("active_stats", {}).get("fallback_layers", [])}),
        "stc_total_keep_ratio": _average(valid, ("stc_stats", "new_seq_len"))
        / max(_average(valid, ("stc_stats", "original_seq_len")), 1.0)
        if stc_state is not None else 1.0,
        "stc_video_keep_ratio": _average(valid, ("stc_stats", "video_kept_tokens"))
        / max(_average(valid, ("stc_stats", "video_original_tokens")), 1.0)
        if stc_state is not None else 1.0,
        "stc_audio_keep_ratio": _average(valid, ("stc_stats", "audio_kept_tokens"))
        / max(_average(valid, ("stc_stats", "audio_original_tokens")), 1.0)
        if stc_state is not None else 1.0,
        "kv_cache_budget": _aggregate_kv_budget(valid),
        "num_samples_requested": len(rows),
        "sample_seed": args.sample_seed,
        "selected_sample_ids": selected_sample_ids,
        "max_new_tokens": args.max_new_tokens,
        "stop_on_eos": args.stop_on_eos,
        "nframes": None if args.video_fps is not None else args.nframes,
        "video_fps": args.video_fps,
        "max_video_frames": args.max_video_frames,
        "video_resolution_policy": args.video_resolution_policy,
        "strategy": strategy_summary,
    }
    _write_json(output_dir / "result.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "WorldSense/Video-MME/Daily-Omni/video-SALMONN 2/LongVALE/"
            "OmniDCBench/AVSD baseline and "
            "FourQ benchmark."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=benchmark_names(),
        default="worldsense",
    )
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--ann", default=None)
    parser.add_argument("--video_root", default=None)
    parser.add_argument("--audio_root", default=None)
    parser.add_argument(
        "--available_media_only", action="store_true", default=False,
        help="evaluate only annotations whose local video file exists and report the omitted count",
    )
    parser.add_argument(
        "--media_cache", action=argparse.BooleanOptionalAction, default=True,
        help="reuse decoded media for adjacent questions from the same video",
    )
    parser.add_argument(
        "--media_prefetch_workers",
        type=int,
        default=0,
        help=(
            "bounded workers that decode future media while the GPU handles the "
            "current request; zero preserves the synchronous input path"
        ),
    )
    parser.add_argument("--stc_config", default=str(project_root() / "configs/stc.yaml"))
    parser.add_argument(
        "--kv_cache_baseline", choices=("none", "kivi-om", "ott-om"), default="none"
    )
    parser.add_argument(
        "--model_dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help=(
            "host-model dtype; float16 is the compatibility profile for "
            "KIVI-OM/OTT-OM because their released kernels were evaluated "
            "with FP16 model states"
        ),
    )
    parser.add_argument("--kv_k_bits", type=int, choices=(2, 4), default=2)
    parser.add_argument("--kv_v_bits", type=int, choices=(2, 4), default=2)
    parser.add_argument("--kv_group_size", type=int, default=128)
    parser.add_argument("--kv_residual_length", type=int, default=128)
    parser.add_argument("--kv_fp_prefix_layers", type=int, default=0)
    parser.add_argument(
        "--kv_readout_backend",
        choices=(
            "native_packed",
            "dequant_sdpa_reference",
            "dequant_fa2_reference",
        ),
        default="native_packed",
        help="reference mode materializes the cache and is not an efficiency result",
    )
    parser.add_argument(
        "--kivi_implementation",
        choices=("legacy", "official"),
        default="legacy",
        help=(
            "KIVI cache state machine; legacy is the original Qwen-compatible "
            "adapter used for the reported experiments"
        ),
    )
    parser.add_argument(
        "--kivi_legacy_root",
        default="",
        help="local KIVI tree used by the legacy Qwen-compatible adapter",
    )
    parser.add_argument(
        "--kivi_official_root",
        default="",
        help="read-only checkout of the official KIVI GitHub repository",
    )
    parser.add_argument(
        "--kivi_native_extension_root",
        default="",
        help="directory containing the unmodified official kivi_gemv build artifact",
    )
    parser.add_argument("--ott_sink_num", type=int, default=5)
    parser.add_argument("--ott_sink_max_size", type=int, default=32)
    parser.add_argument(
        "--ott_official_root",
        default="",
        help="read-only checkout of the official OTT GitHub repository",
    )
    parser.add_argument(
        "--ott_native_extension_root",
        default="",
        help="directory containing the unmodified official kivi_gemv build artifact",
    )
    parser.add_argument(
        "--backend",
        choices=("baseline", "official_generate", "r2omni"),
        default="r2omni",
    )
    parser.add_argument(
        "--prefill_logits_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "project only the final prefill hidden state through lm_head on the "
            "custom/manual generation path"
        ),
    )
    parser.add_argument("--block_start_layer", type=int, default=0)
    parser.add_argument("--block_end_layer", type=int, default=35)
    parser.add_argument("--block_group_size", type=int, default=4)
    parser.add_argument(
        "--r2_key_bits",
        type=int,
        choices=(2, 4),
        default=4,
        help="historical-key precision for the compact fused SSPA cache",
    )
    parser.add_argument("--page_size", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--precision_sigmoid_steepness", type=float, default=8.0)
    parser.add_argument("--shallow_v_payload_bit_ceiling", type=float, default=3.6)
    parser.add_argument("--deep_v_payload_bit_ceiling", type=float, default=3.0)
    parser.add_argument("--attention_weight", type=float, default=0.6)
    parser.add_argument("--query_anchor_count", type=int, default=8)
    parser.add_argument(
        "--query_anchor_reduction", choices=("mean", "max", "ema"), default="ema"
    )
    parser.add_argument("--query_anchor_ema_decay", type=float, default=0.8)
    parser.add_argument(
        "--value_int2_group_size", type=int, choices=(32, 64, 128), default=32
    )
    parser.add_argument("--residual_allocation_rounds", type=int, default=4)
    parser.add_argument("--k_conditioned_iterations", type=int, default=2)
    parser.add_argument("--residual_query_cvar_fraction", type=float, default=0.25)
    parser.add_argument("--residual_transition_weight", type=float, default=0.5)
    parser.add_argument("--output_priority_weight", type=float, default=0.25)
    parser.add_argument("--semantic_priority_weight", type=float, default=0.75)
    parser.add_argument("--video_precision_weight", type=float, default=0.6)
    parser.add_argument("--audio_precision_weight", type=float, default=1.8)
    parser.add_argument("--recent_fp_window", type=int, default=128)
    parser.add_argument(
        "--matched_fp_until_layer",
        type=int,
        default=-1,
        help=(
            "keep survivor K/V in BF16 through this zero-based layer, then use the "
            "configured low-bit cache variant in deeper groups; -1 disables"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate the full selected benchmark unless an explicit limit is set",
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--nframes", type=int, default=64)
    parser.add_argument(
        "--video_fps",
        type=float,
        default=None,
        help=(
            "sample videos at this FPS instead of using fixed --nframes; "
            "currently supported by video-SALMONN 2"
        ),
    )
    parser.add_argument(
        "--max_video_frames",
        type=int,
        default=None,
        help="maximum decoded frames when --video_fps is set",
    )
    parser.add_argument(
        "--video_resolution_policy",
        choices=("benchmark_cap", "qwen_native"),
        default="benchmark_cap",
        help=(
            "video-SALMONN 2 spatial policy: benchmark_cap preserves the "
            "fixed per-frame experiment cap; qwen_native removes experiment-"
            "level pixel overrides and uses the official Qwen processor limits"
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help=(
            "generation ceiling; by default use the official dataset profile "
            "(WorldSense/Video-MME 2048, Daily-Omni 10, "
            "videoSALMONN2/LongVALE 1024, OmniDCBench 8192, AVSD 20)"
        ),
    )
    parser.add_argument(
        "--stop_on_eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="allow open-ended generation to stop at the tokenizer EOS token",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=2024)
    parser.add_argument(
        "--sequential_selection",
        action="store_true",
        help=(
            "with --start_idx/--limit, select one contiguous dataset slice "
            "instead of the default seeded smoke-test sample"
        ),
    )
    parser.add_argument(
        "--open_question",
        default=None,
        help="replace the dataset MCQ with one shared open-ended generation prompt",
    )
    parser.add_argument("--use_audio_in_video", action="store_true", default=False)
    parser.add_argument("--eager_query_chunk_size", type=int, default=512)
    parser.add_argument(
        "--text_prefill_attention",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
        help=(
            "text attention used for native prefill; SSPA still owns active "
            "decode layers after cache finalization"
        ),
    )
    parser.add_argument(
        "--allow_official_flash_attention_2",
        action="store_true",
        default=False,
        help=(
            "explicit experiment-only opt-in allowing official model.generate "
            "to honor --text_prefill_attention flash_attention_2; the default "
            "official baseline remains SDPA"
        ),
    )
    parser.add_argument("--baseline_predictions", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--traceback", action="store_true", default=False)
    parser.add_argument("--raise_errors", action="store_true", default=False)
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = OFFICIAL_MAX_NEW_TOKENS[get_benchmark(args.dataset).name]
    workspace_root = resolve_workspace()
    default_paths = get_benchmark(args.dataset).paths(workspace_root)
    args.model_path = args.model_path or str(
        workspace_root / "Models" / "Qwen2.5-Omni-3B"
    )
    args.ann = args.ann or str(default_paths.annotation)
    args.video_root = args.video_root or str(default_paths.video_root)
    args.audio_root = args.audio_root or (
        None if default_paths.audio_root is None else str(default_paths.audio_root)
    )
    if args.media_prefetch_workers < 0:
        parser.error("--media_prefetch_workers must be non-negative")
    if (args.video_fps is None) != (args.max_video_frames is None):
        parser.error("--video_fps and --max_video_frames must be provided together")
    if args.video_fps is not None:
        if get_benchmark(args.dataset).name != "video_salmonn2":
            parser.error(
                "--video_fps/--max_video_frames currently support only video_salmonn2"
            )
        if args.video_fps <= 0 or args.max_video_frames <= 0:
            parser.error("--video_fps and --max_video_frames must be positive")
    if args.kv_cache_baseline != "none":
        if args.backend != "official_generate":
            parser.error("native KV baselines require --backend official_generate")
        if resolve_text_attention(args) != "flash_attention_2":
            parser.error(
                "native KV baselines require --text_prefill_attention "
                "flash_attention_2 --allow_official_flash_attention_2"
            )
        if args.kv_group_size <= 0 or args.kv_residual_length <= 0:
            parser.error("KV group/residual lengths must be positive")
        if args.kv_fp_prefix_layers < 0:
            parser.error("--kv_fp_prefix_layers must be non-negative")
        # OTT's released main configuration uses G=128 and R=32.  Its flush
        # rule quantizes only complete G-token groups beyond the R-token
        # window, so R does not need to be divisible by G.
        if (
            args.kv_cache_baseline == "kivi-om"
            and args.kv_residual_length % args.kv_group_size
        ):
            parser.error(
                "KIVI requires --kv_residual_length to be divisible by "
                "--kv_group_size"
            )
        if args.ott_sink_num < 0 or args.ott_sink_max_size < args.ott_sink_num:
            parser.error("invalid OTT sink sizes")
    return args


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = run(args, output_dir)
    except Exception as exc:
        failure = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        if args.traceback:
            failure["traceback"] = traceback.format_exc()
        _write_json(output_dir / "result.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed_count"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
