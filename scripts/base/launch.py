#!/usr/bin/env python3
"""Translate release YAML profiles into runner arguments without loading models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "3b": "qwen2.5-omni-3b",
    "7b": "qwen2.5-omni-7b",
    "qwen2.5-omni-3b": "qwen2.5-omni-3b",
    "qwen2.5-omni-7b": "qwen2.5-omni-7b",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def profile_for_model(model: Path, override: str | None = None) -> str:
    """Identify local Qwen weights by text geometry, never the directory name."""
    selected = None
    if override:
        try:
            selected = PROFILES[override.lower()]
        except KeyError as exc:
            raise ValueError("R2OMNI_MODEL_PROFILE must be 3b or 7b") from exc
    config_path = model / "config.json"
    detected = None
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        text = config.get("thinker_config", {}).get("text_config", {})
        geometry = (text.get("hidden_size"), text.get("num_hidden_layers"))
        detected = {(2048, 36): PROFILES["3b"], (3584, 28): PROFILES["7b"]}.get(geometry)
    if selected and detected and selected != detected:
        raise ValueError(f"Explicit profile {selected} conflicts with {config_path}: {detected}")
    if selected or detected:
        return selected or detected
    raise ValueError(
        f"Cannot identify Qwen2.5-Omni profile from {config_path}; "
        "set R2OMNI_MODEL_PROFILE=3b or 7b explicitly."
    )


def add_value(args: list[str], name: str, value: Any) -> None:
    args.extend((name, str(value)))


def require_fixed(data: dict[str, Any], expected: dict[str, Any], section: str) -> None:
    for key, value in expected.items():
        if key in data and data[key] != value:
            raise ValueError(f"{section}.{key} is fixed to {value!r} by this runner")


def evaluation_args(config: dict[str, Any], dataset: str, *, minicpm: bool = False) -> list[str]:
    require_fixed(config, {"processor_input_cache": False, "batch_size": 1, "decoding": "greedy"}, "evaluation")
    args: list[str] = []
    preset = config.get("benchmarks", {}).get(dataset, {})
    if minicpm:
        # MiniCPM's native chat path owns EOS and attention selection.
        require_fixed(config, {"stop_on_eos": True, "prefill_attention": "flash_attention_2"}, "evaluation")
    else:
        args.append("--stop_on_eos" if config.get("stop_on_eos", True) else "--no-stop_on_eos")
        add_value(args, "--text_prefill_attention", config.get("prefill_attention", "flash_attention_2"))
    if "max_new_tokens" in preset:
        add_value(args, "--max-new-tokens" if minicpm else "--max_new_tokens", preset["max_new_tokens"])
    sampling = preset.get("frame_sampling", "uniform")
    if dataset == "omnidcbench":
        # The official OmniDCBench loader enforces these source-data settings.
        require_fixed(preset, {"frame_sampling": "fps", "fps": 2, "max_frames": 160}, "omnidcbench")
        if minicpm:
            args.extend(("--video-fps", str(preset.get("fps", 2)), "--max-video-frames", str(preset.get("max_frames", 160))))
        else:
            args.extend(("--nframes", str(preset.get("max_frames", 160)), "--use_audio_in_video"))
    elif sampling == "uniform":
        if "frames" in preset:
            add_value(args, "--video-frames" if minicpm else "--nframes", preset["frames"])
    elif sampling == "fps":
        if dataset != "video_salmonn2":
            raise ValueError("FPS CLI overrides currently support only video_salmonn2")
        add_value(args, "--video-fps" if minicpm else "--video_fps", preset["fps"])
        add_value(args, "--max-video-frames" if minicpm else "--max_video_frames", preset["max_frames"])
    else:
        raise ValueError(f"Unknown frame_sampling: {sampling}")
    return args


def sspa_args(config: dict[str, Any]) -> list[str]:
    s = config["sspa"]
    require_fixed(s, {"enabled": True, "method": "SSPA", "backend": "r2omni", "cache_variant": "int4_k_tiered_v"}, "sspa")
    require_fixed(s["layers"], {"strategy": "block"}, "sspa.layers")
    require_fixed(s["scoring"], {"av_alignment_topk": 4}, "sspa.scoring")
    require_fixed(s["quantization"], {
        "key_all_historical_tokens": "int4", "key_scheme": "per_channel_affine_int4",
        "key_preconditioner": "hadamard_head_dim", "key_quantization_superblock": 32,
        "value_tiers": ["bf16", "int4", "int2"], "value_preconditioner": "hadamard_head_dim",
        "int4_value_scheme": "per_token_symmetric", "int2_value_scheme": "per_token_group_affine",
    }, "sspa.quantization")
    require_fixed(s["kernel"], {
        "backend": "triton_mixed_fused_splitk", "split_k_block_n": 0,
        "tensor_core_mode": "qk", "hierarchical_reduce": True, "hierarchical_group_blocks": 64,
    }, "sspa.kernel")
    args = ["--backend", "r2omni", "--r2_key_bits", "4"]
    mapping = {
        "layers": {"start_layer": "block_start_layer", "end_layer": "block_end_layer", "assignment_group_size": "block_group_size", "matched_fp_until_layer": "matched_fp_until_layer"},
        "scoring": {name: name for name in ("attention_weight", "query_anchor_count", "query_anchor_reduction", "query_anchor_ema_decay", "semantic_priority_weight")},
        "precision_allocation": {name: name for name in ("residual_allocation_rounds", "k_conditioned_iterations", "residual_query_cvar_fraction", "residual_transition_weight", "output_priority_weight", "video_precision_weight", "audio_precision_weight")},
        "cache": {name: name for name in ("page_size", "precision_sigmoid_steepness", "shallow_v_payload_bit_ceiling", "deep_v_payload_bit_ceiling", "recent_fp_window")},
        "quantization": {"int2_group_size": "value_int2_group_size"},
    }
    for section, fields in mapping.items():
        for key, flag in fields.items():
            if key in s[section]:
                add_value(args, "--" + flag, s[section][key])
    return args


def qwen_args(method: str, dataset: str, model: Path, output: Path, env: dict[str, str]) -> list[str]:
    config_dir = Path(env.get("R2OMNI_CONFIG_DIR", ROOT / "configs")).resolve()
    evaluation = load_yaml(Path(env.get("R2OMNI_EVALUATION_CONFIG", config_dir / "evaluation.yaml")))
    profile = profile_for_model(model, env.get("R2OMNI_MODEL_PROFILE"))
    args = ["--dataset", dataset, "--model_path", str(model), "--output_dir", str(output), "--media_prefetch_workers", "0", "--raise_errors"]
    args += evaluation_args(evaluation, dataset)
    if method == "r2omni":
        suffix = "7b" if profile.endswith("7b") else "3b"
        method_file = Path(env.get("R2OMNI_METHOD_CONFIG", config_dir / f"r2omni_{suffix}.yaml"))
        composite = load_yaml(method_file)
        require_fixed(composite, {"method": "R2-Omni", "execution_order": ["stc", "prefill", "sspa", "fused_low_bit_decode"]}, "method")
        sspa_file = Path(env.get("R2OMNI_SSPA_CONFIG", method_file.parent / composite["sspa"]))
        stc_file = Path(env.get("R2OMNI_STC_CONFIG", method_file.parent / composite["stc"]))
        args += ["--stc_config", str(stc_file.resolve())] + sspa_args(load_yaml(sspa_file))
    elif method in {"baseline", "kivi-om", "ott-om"}:
        args += ["--backend", "official_generate", "--stc_config", ""]
        attention = evaluation.get("prefill_attention", "flash_attention_2")
        if attention == "flash_attention_2":
            args.append("--allow_official_flash_attention_2")
        elif attention != "sdpa":
            raise ValueError("Official generation supports sdpa or flash_attention_2 prefill")
        if method != "baseline":
            method_file = Path(env.get("R2OMNI_METHOD_CONFIG", config_dir / f"{method}.yaml"))
            p = load_yaml(method_file)["profiles"][profile]
            args += ["--kv_cache_baseline", method]
            fields = {"k_bits": "kv_k_bits", "v_bits": "kv_v_bits", "group_size": "kv_group_size", "residual_length": "kv_residual_length", "fp_prefix_layers": "kv_fp_prefix_layers", "host_dtype": "model_dtype", "readout_backend": "kv_readout_backend", "implementation": "kivi_implementation", "sink_num": "ott_sink_num", "sink_max_size": "ott_sink_max_size"}
            for key, flag in fields.items():
                if key in p:
                    value = env.get("KV_OM_READOUT_BACKEND", p[key]) if key == "readout_backend" else p[key]
                    add_value(args, "--" + flag, value)
            third_party = Path(env.get("R2OMNI_THIRD_PARTY_DIR", ROOT / "third_party"))
            if method == "kivi-om":
                source = Path(env.get("KIVI_ROOT", third_party / "KIVI"))
                args += ["--kivi_legacy_root", str(source), "--kivi_official_root", str(source), "--kivi_native_extension_root", env.get("KIVI_EXTENSION_ROOT", str(source / "quant"))]
            else:
                source = Path(env.get("OTT_ROOT", third_party / "OTT"))
                args += ["--ott_official_root", str(source), "--ott_native_extension_root", env.get("OTT_EXTENSION_ROOT", str(source / "quant"))]
    else:
        raise ValueError(f"Unknown method: {method}")
    return args


def minicpm_args(method: str, dataset: str, model: Path, output: Path, env: dict[str, str]) -> list[str]:
    config_dir = Path(env.get("R2OMNI_CONFIG_DIR", ROOT / "configs")).resolve()
    config_file = Path(env.get("R2OMNI_MINICPM_CONFIG", config_dir / "minicpmo-4.5.yaml"))
    config = load_yaml(config_file)
    require_fixed(config.get("prefill", {}), {"llm": "flash_attention_2", "vision": "flash_attention_2", "audio": "sdpa"}, "minicpm.prefill")
    evaluation = load_yaml(Path(env.get("R2OMNI_EVALUATION_CONFIG", config_dir / "evaluation.yaml")))
    args = ["--workspace", env.get("R2OMNI_WORKSPACE", str(ROOT)), "--model", str(model), "--method", method, "--output-dir", str(output), "--dataset", "videosalmonn2" if dataset == "video_salmonn2" else dataset]
    args += evaluation_args(evaluation, dataset, minicpm=True)
    if method == "r2omni":
        p = config["r2omni"]
        fields = {"layer_group_size": "r2-layer-group-size", "matched_fp_until_layer": "r2-matched-fp-until-layer", "shallow_v_payload_bit_ceiling": "r2-shallow-v-bits", "deep_v_payload_bit_ceiling": "r2-deep-v-bits", "value_int2_group_size": "r2-group-size"}
        if p.get("key_quantization_superblock") != p.get("value_int2_group_size"):
            raise ValueError("MiniCPM key_quantization_superblock must equal value_int2_group_size")
        for key, flag in fields.items():
            if key in p and p[key] is not None:
                add_value(args, "--" + flag, p[key])
        add_value(args, "--r2-stc-config", Path(env.get("R2OMNI_STC_CONFIG", config_file.parent / p["stc"])).resolve())
    elif method in {"kivi-om", "ott-om"}:
        p = config[method]
        fields = {"k_bits": "kv-k-bits", "v_bits": "kv-v-bits", "group_size": "kv-group-size", "residual_length": "kv-residual-length", "fp_prefix_layers": "kv-fp-prefix-layers", "readout_backend": "kv-readout-backend", "implementation": "kivi-implementation", "sink_num": "ott-sink-num", "sink_max_size": "ott-sink-max-size"}
        for key, flag in fields.items():
            if key in p:
                add_value(args, "--" + flag, p[key])
    elif method == "oscar":
        p = config["oscar"]
        require_fixed(p, {"quant_mode": "k-channel", "residual_block_size": 128, "offline_v_hadamard": True, "decode_backend": "official_cuda_packed"}, "minicpm.oscar")
        for key in ("num_bits", "group_size", "kv_rotation", "kv_norm"):
            if key in p:
                add_value(args, "--oscar-" + key.replace("_", "-"), p[key])
    elif method != "bf16":
        raise ValueError(f"Unknown MiniCPM method: {method}")
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_kind", choices=("qwen", "minicpm"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", default="video_salmonn2")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    known, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    # MiniCPM's public shell entry accepts --dataset among runner overrides.
    # Select its evaluation preset before forwarding the original arguments.
    dataset = known.dataset
    for index, arg in enumerate(extra):
        if arg == "--dataset" and index + 1 < len(extra):
            dataset = extra[index + 1]
        elif arg.startswith("--dataset="):
            dataset = arg.split("=", 1)[1]
    if dataset == "videosalmonn2":
        dataset = "video_salmonn2"
    env = dict(os.environ)
    model = known.model.expanduser().resolve()
    output = known.output.expanduser().resolve()
    try:
        if known.model_kind == "qwen":
            module = "benchmarks.benchmark_mixed_active"
            args = qwen_args(known.method, dataset, model, output, env)
        else:
            module = "benchmarks.run_minicpmo_videosalmonn2"
            args = minicpm_args(known.method, dataset, model, output, env)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    command = [sys.executable, "-m", module, *args, *extra]
    if env.get("R2OMNI_DRY_RUN") == "1":
        print(json.dumps({"cwd": str(ROOT), "argv": command}, indent=2))
        return 0
    print("Launching: " + shlex.join(command), file=sys.stderr)
    os.chdir(ROOT)
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
