"""Release launchers must honor configs and remain independent of machine paths."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_launch", ROOT / "scripts/base/launch.py")
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


def model_config(tmp_path: Path, layers: int = 36, hidden: int = 2048) -> Path:
    model = tmp_path / "renamed model weights"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "thinker_config": {"text_config": {"num_hidden_layers": layers, "hidden_size": hidden}}
    }))
    return model


def cli_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def runner_flags(path: Path) -> set[str]:
    flags = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                    flags.add(arg.value)
    return flags


def test_model_profile_uses_geometry_after_renaming(tmp_path: Path) -> None:
    model = model_config(tmp_path, layers=28, hidden=3584)
    assert launch.profile_for_model(model) == "qwen2.5-omni-7b"
    args = launch.qwen_args("r2omni", "video_salmonn2", model, tmp_path / "out", {})
    assert cli_value(args, "--block_end_layer") == "27"
    assert cli_value(args, "--matched_fp_until_layer") == "1"
    with pytest.raises(ValueError, match="conflicts"):
        launch.profile_for_model(model, "3b")


def test_explicit_profile_allows_inspecting_without_weights(tmp_path: Path) -> None:
    assert launch.profile_for_model(tmp_path / "absent", "3b") == "qwen2.5-omni-3b"
    with pytest.raises(ValueError, match="Cannot identify"):
        launch.profile_for_model(tmp_path / "absent")


def test_yaml_edits_change_sspa_evaluation_and_baseline_cli(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", config_dir)
    evaluation = yaml.safe_load((config_dir / "evaluation.yaml").read_text())
    evaluation["benchmarks"]["video_salmonn2"]["frames"] = 48
    evaluation["benchmarks"]["video_salmonn2"]["max_new_tokens"] = 256
    (config_dir / "evaluation.yaml").write_text(yaml.safe_dump(evaluation))
    sspa = yaml.safe_load((config_dir / "sspa_3b.yaml").read_text())
    sspa["sspa"]["cache"]["page_size"] = 64
    sspa["sspa"]["scoring"]["semantic_priority_weight"] = 0.25
    (config_dir / "sspa_3b.yaml").write_text(yaml.safe_dump(sspa))
    kivi = yaml.safe_load((config_dir / "kivi-om.yaml").read_text())
    kivi["profiles"]["qwen2.5-omni-3b"]["residual_length"] = 1792
    (config_dir / "kivi-om.yaml").write_text(yaml.safe_dump(kivi))
    env = {"R2OMNI_CONFIG_DIR": str(config_dir)}
    model = model_config(tmp_path)
    args = launch.qwen_args("r2omni", "video_salmonn2", model, tmp_path / "out", env)
    assert cli_value(args, "--nframes") == "48"
    assert cli_value(args, "--max_new_tokens") == "256"
    assert cli_value(args, "--page_size") == "64"
    assert cli_value(args, "--semantic_priority_weight") == "0.25"
    args = launch.qwen_args("kivi-om", "video_salmonn2", model, tmp_path / "out", env)
    assert cli_value(args, "--kv_residual_length") == "1792"
    assert cli_value(args, "--kivi_legacy_root") == str(ROOT / "third_party/KIVI")


def test_unsupported_fixed_settings_fail_instead_of_being_ignored() -> None:
    config = yaml.safe_load((ROOT / "configs/sspa_3b.yaml").read_text())
    config["sspa"]["kernel"]["backend"] = "invented_kernel"
    with pytest.raises(ValueError, match="sspa.kernel.backend is fixed"):
        launch.sspa_args(config)


@pytest.mark.parametrize("method", ["baseline", "r2omni", "kivi-om", "ott-om"])
def test_generated_qwen_flags_exist_in_runner(tmp_path: Path, method: str) -> None:
    args = launch.qwen_args(method, "video_salmonn2", model_config(tmp_path), tmp_path / "out", {})
    declared = runner_flags(ROOT / "benchmarks/benchmark_mixed_active.py")
    assert {x for x in args if x.startswith("--")} <= declared


@pytest.mark.parametrize("method", ["bf16", "r2omni", "oscar", "kivi-om", "ott-om"])
def test_generated_minicpm_flags_exist_in_runner(tmp_path: Path, method: str) -> None:
    args = launch.minicpm_args(method, "video_salmonn2", tmp_path / "weights", tmp_path / "out", {})
    declared = runner_flags(ROOT / "benchmarks/run_minicpmo_videosalmonn2.py")
    assert {x for x in args if x.startswith("--")} <= declared


def test_shell_launcher_preserves_paths_and_last_cli_override(tmp_path: Path) -> None:
    env = {**os.environ, "MODEL_PATH": str(model_config(tmp_path)), "R2OMNI_DRY_RUN": "1"}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/eval_qwen2_5_omni_r2omni.sh"), "video_salmonn2", "output with spaces", "--page_size", "64"],
        env=env, cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    command = json.loads(result.stdout)
    assert command["argv"][-2:] == ["--page_size", "64"]
    assert cli_value(command["argv"], "--output_dir") == str(tmp_path / "output with spaces")
    assert command["cwd"] == str(ROOT)


def test_no_processor_cache_in_public_configs() -> None:
    evaluation = yaml.safe_load((ROOT / "configs/evaluation.yaml").read_text())
    assert evaluation["processor_input_cache"] is False
    runner = (ROOT / "benchmarks/benchmark_mixed_active.py").read_text()
    assert 'parser.add_argument("--processor_input_cache_dir"' not in runner


def test_minicpm_dataset_override_selects_corresponding_budget(tmp_path: Path) -> None:
    env = {**os.environ, "MODEL_PATH": str(tmp_path / "weights"), "R2OMNI_DRY_RUN": "1"}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/eval_minicpmo_4_5.sh"), "r2omni", str(tmp_path / "out"), "--dataset", "longvale"],
        env=env, cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    args = json.loads(result.stdout)["argv"]
    assert cli_value(args, "--video-frames") == "100"
    assert cli_value(args, "--max-new-tokens") == "1024"


@pytest.mark.parametrize("method", ["bf16", "r2omni", "oscar", "kivi-om", "ott-om"])
def test_minicpm_omnidcbench_supplies_required_sampling_and_shared_output_budget(
    tmp_path: Path, method: str,
) -> None:
    args = launch.minicpm_args(method, "omnidcbench", tmp_path / "weights", tmp_path / "out", {})
    assert cli_value(args, "--dataset") == "omnidcbench"
    assert cli_value(args, "--video-fps") == "2"
    assert cli_value(args, "--max-video-frames") == "160"
    assert cli_value(args, "--max-new-tokens") == "8192"
    assert "--video-frames" not in args
