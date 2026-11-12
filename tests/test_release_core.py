"""Portable paths and evaluator failure handling; no models, GPUs, or API calls."""

from __future__ import annotations

import json
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stdout

import torch

from benchmarks import benchmark_mixed_active as runner
from benchmarks import evaluate_videosalmonn2_gpt as scorer
from benchmarks import run_minicpmo_videosalmonn2 as minicpm
from benchmarks import score_omnidc
from benchmarks.api_utils import resolve_api_credentials
from r2omni.common import paths
from r2omni.common.longvale import normalize_longvale_row
from r2omni.sspa.integration.sspa_config import SSPAConfig


class ReleaseCoreTest(unittest.TestCase):
    def test_cli_help_returns_before_models_credentials_or_media(self):
        for entry in (runner.parse_args, minicpm.main, scorer.main, score_omnidc.main):
            with self.subTest(entry=entry.__module__), patch.object(sys, "argv", ["runner", "--help"]), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    entry()
                self.assertEqual(result.exception.code, 0)

    def test_workspace_and_stc_defaults_do_not_depend_on_cwd(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with patch.object(sys, "argv", ["runner", "--dataset", "video_salmonn2", "--output_dir", "out"]):
                    args = runner.parse_args()
                self.assertEqual(paths.resolve_workspace(), paths.project_root())
                self.assertEqual(Path(args.stc_config), paths.project_root() / "configs/stc.yaml")
                self.assertEqual(Path(args.ann), paths.project_root() / "Benchmarks/videoSALMONN2/video_salmonn2_test.json")
            finally:
                os.chdir(previous)

    def test_third_party_prefers_checkout_then_legacy_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, workspace = root / "checkout", root / "workspace"
            legacy = workspace / "Baselines/KIVI"
            legacy.mkdir(parents=True)
            with patch.object(paths, "project_root", return_value=checkout):
                self.assertEqual(paths.resolve_third_party("KIVI", workspace), legacy)
                local = checkout / "third_party/KIVI"
                local.mkdir(parents=True)
                self.assertEqual(paths.resolve_third_party("KIVI", workspace), local)

    def test_backing_storage_retained_by_views_is_disclosed(self):
        key, value = torch.zeros(32), torch.zeros(32)
        cache = SimpleNamespace(key_cache=[key[:4]], value_cache=[value[:4]])
        thinker = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1, hidden_size=4, num_attention_heads=1))
        budget = runner._kv_budget_record(
            thinker=thinker, processor_input_tokens=4,
            decode={"generated_ids": torch.zeros((1, 1), dtype=torch.long)},
            stc_stats={}, manager_budget_stats={}, final_cache=cache, active_manager=False,
        )
        self.assertEqual(budget["active_persistent_cache_bytes"], 32)
        self.assertEqual(budget["allocated_persistent_cache_bytes"], 256)
        self.assertEqual(budget["storage_breakdown"]["allocated_persistent_cache_bytes"], 256)
        aggregate = runner._aggregate_kv_budget([{"kv_cache_budget": budget}])
        self.assertEqual(aggregate["allocated_persistent_cache_bytes_total"], 256)

    def test_explicit_minicpm_helper_does_not_require_original_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "utils.py"
            helper.write_text("def get_video_frame_audio_segments(*args): return 'video'\ndef get_audio_segments(*args): return 'audio'\n")
            video, audio = minicpm._media_loaders(Path(directory), helper)
            self.assertEqual(video(), "video")
            self.assertEqual(audio(), "audio")

    def test_minicpm_kv_flags_reach_native_adapter(self):
        install = Mock(return_value=SimpleNamespace(new_cache=lambda: None))
        bridge = Mock()
        fake_module = SimpleNamespace(install_kivi_om=install)
        with patch.dict(sys.modules, {"baselines.kivi_om": fake_module}), patch.object(minicpm, "MiniCPMGenerationAdapter", bridge):
            minicpm._install_method(object(), "kivi-om", Path.cwd(), 10,
                                    kv_k_bits=4, kv_v_bits=4, kv_group_size=32,
                                    kv_residual_length=256, kv_fp_prefix_layers=2,
                                    kivi_implementation="legacy")
        options = install.call_args.kwargs
        self.assertEqual((options["k_bits"], options["v_bits"], options["group_size"]), (4, 4, 32))
        self.assertEqual(options["fp_prefix_layers"], 2)
        self.assertEqual(options["residual_length"], 256)
        self.assertEqual(options["implementation"], "legacy")

    def test_environment_credentials_take_precedence(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unit-test-value", "OPENAI_BASE_URL": "https://example.invalid/v1/"}, clear=True):
            self.assertEqual(resolve_api_credentials(Path("missing-credentials")), ("https://example.invalid/v1", "unit-test-value"))
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                resolve_api_credentials()

    def test_prediction_validation_and_explicit_subset_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions, annotation = Path(directory) / "predictions.jsonl", Path(directory) / "annotation.json"
            annotation.write_text(json.dumps([{"video": name, "events": ["event"]} for name in ("a.mp4", "b.mp4")]))
            def write(rows):
                predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
            write([{"sample_id": "a.mp4", "response": "caption"}])
            with self.assertRaisesRegex(ValueError, "allow-subset"):
                scorer.load_inputs(predictions, annotation)
            coverage = {}
            self.assertEqual(len(scorer.load_inputs(predictions, annotation, allow_subset=True, coverage=coverage)), 1)
            self.assertEqual(coverage, {"annotation_total": 2, "prediction_total": 1, "missing_prediction_count": 1})
            for rows, message in [
                ([{"sample_id": "a.mp4", "response": "caption"}] * 2, "Duplicate"),
                ([{"sample_id": "unknown.mp4", "response": "caption"}], "absent"),
                ([{"sample_id": "a.mp4", "response": "", "error": "failed generation"}], "generation error"),
                ([{"sample_id": "a.mp4", "response": "   "}], "empty"),
            ]:
                write(rows)
                with self.assertRaisesRegex(ValueError, message):
                    scorer.load_inputs(predictions, annotation, allow_subset=True)

    def test_invalid_judge_keeps_raw_responses_and_has_finite_retries(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"type":"object"}'), finish_reason="stop")])
        create = Mock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(scorer.time, "sleep"):
            row = scorer.judge_one(client, {"sample_id": "sample", "base_events": ["event"], "reference_caption": "reference", "prediction": "candidate"},
                                   model="example-model", api_backend="chat-completions", reasoning_effort=None, max_output_tokens=256, max_retries=2)
        self.assertEqual(create.call_count, 2)
        self.assertIsNotNone(row["error"])
        self.assertEqual([attempt["raw_response"] for attempt in row["failed_attempts"]], ['{"type":"object"}'] * 2)
        self.assertEqual(scorer.aggregate([row])["status"], "JUDGE_INCOMPLETE")
        self.assertIsNone(scorer.aggregate([row])["correct_rate"])

    def test_incomplete_scoring_does_not_report_successful_subset_as_headline(self):
        valid = {"error": None, "judgment": {"counts": {"Correct": 1, "Missed": 0, "Incorrect": 0, "Hallucination": 0}}}
        self.assertEqual(scorer.aggregate([valid])["correct_rate"], 1.0)
        partial = scorer.aggregate([valid, {"error": "invalid format"}])
        self.assertIsNone(partial["correct_rate"])
        self.assertFalse(partial["headline_metrics_available"])

    def test_longvale_and_sspa_invalid_configuration_fail_early(self):
        with self.assertRaisesRegex(ValueError, "mismatched"):
            normalize_longvale_row("video", {"timestamps": [[0, 1]], "sentences": []}, 0)
        with self.assertRaisesRegex(ValueError, "V payload ceilings"):
            SSPAConfig(shallow_v_payload_bit_ceiling=2.0, deep_v_payload_bit_ceiling=3.0)


if __name__ == "__main__":
    unittest.main()
