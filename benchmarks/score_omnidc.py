"""OmniDCBench temporal and semantic scoring entry point."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from benchmarks.api_utils import resolve_api_credentials


DIM_KEYS = (
    "segment_detail_caption",
    "video_background",
    "acoustics_content",
    "shooting_style",
    "speech_content",
    "camera_state",
)


def _load_official(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("omnidcbench_eval_time", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_api_pair(path: Path | None, pair: int) -> tuple[str, str]:
    base_url, api_key = resolve_api_credentials(path, pair)
    if base_url.rstrip("/") == "https://dashscope.aliyuncs.com/api/v1":
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return base_url.rstrip("/"), api_key


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class TemporalOnlyScorer:
    DIM_KEYS = DIM_KEYS

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def compute_score(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("semantic scorer must not run in temporal-only mode")


class QwenChecklistScorer:
    DIM_KEYS = DIM_KEYS

    def __init__(
        self,
        *_: Any,
        client: Any,
        model: str,
        max_output_tokens: int,
        judgment_dir: Path,
        **__: Any,
    ) -> None:
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.judgment_dir = judgment_dir
        self.judgment_dir.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    @staticmethod
    def _norm_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _normalize_gt(self, raw_gt: Any) -> dict[str, list[str]]:
        if isinstance(raw_gt, dict) and isinstance(raw_gt.get("keypoints"), dict):
            return {
                dim: [
                    self._norm_text(item)
                    for item in (
                        raw_gt["keypoints"].get(dim, {}).get("gt_keypoints", [])
                        or []
                    )
                    if self._norm_text(item)
                ]
                for dim in self.DIM_KEYS
            }
        aliases = {
            dim: (dim, f"{dim}_en")
            for dim in self.DIM_KEYS
        }
        result: dict[str, list[str]] = {}
        for dim, keys in aliases.items():
            text = next(
                (self._norm_text(raw_gt.get(key)) for key in keys if raw_gt.get(key)),
                "",
            ) if isinstance(raw_gt, dict) else ""
            result[dim] = [
                item.strip()
                for item in re.split(r"[.!?]", text)
                if item.strip()
            ]
        return result

    def _prompt(self, gt: dict[str, list[str]], prediction: str) -> str:
        payload = {"ground_truth_keypoints": gt, "candidate_caption": prediction}
        return (
            "For each ground-truth keypoint, decide whether its meaning is stated "
            "or clearly implied by the candidate caption. Return JSON in this shape: "
            '{"by_dim":{"segment_detail_caption":{"correct_count":0},'
            '"video_background":{"correct_count":0},'
            '"acoustics_content":{"correct_count":0},'
            '"shooting_style":{"correct_count":0},'
            '"speech_content":{"correct_count":0},'
            '"camera_state":{"correct_count":0}}}. '
            "Input:\n"
            + json.dumps(payload, ensure_ascii=False)
        )

    def _judge(
        self,
        gt: dict[str, list[str]],
        prediction: str,
        sample_name: str,
        cache_key: str,
    ) -> dict[str, Any]:
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_key)
        path = self.judgment_dir / f"{safe_key}.json"
        identity = {
            "model": self.model,
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(json.dumps({"gt": gt, "prediction": prediction}, sort_keys=True).encode()).hexdigest(),
        }
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("judge_identity") == identity:
                return cached
        prompt = self._prompt(gt, prediction)
        last_error: Exception | None = None
        for attempt in range(1, 4):
            content = None
            finish_reason = None
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_completion_tokens=self.max_output_tokens,
                )
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                parsed = json.loads(content or "")
                by_dim = parsed.get("by_dim")
                if not isinstance(by_dim, dict):
                    raise ValueError("judge response lacks by_dim")
                normalized: dict[str, dict[str, int]] = {}
                for dim in self.DIM_KEYS:
                    value = by_dim.get(dim, {})
                    count = int(value.get("correct_count", -1))
                    if count < 0 or count > len(gt[dim]):
                        raise ValueError(f"invalid correct_count for {dim}: {count}")
                    normalized[dim] = {"correct_count": count}
                record = {
                    "sample_name": sample_name,
                    "model": self.model,
                    "prediction": prediction,
                    "by_dim": normalized,
                    "judge_identity": identity,
                }
                with self._write_lock:
                    path.write_text(
                        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                return record
            except Exception as exc:
                last_error = exc
                failure = {"sample_name": sample_name, "model": self.model,
                           "attempt": attempt, "raw_response": content,
                           "finish_reason": finish_reason,
                           "error": f"{type(exc).__name__}: {exc}"}
                with self._write_lock:
                    with path.with_suffix(".failures.jsonl").open("a", encoding="utf-8") as writer:
                        writer.write(json.dumps(failure, ensure_ascii=False) + "\n")
                if attempt < 3:
                    time.sleep(2.0 ** (attempt - 1))
        raise RuntimeError(f"semantic judge failed after retries: {last_error}")

    def compute_score(
        self,
        res: dict[str, list[str]],
        gts: dict[str, Any],
        sample_names: dict[str, str] | None = None,
    ) -> tuple[float, list[int], dict[str, dict[str, Any]]]:
        total_correct = 0
        total_gt = 0
        correct_counts: list[int] = []
        details: dict[str, dict[str, Any]] = {}
        for index in res:
            prediction = self._norm_text(res[index][0] if res[index] else "")
            gt = self._normalize_gt(gts[index])
            sample_name = (sample_names or {}).get(index, index)
            judged = self._judge(gt, prediction, sample_name, str(index))
            by_dim_ratio: dict[str, float] = {}
            sample_correct = 0
            sample_total = 0
            for dim in self.DIM_KEYS:
                correct = int(judged["by_dim"][dim]["correct_count"])
                count = len(gt[dim])
                sample_correct += correct
                sample_total += count
                by_dim_ratio[dim] = correct / count if count else 0.0
            total_correct += sample_correct
            total_gt += sample_total
            correct_counts.append(sample_correct)
            details[index] = {
                "overall": sample_correct / sample_total if sample_total else 0.0,
                "by_dim": by_dim_ratio,
            }
        return total_correct / max(total_gt, 1), correct_counts, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("temporal", "qwen"), required=True)
    parser.add_argument("--official-script", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--api-file", type=Path)
    parser.add_argument("--api-pair", type=int, default=1)
    parser.add_argument("--model", help="Required in qwen mode: judge model served by your selected API")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.workers < 1 or args.max_output_tokens < 1:
        parser.error("workers and max-output-tokens must be positive")
    if args.mode == "qwen" and not args.model:
        parser.error("--model is required in qwen mode")
    module = _load_official(args.official_script.resolve())
    normalize_segments = module._normalize_from_gt_like_ann

    def normalize_list_only(segments: Any) -> list[dict[str, Any]]:
        if not isinstance(segments, list):
            return []
        return normalize_segments(segments)

    module._normalize_from_gt_like_ann = normalize_list_only
    args.score_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "temporal":
        module.Checklist_Score = TemporalOnlyScorer
        result = module.eval_with_files(
            str(args.predictions),
            str(args.ground_truth),
            tmponly=True,
            max_workers=args.workers,
            cache_dir=str(args.score_dir / "unused_temporal_cache"),
            output_dir=str(args.score_dir),
        )
        result["evaluation_mode"] = "official_temporal_only"
    else:
        from openai import OpenAI

        base_url, api_key = _load_api_pair(args.api_file, args.api_pair)
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=0)

        def scorer_factory(*positional: Any, **keywords: Any) -> QwenChecklistScorer:
            return QwenChecklistScorer(
                *positional,
                client=client,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
                judgment_dir=args.score_dir / "qwen_judgments",
                **keywords,
            )

        module.Checklist_Score = scorer_factory
        result = module.eval_with_files(
            str(args.predictions),
            str(args.ground_truth),
            tmponly=False,
            max_workers=args.workers,
            cache_dir=str(args.score_dir / "qwen_judgments"),
            output_dir=str(args.score_dir),
        )
        result["semantic_judge"] = args.model
        result["evaluation_mode"] = "official_sodam_with_qwen_checklist"
    parsed_predictions = module.load_pred_file(str(args.predictions))
    result["valid_parsed_video_num"] = sum(
        bool(segments) for segments in parsed_predictions.values()
    )
    result = _jsonable(result)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
