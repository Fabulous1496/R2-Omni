from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel
from benchmarks.api_utils import resolve_api_credentials


LABELS = ("Correct", "Missed", "Incorrect")
SYSTEM_PROMPT = """Evaluate a candidate video caption against the supplied
reference events. Label every reference event exactly once as Correct, Missed,
or Incorrect. Also list distinct claims in the candidate that are unsupported
by either the events or the reference caption. Judge meaning rather than exact
wording, and keep the evidence brief."""


class EventJudgment(BaseModel):
    event_index: int
    label: Literal["Correct", "Missed", "Incorrect"]
    evidence: str


class Hallucination(BaseModel):
    claim: str
    rationale: str


class CaptionJudgment(BaseModel):
    base_event_judgments: list[EventJudgment]
    hallucinations: list[Hallucination]
    summary: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_api_credentials(path: Path | None = None, pair: int = 1) -> tuple[str, str]:
    return resolve_api_credentials(path, pair)


def chat_compatible_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.netloc == "dashscope.aliyuncs.com" and parsed.path.rstrip("/") == "/api/v1":
        return f"{parsed.scheme}://{parsed.netloc}/compatible-mode/v1"
    return base_url.rstrip("/")


def load_inputs(
    predictions_path: Path,
    annotation_path: Path,
    *,
    allow_subset: bool = False,
    coverage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    prediction_rows = _read_jsonl(predictions_path)
    predictions: dict[str, str] = {}
    for row in prediction_rows:
        if "sample_id" not in row:
            raise ValueError("Each prediction must include sample_id")
        sample_id = str(row["sample_id"])
        if sample_id in predictions:
            raise ValueError(f"Duplicate prediction sample_id: {sample_id}")
        if row.get("error"):
            raise ValueError(f"Prediction {sample_id} has a generation error; resolve it before scoring")
        caption = row.get("response", row.get("prediction"))
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Prediction {sample_id} has an empty or invalid caption")
        predictions[sample_id] = caption
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation_ids = [Path(str(row["video"])).name for row in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Annotation video IDs must be unique")
    unknown = set(predictions) - set(annotation_ids)
    if unknown:
        raise ValueError(f"Prediction sample IDs are absent from annotations: {sorted(unknown)}")
    if not predictions:
        raise ValueError("No predictions supplied")
    if len(predictions) != len(annotations) and not allow_subset:
        raise ValueError("Predictions cover only part of the annotation; pass --allow-subset for an explicitly scoped subset score")
    if coverage is not None:
        coverage.update(annotation_total=len(annotations), prediction_total=len(predictions),
                        missing_prediction_count=len(annotations) - len(predictions))
    samples = []
    for row in annotations:
        sample_id = Path(str(row["video"])).name
        if sample_id not in predictions:
            continue
        conversations = row.get("conversations") or []
        samples.append(
            {
                "sample_id": sample_id,
                "base_events": [str(event) for event in row["events"]],
                "reference_caption": (
                    str(conversations[-1].get("value", ""))
                    if conversations
                    else ""
                ),
                "prediction": predictions[sample_id],
            }
        )
    return samples


def prepare_judgment(value: Any, event_count: int) -> dict[str, Any]:
    parsed = CaptionJudgment.model_validate(value)
    judgments = sorted(parsed.base_event_judgments, key=lambda item: item.event_index)
    if [item.event_index for item in judgments] != list(range(event_count)):
        raise ValueError("the judge must return one label per reference event")
    result = parsed.model_dump()
    result["base_event_judgments"] = [item.model_dump() for item in judgments]
    result["counts"] = {
        label: sum(item.label == label for item in judgments)
        for label in LABELS
    }
    result["counts"]["Hallucination"] = len(parsed.hallucinations)
    return result


def _usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    value = usage.model_dump()
    value.setdefault("input_tokens", value.get("prompt_tokens", 0))
    value.setdefault("output_tokens", value.get("completion_tokens", 0))
    value.setdefault("total_tokens", value["input_tokens"] + value["output_tokens"])
    return value


def judge_one(
    client: Any,
    sample: dict[str, Any],
    *,
    model: str,
    api_backend: str,
    reasoning_effort: str | None,
    max_output_tokens: int,
    max_retries: int,
    chat_response_format: str = "json_object",
) -> dict[str, Any]:
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    payload = {
        "base_events": [
            {"event_index": index, "event": event}
            for index, event in enumerate(sample["base_events"])
        ],
        "reference_caption": sample["reference_caption"],
        "candidate_caption": sample["prediction"],
    }
    started = time.perf_counter()
    last_error: Exception | None = None
    failed_attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_retries + 1):
        raw_text: str | None = None
        finish_reason: str | None = None
        try:
            if api_backend == "responses":
                options = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}
                response = client.responses.parse(
                    model=model,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    text_format=CaptionJudgment,
                    max_output_tokens=max_output_tokens,
                    store=False,
                    **options,
                )
                raw = response.output_parsed
                raw_text = getattr(response, "output_text", None)
                finish_reason = getattr(response, "status", None)
            else:
                schema = CaptionJudgment.model_json_schema()
                response_format: dict[str, Any] = {"type": "json_object"}
                if chat_response_format == "json_schema":
                    def strict_objects(value: Any) -> None:
                        if isinstance(value, dict):
                            if value.get("type") == "object":
                                value["additionalProperties"] = False
                            for nested in value.values():
                                strict_objects(nested)
                        elif isinstance(value, list):
                            for nested in value:
                                strict_objects(nested)
                    strict_objects(schema)
                    response_format = {"type": "json_schema", "json_schema": {
                        "name": "caption_judgment", "strict": True, "schema": schema,
                    }}
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                            + "\nReturn a JSON DATA INSTANCE, never a schema definition. "
                            + "Label all supplied events using zero-based indices. "
                            + 'The object has this shape: {"base_event_judgments":'
                            + '[{"event_index":0,"label":"Correct","evidence":"brief evidence"}],'
                            + '"hallucinations":[],"summary":"brief summary"}.\n'
                            + "Required schema:\n" + json.dumps(schema),
                        },
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    response_format=response_format,
                    max_completion_tokens=max_output_tokens,
                    **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
                )
                raw_text = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                raw = json.loads(raw_text or "")
            judgment = prepare_judgment(raw, len(sample["base_events"]))
            return {
                "sample_id": sample["sample_id"],
                "event_count": len(sample["base_events"]),
                "judgment": judgment,
                "model_returned": getattr(response, "model", None),
                "usage": _usage(response),
                "attempts": attempt,
                "latency_seconds": time.perf_counter() - started,
                "error": None,
                "failed_attempts": failed_attempts,
            }
        except Exception as exc:
            last_error = exc
            failed_attempts.append({
                "attempt": attempt, "raw_response": raw_text,
                "finish_reason": finish_reason,
                "error": f"{type(exc).__name__}: {exc}",
            })
            if attempt < max_retries:
                time.sleep(min(8.0, 2.0 ** (attempt - 1)))
    return {
        "sample_id": sample["sample_id"],
        "event_count": len(sample["base_events"]),
        "judgment": None,
        "attempts": max_retries,
        "latency_seconds": time.perf_counter() - started,
        "error": f"{type(last_error).__name__}: {last_error}",
        "failed_attempts": failed_attempts,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]
    totals = {label: 0 for label in (*LABELS, "Hallucination")}
    for row in successful:
        for label in totals:
            totals[label] += int(row["judgment"]["counts"][label])
    denominator = totals["Correct"] + totals["Missed"] + totals["Incorrect"]
    complete = len(successful) == len(rows) and bool(rows) and denominator > 0
    metrics = {
        "samples_judged": len(successful),
        "samples_failed": len(rows) - len(successful),
        "denominator_events": denominator,
        "totals": totals,
        "correct_rate": totals["Correct"] / denominator if complete else None,
        "MISS": totals["Missed"] / denominator if complete else None,
        "INCORRECT": totals["Incorrect"] / denominator if complete else None,
        "HALLUCINATION": totals["Hallucination"] / denominator if complete else None,
        "EER": (
            totals["Missed"] + totals["Incorrect"] + totals["Hallucination"]
        ) / denominator if complete else None,
        "status": "COMPLETE" if complete else "JUDGE_INCOMPLETE",
        "headline_metrics_available": complete,
    }
    return metrics


def _latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["sample_id"]): row for row in _read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score VideoSALMONN2 captions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--api-file", type=Path, help="Optional legacy URL/key pairs; environment credentials take precedence")
    parser.add_argument("--api-pair", type=int, default=1)
    parser.add_argument("--api-backend", choices=("responses", "chat-completions"), default="responses")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Judge model served by your selected API")
    parser.add_argument("--reasoning-effort", default=None, help="Optional, only for models supporting this parameter")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--allow-subset", action="store_true", help="Explicitly score supplied predictions covering a subset of the annotation")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--chat-response-format", choices=("json_object", "json_schema"), default="json_object")
    args = parser.parse_args()
    if args.workers < 1 or args.max_retries < 1 or args.max_output_tokens < 1:
        parser.error("workers, max-retries and max-output-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")

    base_url, api_key = load_api_credentials(args.api_file, args.api_pair)
    if args.api_backend == "chat-completions":
        base_url = chat_compatible_base_url(base_url)
    coverage: dict[str, Any] = {}
    samples = load_inputs(args.predictions, args.annotation,
                          allow_subset=args.allow_subset or args.limit is not None or bool(args.sample_id),
                          coverage=coverage)
    if args.sample_id:
        samples = [row for row in samples if row["sample_id"] == args.sample_id]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        parser.error("No samples remain after selection")
    coverage.update(selected_count=len(samples),
                    scoring_scope="full_annotation" if len(samples) == coverage["annotation_total"] else "explicit_subset")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = args.output_dir / "judgments.jsonl"
    identity_path = args.output_dir / "judge_identity.json"
    identity = {
        "model": args.model, "api_backend": args.api_backend,
        "chat_response_format": args.chat_response_format,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "endpoint_sha256": hashlib.sha256(base_url.encode()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prediction_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "annotation_sha256": hashlib.sha256(args.annotation.read_bytes()).hexdigest(),
        "selected_sample_ids_sha256": hashlib.sha256(json.dumps([sample["sample_id"] for sample in samples]).encode()).hexdigest(),
    }
    if args.fresh and judgments_path.exists():
        parser.error("--fresh requires an empty output directory; use a new directory to preserve previous judgments")
    if judgments_path.exists() and (
        not identity_path.exists() or json.loads(identity_path.read_text()) != identity
    ):
        parser.error("Existing judgments have a different or missing identity; choose a new output directory")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    existing = _latest_rows(judgments_path)
    pending = [
        sample
        for sample in samples
        if sample["sample_id"] not in existing
        or existing[sample["sample_id"]].get("error")
    ]

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=0)
    write_lock = threading.Lock()
    with judgments_path.open("a", encoding="utf-8") as writer:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    judge_one,
                    client,
                    sample,
                    model=args.model,
                    api_backend=args.api_backend,
                    reasoning_effort=args.reasoning_effort,
                    max_output_tokens=args.max_output_tokens,
                    max_retries=args.max_retries,
                    chat_response_format=args.chat_response_format,
                )
                for sample in pending
            ]
            for future in as_completed(futures):
                row = future.result()
                with write_lock:
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                    writer.flush()
                existing[row["sample_id"]] = row

    selected = [existing[sample["sample_id"]] for sample in samples]
    metrics = aggregate(selected)
    metrics.update(
        {
            "model": args.model,
            "api_backend": args.api_backend,
            "workers": args.workers,
            "input_coverage": coverage,
            "full_annotation_quality_complete": metrics["headline_metrics_available"] and coverage["scoring_scope"] == "full_annotation",
        }
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not metrics["headline_metrics_available"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
