from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_segments(raw_prediction: str) -> list[dict[str, Any]] | None:
    try:
        parsed = json.loads(raw_prediction)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed

    text = raw_prediction.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    body = text[1:-1]
    decoder = json.JSONDecoder()
    segments: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(body):
        while cursor < len(body) and (body[cursor].isspace() or body[cursor] == ","):
            cursor += 1
        if cursor == len(body):
            break
        try:
            item, cursor = decoder.raw_decode(body, cursor)
        except json.JSONDecodeError:
            return None
        if not isinstance(item, dict):
            return None
        segments.append(item)
    return segments


def export_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for record in records:
        if record.get("dataset") != "omnidcbench":
            continue
        metadata = record.get("dataset_metadata") or {}
        raw_prediction = str(record.get("response") or record.get("prediction") or "")
        parsed = _parse_segments(raw_prediction)
        item: dict[str, Any] = {
            "clip_path": metadata.get("clip_path") or record.get("sample_id"),
            "prediction": raw_prediction if raw_prediction else "FAILED",
        }
        if parsed:
            item["prediction_json"] = parsed
        exported.append(item)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--expected", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.predictions_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        records = records[: args.limit]
    exported = export_records(records)
    if args.expected is not None and len(exported) != args.expected:
        raise ValueError(f"expected {args.expected} records, got {len(exported)}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exported),
        encoding="utf-8",
    )
    print(f"wrote {len(exported)} OmniDCBench predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
