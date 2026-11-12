from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def export_records(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dialogs: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
    for record in records:
        if record.get("dataset") != "avsd":
            continue
        metadata = record.get("dataset_metadata") or {}
        image_id = str(metadata["image_id"])
        official_image_id = int(metadata["official_image_id"])
        response = str(record.get("response") or record.get("prediction") or "").strip()
        history = [dict(turn) for turn in metadata.get("dialog_history") or []]
        history.append({"question": str(record.get("question") or ""), "answer": response})
        dialogs.append({"image_id": image_id, "dialog": history})
        hypotheses.append({"image_id": official_image_id, "caption": response})
    return {"dialogs": dialogs}, hypotheses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_jsonl", type=Path)
    parser.add_argument("submission_json", type=Path)
    parser.add_argument("hypotheses_json", type=Path)
    parser.add_argument("--expected", type=int, default=None)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.predictions_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    submission, hypotheses = export_records(records)
    if args.expected is not None and len(hypotheses) != args.expected:
        raise ValueError(f"expected {args.expected} records, got {len(hypotheses)}")
    for path in (args.submission_json, args.hypotheses_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.submission_json.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.hypotheses_json.write_text(
        json.dumps(hypotheses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(hypotheses)} AVSD predictions")


if __name__ == "__main__":
    main()
