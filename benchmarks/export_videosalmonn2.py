#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def export_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    official: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if record.get("dataset") != "video_salmonn2":
            raise ValueError(
                f"unexpected dataset for sample {record.get('sample_id')!r}: "
                f"{record.get('dataset')!r}"
            )
        if record.get("error"):
            raise ValueError(
                f"sample {record.get('sample_id')!r} failed: {record['error']}"
            )
        metadata = record.get("dataset_metadata") or {}
        video_id = Path(str(metadata.get("video") or record.get("sample_id") or "")).name
        prediction = str(record.get("response") or "").strip()
        if not video_id or not prediction:
            raise ValueError(
                f"sample {record.get('sample_id')!r} has an empty id or prediction"
            )
        if video_id in seen:
            raise ValueError(f"duplicate video id: {video_id}")
        seen.add(video_id)
        official.append({"id": [video_id], "pred": prediction})
    return official


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export shared runner JSONL to the official video-SALMONN 2 schema."
    )
    parser.add_argument("predictions_jsonl", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--expected", type=int, default=None)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.predictions_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    official = export_records(records)
    if args.expected is not None and len(official) != args.expected:
        raise ValueError(f"expected {args.expected} predictions, got {len(official)}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(official, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(official)} predictions to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
