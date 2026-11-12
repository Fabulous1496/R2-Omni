from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(predictions: Path) -> list[dict]:
    """Convert shared-runner records to the official LongVALE captioning log."""
    logs = []
    with predictions.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("error"):
                continue
            logs.append({
                "video_id": str(record["sample_id"]),
                "task": "captioning",
                "query_id": 0,
                "answer": str(record.get("response", "")),
            })
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a benchmark prediction JSONL for LongVALE metric.py"
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    logs = convert(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in logs:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"prepared {len(logs)} LongVALE captioning records")


if __name__ == "__main__":
    main()
