from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


GROUP_FIELDS = ("duration", "domain", "sub_category", "task_type")


def normalize_videomme_row(row: dict[str, Any], position: int) -> dict[str, Any]:
    """Map the public Video-MME schema onto the WorldSense MCQ interface."""
    video_id = str(row.get("videoID") or "").strip()
    question_id = str(row.get("question_id") or "").strip()
    options = list(row.get("options") or [])
    if not video_id:
        raise ValueError(f"Video-MME row {position} has no videoID")
    if not question_id:
        raise ValueError(f"Video-MME row {position} has no question_id")
    if len(options) != 4:
        raise ValueError(
            f"Video-MME question {question_id} expects four options, got {len(options)}"
        )
    normalized = dict(row)
    normalized.update({
        "index": question_id,
        "video": video_id,
        "video_path": f"{video_id}.mp4",
        "candidates": options,
        "answer": str(row.get("answer") or "").strip().upper(),
    })
    return normalized


def load_videomme_rows(parquet_path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(parquet_path).to_pylist()
    return [normalize_videomme_row(row, position) for position, row in enumerate(rows)]


def videomme_record_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": row.get("video_id"),
        "videoID": row.get("videoID"),
        "question_id": row.get("question_id"),
        **{field: row.get(field) for field in GROUP_FIELDS},
    }


def aggregate_videomme_groups(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return official-style slices, counting failed records as incorrect."""
    groups: dict[str, dict[str, list[bool]]] = {
        field: defaultdict(list) for field in GROUP_FIELDS
    }
    for record in records:
        metadata = record.get("dataset_metadata") or {}
        for field in GROUP_FIELDS:
            value = metadata.get(field)
            if value is not None:
                groups[field][str(value)].append(bool(record.get("correct")))
    return {
        field: {
            name: {
                "correct": sum(values),
                "total": len(values),
                "accuracy": sum(values) / len(values),
            }
            for name, values in sorted(field_groups.items())
        }
        for field, field_groups in groups.items()
    }
