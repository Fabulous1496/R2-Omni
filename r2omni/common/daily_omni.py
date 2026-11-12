from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any


GROUP_FIELDS = (
    "qa_type",
    "video_duration",
    "content_parent_category",
    "content_fine_category",
    "video_category",
)

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def normalize_daily_omni_row(row: dict[str, Any], position: int) -> dict[str, Any]:
    """Map the public Daily-Omni JSON schema onto the shared MCQ interface."""
    video_id = str(row.get("video_id") or "").strip()
    question = str(row.get("Question") or "").strip()
    choices = list(row.get("Choice") or [])
    answer = str(row.get("Answer") or "").strip().upper()[:1]
    if not video_id:
        raise ValueError(f"Daily-Omni row {position} has no video_id")
    if not question:
        raise ValueError(f"Daily-Omni row {position} has no Question")
    if len(choices) != 4:
        raise ValueError(
            f"Daily-Omni row {position} expects four choices, got {len(choices)}"
        )
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(
            f"Daily-Omni row {position} has invalid answer {row.get('Answer')!r}"
        )
    normalized = dict(row)
    normalized.update({
        "index": position,
        "video": video_id,
        "video_path": f"Videos/{video_id}/{video_id}_video.mp4",
        "audio_path": f"Videos/{video_id}/{video_id}_audio.wav",
        "question": question,
        "candidates": choices,
        "answer": answer,
        "qa_type": row.get("Type"),
    })
    return normalized


def load_daily_omni_rows(json_path: Path) -> list[dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Daily-Omni annotations must be a JSON list")
    return [
        normalize_daily_omni_row(row, position)
        for position, row in enumerate(data)
    ]


def resolve_video_path(dataset_root: Path, row: dict[str, Any]) -> str:
    path = Path(str(row["video_path"]))
    if not path.is_absolute():
        path = dataset_root / path
    return str(path.resolve())


def resolve_audio_path(dataset_root: Path, row: dict[str, Any]) -> Path:
    path = Path(str(row["audio_path"]))
    if not path.is_absolute():
        path = dataset_root / path
    return path.resolve()


def build_question(row: dict[str, Any]) -> str:
    """Reproduce the official Qwen2.5-Omni benchmark prompt."""
    if row.get("open_ended"):
        return str(row["question"])
    return (
        "\nYour task is to accurately answer multiple-choice questions based on the "
        "given video and audio together.\n"
        "Select the single most accurate answer from the given choices.\n"
        f"Question: {row['question']}\n"
        f"Choices: {list(row.get('candidates') or [])}\n"
        "Your answer should be a capital letter representing your choice: A, B, C, or D. "
        "Don't generate any other text.\n"
    )


def build_message(
    row: dict[str, Any],
    dataset_root: Path,
    nframes: int,
    *,
    include_audio: bool = True,
) -> list[dict[str, Any]]:
    # The official Transformers path samples video at the Qwen utility default
    # of 2 FPS. Daily-Omni clips are 30/60 seconds, so max_frames=128 does not
    # change the paper path while retaining the benchmark's explicit frame cap.
    media: list[dict[str, Any]] = [{
        "type": "video",
        "video": resolve_video_path(dataset_root, row),
        "max_frames": nframes,
    }]
    if include_audio:
        media.append({
            "type": "audio",
            "audio": str(resolve_audio_path(dataset_root, row)),
        })
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": media + [{"type": "text", "text": build_question(row)}]},
    ]


def daily_omni_record_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": row.get("video_id"),
        **{field: row.get(field) for field in GROUP_FIELDS},
    }


def aggregate_daily_omni_groups(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report paper-style task and duration slices; failures count as wrong."""
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
