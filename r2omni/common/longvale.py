from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from r2omni.common.worldsense import MAX_PIXELS, MIN_PIXELS, SYSTEM_PROMPT


DENSE_CAPTION_PROMPT = (
    "Could you please detail the events that took place during different time "
    "segments in the video? List the events in the format: From xx to xx, "
    "event1.\nFrom xx to xx, event2.\n... Use integer timestamps normalized "
    "to the range 00 to 99. Include relevant visual events, speech, sounds, "
    "and their relationships."
)


def normalize_longvale_row(
    video_id: str, annotation: dict[str, Any], position: int
) -> dict[str, Any]:
    """Map the official LongVALE dense-caption annotation to the shared runner."""
    video_id = str(video_id).strip()
    if not video_id:
        raise ValueError(f"LongVALE row {position} has no video id")
    timestamps = annotation.get("timestamps") or []
    sentences = annotation.get("sentences") or []
    if len(timestamps) != len(sentences):
        raise ValueError(f"LongVALE row {video_id} has mismatched timestamps and sentences")
    return {
        **annotation,
        "index": video_id,
        "video_id": video_id,
        "video": f"{video_id}.mp4",
        "question": DENSE_CAPTION_PROMPT,
        "answer": None,
        "open_ended": True,
    }


def load_longvale_rows(json_path: Path) -> list[dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("LongVALE annotations must be a video-id keyed JSON object")
    return [
        normalize_longvale_row(video_id, annotation, position)
        for position, (video_id, annotation) in enumerate(data.items())
    ]


def resolve_video_path(dataset_root: Path, row: dict[str, Any]) -> str:
    video_id = str(row["video_id"])
    filename = f"{video_id}.mp4"
    candidates = (
        dataset_root / filename,
        dataset_root / "video_test_1171" / filename,
        dataset_root / "videos_eval" / filename,
        dataset_root / "videos_eval" / "video_test_1171" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return str(candidates[-1].resolve())


def build_message(
    row: dict[str, Any], dataset_root: Path, nframes: int
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": resolve_video_path(dataset_root, row),
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                    "total_pixels": nframes * MAX_PIXELS,
                    "nframes": nframes,
                },
                {"type": "text", "text": str(row["question"])},
            ],
        },
    ]


def longvale_record_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": row.get("video_id"),
        "duration": row.get("duration"),
        "timestamps": row.get("timestamps"),
        "sentences": row.get("sentences"),
    }
