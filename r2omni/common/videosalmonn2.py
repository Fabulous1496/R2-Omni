from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from r2omni.common.worldsense import MAX_PIXELS, MIN_PIXELS, SYSTEM_PROMPT


def _official_prompt(row: dict[str, Any], position: int) -> str:
    conversations = row.get("conversations") or []
    for turn in conversations:
        if str(turn.get("from") or "").lower() == "human":
            prompt = str(turn.get("value") or "").replace("<image>", "").strip()
            if prompt:
                return prompt
    raise ValueError(f"video-SALMONN 2 row {position} has no human prompt")


def normalize_videosalmonn2_row(
    row: dict[str, Any], position: int
) -> dict[str, Any]:
    """Map the official JSON schema onto the shared open-generation runner."""
    video_path = str(row.get("video") or "").strip()
    if not video_path:
        raise ValueError(f"video-SALMONN 2 row {position} has no video")
    video_id = Path(video_path).name
    normalized = dict(row)
    normalized.update({
        "index": video_id,
        "video": video_id,
        "video_path": video_path,
        "question": _official_prompt(row, position),
        "open_ended": True,
        "answer": None,
    })
    return normalized


def load_videosalmonn2_rows(json_path: Path) -> list[dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("video-SALMONN 2 annotations must be a JSON list")
    return [
        normalize_videosalmonn2_row(row, position)
        for position, row in enumerate(data)
    ]


def resolve_video_path(dataset_root: Path, row: dict[str, Any]) -> str:
    path = Path(str(row["video_path"]))
    if not path.is_absolute():
        path = dataset_root / path
    return str(path.resolve())


def build_message(
    row: dict[str, Any],
    dataset_root: Path,
    nframes: int,
    *,
    resolution_policy: str = "benchmark_cap",
    video_fps: float | None = None,
    max_video_frames: int | None = None,
) -> list[dict[str, Any]]:
    video_content: dict[str, Any] = {
        "type": "video",
        "video": resolve_video_path(dataset_root, row),
    }
    if video_fps is None:
        video_content["nframes"] = nframes
        resolution_frame_budget = nframes
    else:
        if video_fps <= 0:
            raise ValueError("video_fps must be positive")
        if max_video_frames is None or max_video_frames <= 0:
            raise ValueError(
                "max_video_frames must be positive when video_fps is provided"
            )
        video_content.update({
            "fps": float(video_fps),
            "max_frames": int(max_video_frames),
        })
        resolution_frame_budget = int(max_video_frames)
    if resolution_policy == "benchmark_cap":
        video_content.update({
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
            "total_pixels": resolution_frame_budget * MAX_PIXELS,
        })
    elif resolution_policy != "qwen_native":
        raise ValueError(f"unsupported video resolution policy: {resolution_policy}")
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": str(row["question"])},
            ],
        },
    ]


def videosalmonn2_record_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {"video": row.get("video"), "video_path": row.get("video_path")}
