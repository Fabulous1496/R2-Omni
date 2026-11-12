from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MAX_PIXELS = 297920
MAX_FRAMES = 160
FPS = 2.0
def build_prompt(duration: int) -> str:
    end = f"{duration // 60:02d}:{duration % 60:02d}"
    return f"""First identify the meaningful scene changes throughout this entire {duration}-second video, then return only the final caption as valid JSON.
The output must contain 6 to 12 chronological JSON objects, begin at 00:00, and cover the video through {end}. The final timestamp must end at {end}. Use scene boundaries rather than fixed-duration slicing. Never generate a timestamp beyond the video.
Every object must contain exactly these string fields: timestamp, segment_detail_caption, video_background, acoustics_content, shooting_style, speech_content, camera_state. Use an empty string only when the video provides no evidence.
This is a syntax example for a different, hypothetical 10-second video; do not copy its timestamps or captions:
[
  {{"timestamp":"00:00-00:04","segment_detail_caption":"First event.","video_background":"First setting.","acoustics_content":"First sounds.","shooting_style":"First shot style.","speech_content":"First dialogue or empty string.","camera_state":"First camera state."}},
  {{"timestamp":"00:04-00:10","segment_detail_caption":"Final event.","video_background":"Final setting.","acoustics_content":"Final sounds.","shooting_style":"Final shot style.","speech_content":"Final dialogue or empty string.","camera_state":"Final camera state."}}
]
For the actual video, replace the example with 6 to 12 detailed objects covering 00:00-{end}. Output no Markdown or explanation. The first character must be [ and the last must be ]. Stop immediately after ]."""


def load_omnidcbench_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        clip_path = str(row.get("clip_path") or "").strip()
        if not clip_path:
            raise ValueError(f"OmniDCBench row {position} has no clip_path")
        duration = int(round(float(row["duration"])))
        rows.append({
            **row,
            "index": clip_path,
            "video_path": clip_path,
            "question": build_prompt(duration),
            "answer": None,
            "open_ended": True,
        })
    return rows


def resolve_video_path(dataset_root: Path, row: dict[str, Any]) -> str:
    path = Path(str(row["video_path"]))
    if not path.is_absolute():
        path = dataset_root / path
    return str(path.resolve())


def build_message(
    row: dict[str, Any], dataset_root: Path, _max_frames: int
) -> list[dict[str, Any]]:
    """Build the exact request used by the official OmniDCBench inference code."""
    return [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": resolve_video_path(dataset_root, row),
                "max_pixels": MAX_PIXELS,
                "max_frames": MAX_FRAMES,
                "fps": FPS,
                "video_max_pixels": MAX_PIXELS,
            },
            {"type": "text", "text": str(row["question"])},
        ],
    }]


def omnidcbench_record_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "clip_path": row.get("clip_path"),
        "orig_video_path": row.get("orig_video_path"),
        "duration": row.get("duration"),
        "seg_num": row.get("seg_num"),
        "official_fps": FPS,
        "official_max_frames": MAX_FRAMES,
        "official_max_pixels": MAX_PIXELS,
    }
