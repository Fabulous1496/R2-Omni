from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from r2omni.common.worldsense import MAX_PIXELS, MIN_PIXELS, SYSTEM_PROMPT


TASK_INSTRUCTION = (
    "Answer the final question about the video using the video, its audio, and "
    "the preceding dialogue. Give only a concise, natural-language answer."
)


def load_avsd_rows(json_path: Path) -> list[dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    dialogs = data.get("dialogs") if isinstance(data, dict) else None
    if not isinstance(dialogs, list):
        raise ValueError("AVSD annotations must contain a dialogs list")
    rows: list[dict[str, Any]] = []
    for position, dialog in enumerate(dialogs):
        turns = dialog.get("dialog") or []
        if not turns:
            raise ValueError(f"AVSD dialog {position} has no turns")
        image_id = str(dialog.get("image_id") or "").strip()
        if not image_id:
            raise ValueError(f"AVSD dialog {position} has no image_id")
        final_turn = turns[-1]
        rows.append({
            **dialog,
            "index": f"{image_id}_{len(turns) - 1}",
            "official_image_id": position + 1,
            "video": f"{image_id}.mp4",
            "video_path": f"{image_id}.mp4",
            "question": str(final_turn.get("question") or "").strip(),
            "answer": str(final_turn.get("answer") or "").strip(),
            "open_ended": True,
        })
    return rows


def resolve_video_path(dataset_root: Path, row: dict[str, Any]) -> str:
    return str((dataset_root / str(row["video_path"])).resolve())


def build_message(
    row: dict[str, Any], dataset_root: Path, max_frames: int
) -> list[dict[str, Any]]:
    turns = row.get("dialog") or []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ]
    for turn_index, turn in enumerate(turns):
        question = str(turn.get("question") or "").strip()
        if turn_index == 0:
            content: list[dict[str, Any]] = [
                {
                    "type": "video",
                    "video": resolve_video_path(dataset_root, row),
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                    "total_pixels": max_frames * MAX_PIXELS,
                    "max_frames": max_frames,
                },
                {"type": "text", "text": f"{TASK_INSTRUCTION}\n\n{question}"},
            ]
        else:
            content = [{"type": "text", "text": question}]
        messages.append({"role": "user", "content": content})
        if turn_index + 1 < len(turns):
            answer = str(turn.get("answer") or "").strip()
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            })
    return messages


def avsd_record_metadata(row: dict[str, Any]) -> dict[str, Any]:
    turns = row.get("dialog") or []
    return {
        "image_id": row.get("image_id"),
        "turn_index": len(turns) - 1,
        "official_image_id": row.get("official_image_id"),
        "dialog_history": turns[:-1],
    }
