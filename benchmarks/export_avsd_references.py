from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_json", type=Path)
    parser.add_argument("hypotheses_json", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    dialogs = json.loads(args.annotation_json.read_text(encoding="utf-8"))["dialogs"]
    hypotheses = json.loads(args.hypotheses_json.read_text(encoding="utf-8"))
    selected = {int(item["image_id"]) for item in hypotheses}
    images = []
    annotations = []
    annotation_id = 1
    for position, dialog in enumerate(dialogs, start=1):
        if position not in selected:
            continue
        turns = dialog.get("dialog") or []
        if not turns:
            continue
        images.append({"id": position, "name": f"{dialog['image_id']}_{len(turns) - 1}"})
        annotations.append({
            "id": annotation_id,
            "image_id": position,
            "caption": str(turns[-1].get("answer") or ""),
        })
        annotation_id += 1
    if len(images) != len(selected):
        raise ValueError(f"reference alignment failed: {len(images)} != {len(selected)}")
    payload = {
        "info": {},
        "licenses": [],
        "type": "captions",
        "images": images,
        "annotations": annotations,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(images)} AVSD references")


if __name__ == "__main__":
    main()
