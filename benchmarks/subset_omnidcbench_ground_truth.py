from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth_jsonl", type=Path)
    parser.add_argument("predictions_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument(
        "--path-reference-jsonl",
        type=Path,
        help=(
            "optional annotation JSONL whose row-aligned clip_path values are "
            "authoritative; useful when the released keypoint file contains "
            "path-only normalization mistakes"
        ),
    )
    args = parser.parse_args()
    predictions = [
        json.loads(line)
        for line in args.predictions_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [str(row["clip_path"]) for row in predictions]
    selected_set = set(selected)
    ground_truth = [
        json.loads(line)
        for line in args.ground_truth_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.path_reference_jsonl is not None:
        references = [
            json.loads(line)
            for line in args.path_reference_jsonl.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if len(references) != len(ground_truth):
            raise ValueError(
                "path reference and ground truth must contain the same rows"
            )
        remapped: list[dict] = []
        for index, (row, reference) in enumerate(zip(ground_truth, references)):
            source_name = Path(str(row["clip_path"])).name
            reference_name = Path(str(reference["clip_path"])).name
            if source_name != reference_name:
                raise ValueError(
                    f"path-reference row mismatch at index {index}: "
                    f"{source_name!r} != {reference_name!r}"
                )
            if int(row["clip_start_sec"]) != int(reference["clip_start_sec"]):
                raise ValueError(f"clip-start mismatch at index {index}")
            if int(row["clip_end_sec"]) != int(reference["clip_end_sec"]):
                raise ValueError(f"clip-end mismatch at index {index}")
            remapped.append({**row, "clip_path": str(reference["clip_path"])})
        ground_truth = remapped
    by_path = {str(row["clip_path"]): row for row in ground_truth}
    missing = [path for path in selected if path not in by_path]
    if missing:
        raise ValueError(f"missing ground truth for {missing[:3]}")
    if len(selected_set) != len(selected):
        raise ValueError("duplicate prediction clip_path values")
    subset = [by_path[path] for path in selected]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset),
        encoding="utf-8",
    )
    print(f"wrote {len(subset)} aligned OmniDCBench ground-truth rows")


if __name__ == "__main__":
    main()
