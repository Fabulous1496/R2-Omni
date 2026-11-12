from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    predictions = (workspace / args.predictions).resolve()
    score_dir = (workspace / args.score_dir).resolve()
    annotation = workspace / "Benchmarks/LongVALE/longvale-annotations-eval.json"
    official_eval = workspace / "Benchmarks/LongVALE/official_eval/longvalellm/eval"
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sys.path.insert(0, str(official_eval))
    import metric

    metrics = metric.captioning_metrics(rows, str(annotation))
    output = {
        "task": "dense_video_captioning",
        "samples_judged": len(rows),
        "samples_failed": 0,
        **{key: float(value) for key, value in metrics.items()},
        "caption_parser": "marker_to_marker_v2",
    }
    score_dir.mkdir(parents=True, exist_ok=True)
    (score_dir / "metrics.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"label": args.label, **output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
