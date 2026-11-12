# Data preparation

Download original annotations and media separately. The default workspace is
the repository root; `R2OMNI_WORKSPACE` selects another location.

```text
workspace/
├── Models/
│   ├── Qwen2.5-Omni-3B/
│   ├── Qwen2.5-Omni-7B/
│   └── MiniCPM-o-4_5/
└── Benchmarks/
    ├── videoSALMONN2/
    │   ├── video_salmonn2_test.json
    │   └── video/*.mp4
    ├── LongVALE/
    │   ├── longvale-annotations-eval.json
    │   ├── videos_eval/video_test_1171/*.mp4
    │   └── official_eval/          # Optional official scoring checkout
    └── OmniDCBench/
        ├── ours_gt_file.json
        ├── ours_gt_file_keypoints.json
        ├── Movie/
        └── Youtube/
```

These are lookup defaults. `--ann`, `--video_root`, and `--audio_root` override
them.

## video-SALMONN 2

Use [videoSALMONN2/video-SALMONN_2_testset](https://huggingface.co/datasets/videoSALMONN2/video-SALMONN_2_testset).

```bash
hf download videoSALMONN2/video-SALMONN_2_testset --repo-type dataset \
  --local-dir Benchmarks/videoSALMONN2
```

Extract media archives according to the provider's instructions. Annotations
are a JSON list; each row has `video` and a `conversations` list containing a
human `value` prompt. Relative video paths resolve against `--video_root`:
for `video/0.mp4`, pass the directory containing `video/`. The audiovisual path
uses audio embedded in the video.

## LongVALE

Use the evaluation split from
[ttgeng233/LongVALE](https://huggingface.co/datasets/ttgeng233/LongVALE).

```bash
hf download ttgeng233/LongVALE longvale-annotations-eval.json \
  --repo-type dataset --local-dir Benchmarks/LongVALE
hf download ttgeng233/LongVALE --repo-type dataset \
  --include 'raw_videos_test/*.zip' --local-dir Benchmarks/LongVALE
mkdir -p Benchmarks/LongVALE/videos_eval
for archive in Benchmarks/LongVALE/raw_videos_test/*.zip; do
  unzip -n "$archive" -d Benchmarks/LongVALE/videos_eval
done
```

The annotation is a JSON object keyed by video ID. The adapter resolves
`<video_id>.mp4` directly under the root or through `video_test_1171/` and
`videos_eval/`. Official scoring needs the separate LongVALE evaluator and its
metric dependencies; inference does not.

## OmniDCBench

Use [yaolily/OmniDenseCap-Benchmark](https://huggingface.co/datasets/yaolily/OmniDenseCap-Benchmark)
and follow its media preparation instructions. Preserve paths referenced by
`clip_path`. This adapter consumes one JSON record per line, even if the file
is named `.json`; each row contains `clip_path` and `duration`. Keypoint ground
truth is used for semantic scoring. The official temporal evaluator is
installed separately.

## Additional adapters

WorldSense, Video-MME, Daily-Omni, and AVSD interfaces are retained. Lookup
conventions are in `benchmarks/interface.py`; explicit paths can override
them. The standard long-form launcher defaults cover the three datasets above.
All dataset licenses and access conditions remain those of their providers.
