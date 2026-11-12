# Evaluation and reproducibility

## Qwen generation

Run from the checkout or invoke a launcher by absolute path. `MODEL_PATH` must
point to downloaded local Qwen2.5-Omni weights. The launcher reads text-model
geometry from `config.json` to select 3B/7B; it does not depend on the directory
name. `R2OMNI_MODEL_PROFILE=3b|7b` can select it explicitly.

```bash
export MODEL_PATH=/path/to/Qwen2.5-Omni-3B
export R2OMNI_WORKSPACE=/path/to/workspace
bash scripts/eval_qwen2_5_omni_r2omni.sh video_salmonn2 outputs/r2omni \
  --ann /path/to/video_salmonn2_test.json --video_root /path/to/videoSALMONN2 \
  --limit 1
```

To inspect a resolved command without loading a model:

```bash
R2OMNI_DRY_RUN=1 R2OMNI_MODEL_PROFILE=3b MODEL_PATH=/path/to/weights \
  bash scripts/eval_qwen2_5_omni_r2omni.sh video_salmonn2 outputs/demo
```

The public launchers use greedy decoding, natural EOS stopping and dataset
limits from `configs/evaluation.yaml`. STC/SSPA and baseline YAML files control
supported options. Fixed kernel/layout fields are validated rather than
silently ignored. Trailing CLI options override presets. For example:

```bash
bash scripts/eval_qwen2_5_omni_r2omni.sh video_salmonn2 outputs/custom \
  --shallow_v_payload_bit_ceiling 6.26 --deep_v_payload_bit_ceiling 5.66
```

Changing a precision ceiling changes the operating point; a nominal setting
is not proof of a particular compression ratio. Inspect measured cache bytes.

Configuration overrides include `R2OMNI_CONFIG_DIR`,
`R2OMNI_EVALUATION_CONFIG`, `R2OMNI_METHOD_CONFIG`, `R2OMNI_SSPA_CONFIG`, and
`R2OMNI_STC_CONFIG`. Paths for optional upstream code include `KIVI_ROOT`,
`KIVI_EXTENSION_ROOT`, `OTT_ROOT`, `OTT_EXTENSION_ROOT`, and
`R2OMNI_THIRD_PARTY_DIR`.

## MiniCPM

In the separate MiniCPM environment, place media under the workspace structure
in `datasets.md`, then run:

```bash
export MODEL_PATH=/path/to/MiniCPM-o-4_5
export R2OMNI_WORKSPACE=/path/to/workspace
bash scripts/eval_minicpmo_4_5.sh r2omni outputs/minicpm-smoke --limit 1
bash scripts/eval_minicpmo_4_5.sh bf16 outputs/minicpm-bf16 --limit 1
# Other included methods: oscar, kivi-om, ott-om.
# Select other datasets using --dataset longvale or --dataset omnidcbench.
```

This path retains native model preprocessing and chat generation. The
`minicpmo-utils` package provides media utilities; `--media-helper` explicitly
selects a compatible utility file when needed. `R2OMNI_MINICPM_CONFIG` selects
another MiniCPM profile. Model weights and official scoring dependencies are
installed separately.

## video-SALMONN 2 scoring

Install the `evaluation` extra, then set credentials through environment
variables. Supply your own deployed judge model name. Use the same model,
prompt, generation and evaluator settings for every compared method.

```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://your-provider.example/v1
python -m benchmarks.evaluate_videosalmonn2_gpt \
  --predictions outputs/r2omni/predictions.jsonl \
  --annotation /path/to/video_salmonn2_test.json \
  --output-dir outputs/r2omni/scores \
  --api-backend chat-completions --model YOUR_JUDGE_MODEL \
  --workers 2 --max-retries 3
```

Use `--chat-response-format json_schema` only if the selected provider supports
strict structured outputs. The `responses` backend is available for compatible
providers. An optional `--api-file` accepts legacy pairs of base-URL/key lines;
environment credentials take precedence. Never commit keys or credential files.

Judgments record failed attempts and raw response text for diagnosis. Resume
requires matching evaluator/input identity; changed protocols need a new output
directory. Scoring a deliberate subset requires `--allow-subset` (or an explicit
`--limit`/`--sample-id`); coverage is recorded in the metrics. Incomplete
scoring does not yield headline quality. Check prediction
coverage against the intended annotation subset, and retain failures instead of
comparing different successful subsets.

## LongVALE and OmniDCBench scoring

Convert LongVALE predictions to the official captioning log:

```bash
python -m benchmarks.prepare_longvale_official_eval \
  outputs/longvale/predictions.jsonl outputs/longvale/official_predictions.jsonl
```

Place the official LongVALE evaluator under
`$R2OMNI_WORKSPACE/Benchmarks/LongVALE/official_eval`, install its metric
requirements, then invoke:

```bash
python -m benchmarks.score_longvale --workspace "$R2OMNI_WORKSPACE" \
  --label r2omni --predictions "$PWD/outputs/longvale/official_predictions.jsonl" \
  --score-dir "$PWD/outputs/longvale/scores"
```

For OmniDCBench, export the official-format JSONL and use the original temporal
evaluator. Supply its path explicitly:

```bash
python -m benchmarks.export_omnidcbench \
  outputs/omnidcbench/predictions.jsonl outputs/omnidcbench/official_predictions.jsonl
python -m benchmarks.score_omnidc --mode temporal \
  --official-script /path/to/TimeChat_Captioner/Eval/eval_time.py \
  --predictions outputs/omnidcbench/official_predictions.jsonl \
  --ground-truth /path/to/ours_gt_file.json \
  --score-dir outputs/omnidcbench/temporal --output outputs/omnidcbench/temporal.json
```

Semantic scoring uses `--mode qwen`, the keypoint ground truth and API
credentials. Consult each module's `--help` for all options. The repository does
not vendor the official metric implementations or their Java/COCO dependencies.

## Reporting a run

Record the source commit, environment, model revision, annotation/sample IDs,
STC and SSPA settings, generation limits, and evaluator identity. Match sample
IDs and the cache checkpoint before comparing methods. Report active payload
and allocated persistent storage separately, including metadata, protected
layers and decode reservations. Final cache size depends on generated length.

Packed-kernel efficiency requires native packed readout. Dense dequantization
reference modes are correctness tools. CPU tests and nonempty captions are not
quality or latency evidence; perform GPU smoke and complete scoring on the
destination system before running or reporting a large experiment.
