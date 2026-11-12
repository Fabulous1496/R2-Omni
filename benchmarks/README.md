# Evaluation code

`benchmark_mixed_active.py` is the shared batch-one inference runner used by
the public method scripts. It records resolved frame/generation policy, token
counts, generated outputs, and physical persistent-KV bytes in `result.json`
and `predictions.jsonl`.

Paths default to this checkout, including `Models/`, `Benchmarks/`, and
`third_party/`. Set `R2OMNI_WORKSPACE` for an existing workspace layout, or
pass `--model_path`, `--ann`, `--video_root`, and `--audio_root` directly.
Third-party sources can also live under `R2OMNI_THIRD_PARTY_DIR`; `KIVI_ROOT`,
`OTT_ROOT`, and `OSCAR_ROOT` override individual source directories.

The Qwen runner records end-of-generation **active** KV bytes and **allocated**
cache storage separately. Active bytes describe populated payload/metadata;
allocated bytes include spare capacity and backing storage retained by views.
Neither is peak GPU memory or the CUDA allocator's reserved pool. Comparisons
should use the same checkpoint and byte definition. Output length changes
end-of-generation cache size.

Dataset adapters live in `interface.py` and `r2omni/common/`. Included scoring
and export helpers preserve the official VideoSALMONN2, LongVALE, and
OmniDCBench output formats used in production. Model weights, datasets, API
credentials, and generated judgments are not part of this repository.

The runner intentionally has no disk-backed processor-input cache interface.
In-memory reuse of already decoded media for adjacent questions can be disabled
with `--no-media_cache` when strict per-request timing is required.

## VideoSALMONN2 scoring

Install the evaluation extra, set `OPENAI_API_KEY`, and optionally set
`OPENAI_BASE_URL` for a compatible service. Select your judge model explicitly:

```bash
python -m benchmarks.evaluate_videosalmonn2_gpt \
  --predictions outputs/r2omni/predictions.jsonl \
  --annotation Benchmarks/videoSALMONN2/video_salmonn2_test.json \
  --api-backend chat-completions --model "$JUDGE_MODEL" \
  --output-dir outputs/r2omni/scores
```

`--api-file` remains available for local URL/key-pair files; environment
credentials take precedence. A service supporting structured schema output
may use `--chat-response-format json_schema`. Responses API services may use
`--api-backend responses`. `--reasoning-effort` is optional and should only be
set when the selected service/model supports it.

Duplicate IDs, unknown IDs, generation errors, and empty captions are rejected.
For an intentional subset run, pass `--allow-subset` (or an explicit `--limit`
or `--sample-id`). Metrics record the annotation, prediction, and selected
counts; subset results are labeled accordingly. API attempts are bounded by
`--max-retries`, failed responses and finish reasons are retained, and failed
judgments are never converted to zero-quality scores. Aggregate quality rates
remain null until every selected sample has a valid judgment. These judgments
are produced by the selected LLM service, not a human evaluation.

Use a new output directory when changing the judge model, prompt/code,
annotation, predictions, or selected sample set. Resume checks the saved
`judge_identity.json` before reusing successful judgments. These checks do not
establish agreement with human judgments or equivalence between judge models.

## MiniCPM media support

Install `minicpmo-utils==1.0.6` in the MiniCPM environment. Alternatively, pass
`--media-helper /path/to/minicpmo/utils.py` or set `MINICPM_MEDIA_HELPER` to an
explicit compatible helper. No runner loads another machine's virtual
environment implicitly. MiniCPM weights require their upstream model-specific
dependencies; see the installation guide for the separate environment.
