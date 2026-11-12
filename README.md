<div align="center">

# R²-Omni

**Training-free structural and precision redundancy reduction for Omni-LLMs**

</div>

R²-Omni combines Structure-Aware Token Consolidation (STC) before language-model
prefill with Semantic-Sensitivity Precision Allocation (SSPA) for the KV cache.
Historical keys use INT4; value pages use BF16, INT4, or INT2. A fused Triton
decoder reads the packed cache directly.

This camera-ready source release includes Qwen2.5-Omni-3B/7B and MiniCPM-o-4.5
integration, benchmark inference and scoring, and optional KV-baseline adapters.

## Installation

Use Linux, Python 3.10, an NVIDIA GPU, and a CUDA-compatible PyTorch installation.
The Qwen reference environment uses PyTorch 2.8.0, CUDA 12.8 wheels, Transformers
4.57.1, and Triton 3.4.0. Building CUDA extensions requires a compatible CUDA
toolkit and C++ compiler.

```bash
git clone https://github.com/Fabulous1496/R2-Omni.git
cd R2-Omni
conda create -n r2omni python=3.10 -y
conda activate r2omni
conda install -c conda-forge ffmpeg -y
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
bash scripts/base/setup.sh --evaluation
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

The setup script applies `configs/constraints-qwen-cu128.txt`. MiniCPM uses a
separate environment with Transformers 4.51.0; see [installation](docs/installation.md).

## Models and data

Download weights locally before inference:

```bash
hf download Qwen/Qwen2.5-Omni-3B --local-dir Models/Qwen2.5-Omni-3B
# Optional:
# hf download Qwen/Qwen2.5-Omni-7B --local-dir Models/Qwen2.5-Omni-7B
# hf download openbmb/MiniCPM-o-4_5 --local-dir Models/MiniCPM-o-4_5
```

Follow [data preparation](docs/datasets.md) for annotations and videos. Weights
and data may live in the ignored `Models/` and `Benchmarks/` directories, or in
an external workspace selected with `R2OMNI_WORKSPACE`. Explicit data paths are
also supported.

## Quick start

Run one video first, using paths on your own machine:

```bash
export MODEL_PATH="$PWD/Models/Qwen2.5-Omni-3B"
export VS2_ROOT=/path/to/videoSALMONN2

bash scripts/eval_qwen2_5_omni_r2omni.sh video_salmonn2 outputs/r2omni-smoke \
  --ann "$VS2_ROOT/video_salmonn2_test.json" \
  --video_root "$VS2_ROOT" --limit 1
```

The video root is the base directory for annotation-relative paths such as
`video/0.mp4`. Remove `--limit 1` for the complete benchmark. Outputs include
`predictions.jsonl` and `result.json`; inspect errors and sample coverage.

Launchers resolve the model profile from its local configuration.
`R2OMNI_MODEL_PROFILE=3b` or `7b` selects it explicitly. Supported options in
method/evaluation YAML files are applied to the runner; trailing CLI arguments
override them.

```bash
# Full precision:
bash scripts/eval_qwen2_5_omni_baseline.sh video_salmonn2 outputs/bf16 \
  --ann "$VS2_ROOT/video_salmonn2_test.json" --video_root "$VS2_ROOT"

# Other long-form datasets, prepared under R2OMNI_WORKSPACE:
bash scripts/eval_qwen2_5_omni_r2omni.sh longvale outputs/longvale
bash scripts/eval_qwen2_5_omni_r2omni.sh omnidcbench outputs/omnidcbench

# Optional quantization-baseline adaptations:
bash scripts/base/setup_kv_baselines.sh
bash scripts/eval_qwen2_5_omni_kivi-om.sh video_salmonn2 outputs/kivi-om
bash scripts/eval_qwen2_5_omni_ott-om.sh video_salmonn2 outputs/ott-om
```

See [evaluation](docs/evaluation.md) for MiniCPM, scoring, credentials, and
reproducibility details. See [baseline adapters](docs/baseline_adapters.md) for
the boundary between our model adaptations and upstream quantizers.

## Repository layout

The layout follows the core/evaluation/script separation used by OmniSelect
and SEATS.

```text
R2-Omni/
├── r2omni/             # STC, SSPA, packed caches, kernels, model integration
├── baselines/          # Model adapters for optional upstream quantizers
├── benchmarks/         # Dataset interfaces, inference, scoring and export
├── configs/            # Method profiles, evaluation and version constraints
├── scripts/
│   ├── base/           # Shared configuration, setup and launch code
│   └── eval_*.sh       # Per-method entry points
├── docs/               # Installation, data, evaluation and implementation
├── tests/              # CPU tests and optional CUDA checks
└── third_party/        # Notices and locally installed upstream code
```

The release contains no weights, datasets, generated results, private review
materials, credentials, or precompiled CUDA binaries.

## Validation

```bash
python -m pip install -e '.[test]'
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests
python -m benchmarks.benchmark_mixed_active --help
```

CPU tests verify interfaces, profiles, and cache accounting. The optional
`tests/check_packed_gqa.py` requires the matching upstream CUDA extension.
Benchmark generation requires local weights, media and sufficient GPU memory;
CPU tests do not establish quality or GPU latency.

## Acknowledgements and license

We thank Qwen2.5-Omni, MiniCPM-o, Transformers, Triton, KIVI, OTT, OScaR,
OmniSelect, and SEATS. See [third-party notices](THIRD_PARTY_NOTICES.md).
Original project code uses the [MIT License](LICENSE). Upstream projects,
models, and datasets retain their own licenses.
