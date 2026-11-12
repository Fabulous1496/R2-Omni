# Installation details

Use an editable checkout on Linux with Python 3.10. Install FFmpeg/ffprobe and
a CUDA-compatible NVIDIA driver. FlashAttention and optional baseline kernels
also require `nvcc` and a C++ compiler compatible with the PyTorch build.

## Qwen2.5-Omni

Follow the root README. The reference environment uses PyTorch 2.8.0+cu128,
Transformers 4.57.1, Triton 3.4.0, qwen-omni-utils 0.0.9, NumPy 1.26.4, and
Decord 0.6.0. The setup script installs into the active Python environment and
applies `configs/constraints-qwen-cu128.txt`. Set `R2OMNI_CONSTRAINTS` to another
constraints file when deliberately using a different environment.

## MiniCPM-o-4.5

Use a separate environment because native MiniCPM model code uses Transformers
4.51.0. The runner loads the official model's custom Python implementation via
`trust_remote_code=True`.

```bash
conda create -n r2omni-minicpm python=3.10 -y
conda activate r2omni-minicpm
conda install -c conda-forge ffmpeg -y
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
bash scripts/base/setup.sh --minicpm --evaluation
python -m pip install flash-attn==2.8.3 --no-build-isolation
hf download openbmb/MiniCPM-o-4_5 --local-dir Models/MiniCPM-o-4_5
```

`configs/constraints-minicpm-cu128.txt` selects Transformers 4.51.0 and
minicpmo-utils 1.0.6, respecting the utility package's declared Pillow, Librosa
and MoviePy requirements. Dependency resolution is separate from GPU
validation; run a single-video smoke on the destination machine.

## Optional baseline extensions

```bash
bash scripts/base/setup_kv_baselines.sh
```

The script checks out pinned upstream sources and builds their extensions.
Build against the same environment's PyTorch/CUDA ABI. Use separate builds when
switching environments; custom checkout/extension paths are supported. The
release does not ship a machine-specific `.so`.

Download weights before an offline experiment. Network and proxy policy is
controlled by your shell; scripts do not require a cluster-specific proxy.
