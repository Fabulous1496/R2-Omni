# Third-party projects

The project-level MIT license covers this repository's original code.
Dependencies, downloaded model code, datasets, and optional upstream source
trees retain their own licenses and terms. No model weights, benchmark videos,
or pretrained AudioCLIP weights are redistributed here.

| Project | Upstream | Role |
| --- | --- | --- |
| Qwen2.5-Omni | https://github.com/QwenLM/Qwen2.5-Omni | Model and audiovisual processing |
| MiniCPM-o | https://github.com/OpenBMB/MiniCPM-o | Model, native chat and audiovisual processing |
| Transformers | https://github.com/huggingface/transformers | Model loading and attention interfaces |
| Triton | https://github.com/triton-lang/triton | Packed-cache kernels |
| KIVI | https://github.com/jy-yuan/KIVI | Optional low-bit quantization backend (MIT) |
| OTT | https://github.com/yisunlp/OTT | Optional low-bit quantization backend |
| OScaR | https://github.com/ZunhaiSu/OScaR-KV-Quant | Optional low-bit quantization backend (MIT) |
| OmniSelect | https://github.com/Yangmrl-nlp/OmniSelect | Reference for source/evaluation organization; optional comparison installed separately |
| SEATS | https://github.com/xxayt/SEATS | Reference for evaluation-script organization |

KIVI, OTT, and OScaR are fetched at the exact revisions specified in
`scripts/base/setup_kv_baselines.sh`; their source is not vendored into this
repository. A KIVI MIT notice is retained at `third_party/KIVI-LICENSE`.
Consult the license in each checkout before redistribution. The public adapters
connect upstream quantizers to Qwen projections, RoPE, GQA and cache lifecycle;
they do not imply that upstream authors officially support these Omni models.

Hadamard preconditioning, asymmetric low-bit quantization, official dataset
prompts, and upstream evaluation protocols are attributed to their respective
projects and are not presented as new algorithms here.
