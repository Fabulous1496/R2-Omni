# Optional upstream baselines

Run `bash scripts/base/setup_kv_baselines.sh` to fetch pinned KIVI, OTT, and
OScaR source trees and build their CUDA extensions in your current environment.
The repositories and compiled binaries are ignored by Git. Rebuild on each
machine against its PyTorch and CUDA installation; do not copy a `.so` from
another environment.

You may instead supply your own checkouts through `KIVI_ROOT`, `OTT_ROOT`, and
the corresponding runner arguments. See `docs/baseline_adapters.md` and
`THIRD_PARTY_NOTICES.md` for the adapter boundary and upstream provenance.
