#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE=qwen
EXTRAS=()
PIP_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --minicpm) PROFILE=minicpm; EXTRAS+=(minicpm) ;;
    --evaluation) EXTRAS+=(evaluation) ;;
    --help|-h)
      echo "Usage: $0 [--minicpm] [--evaluation] [pip install options...]"
      echo "R2OMNI_CONSTRAINTS overrides the default tested-core constraints file."
      exit 0 ;;
    *) PIP_ARGS+=("${arg}") ;;
  esac
done
CONSTRAINTS="${R2OMNI_CONSTRAINTS:-${ROOT}/configs/constraints-${PROFILE}-cu128.txt}"
TARGET="${ROOT}"
if (( ${#EXTRAS[@]} )); then
  EXTRAS_CSV="$(IFS=,; echo "${EXTRAS[*]}")"
  TARGET+="[${EXTRAS_CSV}]"
fi
python -m pip install -c "${CONSTRAINTS}" -e "${TARGET}" "${PIP_ARGS[@]}"

echo "R2-Omni installed. Run scripts/base/setup_kv_baselines.sh only when"
echo "kivi-om, ott-om, or OScaR is needed. CUDA extensions are built locally."
