#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <bf16|r2omni|oscar|kivi-om|ott-om> <output-dir> [runner args...]" >&2
  exit 2
fi

METHOD="$1"
OUTPUT_DIR="$2"
shift 2

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to a local MiniCPM-o-4.5 directory}"
: "${R2OMNI_WORKSPACE:=${ROOT}}"

export R2OMNI_WORKSPACE PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
exec python "${ROOT}/scripts/base/launch.py" minicpm \
  --method "${METHOD}" --model "${MODEL_PATH}" --output "${OUTPUT_DIR}" -- "$@"
