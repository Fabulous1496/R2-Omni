#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <baseline|r2omni|kivi-om|ott-om> <dataset> <output-dir> [runner args...]" >&2
  exit 2
fi

METHOD="$1"
DATASET="$2"
OUTPUT_DIR="$3"
shift 3

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen2.5-Omni-3B/7B directory}"
: "${R2OMNI_WORKSPACE:=${ROOT}}"
export R2OMNI_WORKSPACE PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
# Pass user runner arguments last so explicit CLI values override YAML defaults.
exec python "${ROOT}/scripts/base/launch.py" qwen \
  --method "${METHOD}" --dataset "${DATASET}" \
  --model "${MODEL_PATH}" --output "${OUTPUT_DIR}" -- "$@"
