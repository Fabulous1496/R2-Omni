#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY="${R2OMNI_THIRD_PARTY_DIR:-${ROOT}/third_party}"
mkdir -p "${THIRD_PARTY}"

clone_pinned() {
  local url="$1" destination="$2" revision="$3"
  if [[ ! -e "${destination}" ]]; then
    git clone --no-checkout "${url}" "${destination}"
    git -C "${destination}" checkout --detach "${revision}"
  elif [[ ! -e "${destination}/.git" ]]; then
    echo "Refusing to overwrite non-Git directory: ${destination}" >&2
    return 1
  fi
  local actual
  actual="$(git -C "${destination}" rev-parse HEAD)"
  if [[ "${actual}" != "${revision}" ]]; then
    echo "Revision mismatch in ${destination}: expected ${revision}, got ${actual}." >&2
    echo "Use a separate clean checkout; this installer does not reset existing files." >&2
    return 1
  fi
  if ! git -C "${destination}" diff --quiet || ! git -C "${destination}" diff --cached --quiet; then
    echo "Refusing to build from locally modified tracked sources: ${destination}" >&2
    return 1
  fi
}

METHODS=("$@")
if (( ${#METHODS[@]} == 0 )); then METHODS=(kivi ott oscar); fi
for method in "${METHODS[@]}"; do
  case "${method}" in
    kivi)
      clone_pinned https://github.com/jy-yuan/KIVI.git \
        "${THIRD_PARTY}/KIVI" 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
      (cd "${THIRD_PARTY}/KIVI/quant"; python setup.py build_ext --inplace)
      ;;
    ott)
      clone_pinned https://github.com/yisunlp/OTT.git \
        "${THIRD_PARTY}/OTT" 3a10b32b5d7e163b33bdea7befd07128b9c6d767
      (cd "${THIRD_PARTY}/OTT/quant"; python setup.py build_ext --inplace)
      ;;
    oscar)
      clone_pinned https://github.com/ZunhaiSu/OScaR-KV-Quant.git \
        "${THIRD_PARTY}/OScaR-KV-Quant" d2109f573da841151355976b66b06dade991388b
      git -C "${THIRD_PARTY}/OScaR-KV-Quant" submodule update --init --recursive
      (cd "${THIRD_PARTY}/OScaR-KV-Quant"; python setup.py build_ext --inplace)
      ;;
    --help|-h)
      echo "Usage: $0 [kivi] [ott] [oscar] (default: all three)"
      exit 0 ;;
    *) echo "Unknown baseline: ${method}" >&2; exit 2 ;;
  esac
done
echo "Selected pinned sources and locally compiled CUDA extensions are ready."
