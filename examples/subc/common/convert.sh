#!/usr/bin/env bash
set -euo pipefail

# Usage: convert.sh <8B|30B> <iter_dir> [output_dir]

usage() {
  echo "Usage: $0 <8B|30B> <iter_dir> [output_dir]" >&2
  exit 1
}

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  usage
fi

MODEL_SIZE="$1"
shift
ITER_DIR="${1%/}"
shift
OUTPUT_DIR="${1:-${ITER_DIR}_hf}"

if [ ! -d "${ITER_DIR}" ]; then
  echo "[convert] ${ITER_DIR} does not exist." >&2
  exit 1
fi

case "${MODEL_SIZE}" in
  8B|8b)
    MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-scripts/models/qwen3-8B.sh}"
    MODEL_LOCAL_DIR_DEFAULT="/lambda/nfs/models/Qwen3-8B"
    ;;
  30B|30b)
    MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-scripts/models/qwen3-30B-A3B.sh}"
    MODEL_LOCAL_DIR_DEFAULT="/lambda/nfs/models/Qwen3-30B-A3B"
    ;;
  *)
    echo "[convert] MODEL_SIZE must be 8B or 30B." >&2
    exit 1
    ;;
esac

MODEL_LOCAL_DIR="${MODEL_LOCAL_DIR:-${MODEL_LOCAL_DIR_DEFAULT}}"

if [ -f /workspace/pyproject.toml ] || [ -f /workspace/setup.py ]; then
  cd /workspace
else
  cd /root/slime || { echo "[convert] slime repo not found"; exit 1; }
fi

if [ ! -f "${MODEL_ARGS_SCRIPT}" ]; then
  echo "[convert] MODEL_ARGS_SCRIPT='${MODEL_ARGS_SCRIPT}' not found." >&2
  exit 1
fi
source "${MODEL_ARGS_SCRIPT}"

if [ ! -d "${MODEL_LOCAL_DIR}" ]; then
  echo "[convert] MODEL_LOCAL_DIR='${MODEL_LOCAL_DIR}' not found; override MODEL_LOCAL_DIR to point at the HF checkpoint." >&2
  exit 1
fi

CHUNK_GB="${CHUNK_GB:-5}"
if ! [[ "${CHUNK_GB}" =~ ^[0-9]+$ ]]; then
  echo "[convert] CHUNK_GB='${CHUNK_GB}' must be an integer number of GiB." >&2
  exit 1
fi
CHUNK_SIZE=$((CHUNK_GB * 1024 * 1024 * 1024))

PYTHON_BIN="${PYTHON_BIN:-python3}"
CMD=(
  "${PYTHON_BIN}" tools/convert_torch_dist_to_hf.py
  --input-dir "${ITER_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --origin-hf-dir "${MODEL_LOCAL_DIR}"
  --chunk-size "${CHUNK_SIZE}"
)

if [ "${FORCE_CONVERT:-0}" -eq 1 ]; then
  CMD+=(--force)
fi
if [ -n "${VOCAB_SIZE_OVERRIDE:-}" ]; then
  CMD+=(--vocab-size "${VOCAB_SIZE_OVERRIDE}")
fi

echo "[convert] MODEL_SIZE=${MODEL_SIZE}"
echo "[convert] Using MODEL_ARGS_SCRIPT=${MODEL_ARGS_SCRIPT}"
echo "[convert] Using HF assets from ${MODEL_LOCAL_DIR}"
echo "[convert] Converting ${ITER_DIR}"
echo "[convert] Writing HF checkpoint to ${OUTPUT_DIR}"
echo "[convert] Running: ${CMD[*]}"
"${CMD[@]}"
echo "[convert] Done."
