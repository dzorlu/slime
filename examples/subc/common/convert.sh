#!/usr/bin/env bash
set -euo pipefail

# Minimal wrapper: point it at an iter_* checkpoint directory, get an HF folder.

usage() {
  echo "Usage: $0 <iter_dir> [output_dir]" >&2
  echo "Example: $0 /home/ubuntu/checkpoints/iter_0000039" >&2
  exit 1
}

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  usage
fi

ITER_DIR="${1%/}"
if [ ! -d "${ITER_DIR}" ]; then
  echo "[convert] ${ITER_DIR} does not exist." >&2
  exit 1
fi

OUTPUT_DIR="${2:-${ITER_DIR}_hf}"

if [ -f /workspace/pyproject.toml ] || [ -f /workspace/setup.py ]; then
  cd /workspace
else
  cd /root/slime || { echo "[convert] slime repo not found"; exit 1; }
fi

COMMON_PT="${ITER_DIR}/common.pt"
if [ ! -f "${COMMON_PT}" ]; then
  echo "[convert] Missing ${COMMON_PT}; is this a Megatron checkpoint dir?" >&2
  exit 1
fi

mapfile -t META < <(python3 - "$COMMON_PT" <<'PY'
import torch, sys
args = torch.load(sys.argv[1], map_location="cpu")["args"]
print(getattr(args, "hf_checkpoint", "") or "")
print(getattr(args, "model_name", "") or "")
vocab_size = getattr(args, "vocab_size", None)
print("" if vocab_size in (None, "") else vocab_size)
PY
)

HF_CHECKPOINT="${META[0]}"
MODEL_NAME="${META[1]}"
VOCAB_SIZE="${META[2]}"

MODEL_SPEC=()
if [ -n "${HF_CHECKPOINT}" ]; then
  MODEL_SPEC=(--origin-hf-dir "${HF_CHECKPOINT}")
elif [ -n "${MODEL_NAME}" ]; then
  MODEL_SPEC=(--model-name "${MODEL_NAME}")
else
  echo "[convert] Could not infer hf_checkpoint/model_name from ${COMMON_PT}; set HF_CHECKPOINT env var." >&2
  exit 1
fi

if [ -n "${HF_CHECKPOINT}" ]; then
  echo "[convert] Using HF assets from ${HF_CHECKPOINT}"
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
  --chunk-size "${CHUNK_SIZE}"
  "${MODEL_SPEC[@]}"
)

if [ -n "${VOCAB_SIZE}" ]; then
  CMD+=(--vocab-size "${VOCAB_SIZE}")
fi
if [ "${FORCE_CONVERT:-0}" -eq 1 ]; then
  CMD+=(--force)
fi

echo "[convert] Converting ${ITER_DIR}"
echo "[convert] Writing HF checkpoint to ${OUTPUT_DIR}"
echo "[convert] Running: ${CMD[*]}"
"${CMD[@]}"
echo "[convert] Done."


