#!/usr/bin/env bash
set -euo pipefail

: "${WANDB_KEY:?WANDB_KEY must be set}"

# Prepares a node inside the Docker container: installs slime, downloads datasets/models, converts weights.

# Expect repository mounted at /workspace (repo root).
if [ -f /workspace/pyproject.toml ] || [ -f /workspace/setup.py ]; then
  cd /workspace
else
  cd /root
  if [ ! -d slime ]; then
    git clone https://github.com/THUDM/slime.git
  fi
  cd slime
fi

pip install -e .
pip install "huggingface-hub<1.0" --upgrade
pip install datasets

download_repo() {
  local kind="$1"
  local repo="$2"
  local target="$3"
  if [ -d "$target" ]; then
    echo "[node_setup] Skipping ${kind} download for ${repo}; found ${target}"
  else
    mkdir -p "$target"
    if [ "$kind" = dataset ]; then
      hf download --repo-type dataset "$repo" --local-dir "$target"
    else
      hf download "$repo" --local-dir "$target"
    fi
  fi
  [ "$kind" = model ] && ln -sfn "$target" /root/model
}

download_repo dataset zhuzilin/dapo-math-17k /lambda/nfs/dapo-math-17k &
download_repo dataset zhuzilin/aime-2024 /lambda/nfs/aime-2024 &
download_repo model Qwen/Qwen3-0.6B /lambda/nfs/models/Qwen3-0.6B &
wait

# Convert weights (save alongside default symlinked path)
MODEL_TORCH_DIR=/root/model_torch_dist
if [ -d "$MODEL_TORCH_DIR" ]; then
  echo "[node_setup] Skipping weight conversion; found ${MODEL_TORCH_DIR}"
else
  source scripts/models/qwen3-0.6B.sh
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /lambda/nfs/models/Qwen3-0.6B \
    --save "$MODEL_TORCH_DIR"
fi

echo "[node_setup] Completed"



