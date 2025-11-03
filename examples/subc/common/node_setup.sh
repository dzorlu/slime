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
#pip install -U sglang
#pip install -U sglang-router==0.2.2

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

# Require model size (8B or 30B)
: "${MODEL_SIZE:?MODEL_SIZE must be set to 8B or 30B}"
case "${MODEL_SIZE}" in
  8B)
    MODEL_REPO="SubconsciousDev/TIM-8b-long-grpo"
    MODEL_LOCAL_DIR="/lambda/nfs/models/TIM-8b-long-grpo"
    MODEL_ARGS_SCRIPT="scripts/models/tim-8B.sh"
    MODEL_TORCH_DIR="/lambda/nfs/models/model_torch_dist_8B"
    ;;
  30B)
    MODEL_REPO="SubconsciousDev/Tim-30B-A3B-sft"
    MODEL_LOCAL_DIR="/lambda/nfs/models/Tim-30B-A3B-sft"
    MODEL_ARGS_SCRIPT="scripts/models/tim-30B.sh"
    MODEL_TORCH_DIR="/lambda/nfs/models/model_torch_dist_30B"
    ;;
  *)
    echo "[node_setup] Invalid MODEL_SIZE='${MODEL_SIZE}'. Expected 8B or 30B." >&2
    exit 1
    ;;
esac

# Datasets/models
#download_repo dataset zhuzilin/dapo-math-17k /lambda/nfs/dapo-math-17k &
download_repo dataset mitroitskii/OpenR1-Math-220k-formatted /lambda/nfs/OpenR1-Math-220k-formatted &
download_repo dataset zhuzilin/aime-2024 /lambda/nfs/aime-2024 &
download_repo model "${MODEL_REPO}" "${MODEL_LOCAL_DIR}" &

wait

# Convert weights
if [ -d "$MODEL_TORCH_DIR" ]; then
  echo "[node_setup] Skipping weight conversion; found ${MODEL_TORCH_DIR}"
else
  # Source model-specific args
  source "${MODEL_ARGS_SCRIPT}"
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint "${MODEL_LOCAL_DIR}" \
    --tokenizer-type HuggingFaceTokenizer \
    --save "$MODEL_TORCH_DIR"
fi

echo "[node_setup] Completed"



