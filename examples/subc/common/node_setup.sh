#!/usr/bin/env bash
set -euo pipefail

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
pip install -U huggingface_hub

# Downloads (idempotent: huggingface_hub respects existing local-dir)
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k &
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024 &
hf download zai-org/GLM-Z1-9B-0414 --local-dir /root/GLM-Z1-9B-0414 &
wait

# Convert weights
source scripts/models/glm4-9B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/GLM-Z1-9B-0414 \
  --save /root/GLM-Z1-9B-0414_torch_dist

echo "[node_setup] Completed"


