#!/usr/bin/env bash
set -euo pipefail

# This script runs INSIDE the Docker container on the head node.
# It expects MASTER_ADDR to be set to the head node's IP/hostname.

: "${MASTER_ADDR:?MASTER_ADDR must be set}"

# Ensure we are at repo root
if [ -d /workspace/pyproject.toml ] || [ -f /workspace/pyproject.toml ]; then
  cd /workspace
else
  cd /root/slime || { echo "slime repo not found"; exit 1; }
fi

echo "[start_training] MASTER_ADDR=${MASTER_ADDR}"

# Launch training and log to /root/train.out
bash scripts/run-glm4-9B.sh &> /root/train.out &
echo $! > /root/train.pid

echo "[start_training] Launched PID $(cat /root/train.pid)"


