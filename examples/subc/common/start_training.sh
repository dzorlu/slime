#!/usr/bin/env bash
set -euo pipefail

# Starts training inside the Docker container on the head node.
# Requires MASTER_ADDR (IP/hostname of head) to be set in the environment.

: "${MASTER_ADDR:?MASTER_ADDR must be set}"

# Ensure we are at repo root
if [ -f /workspace/pyproject.toml ] || [ -f /workspace/setup.py ]; then
  cd /workspace
else
  cd /root/slime || { echo "slime repo not found"; exit 1; }
fi

echo "[start_training] MASTER_ADDR=${MASTER_ADDR}"

# Launch training and log to /root/train.out
bash scripts/run-glm4-9B.sh &> /root/train.out &
echo $! > /root/train.pid
echo "[start_training] Launched PID $(cat /root/train.pid)"


