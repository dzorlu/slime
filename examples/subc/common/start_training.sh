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

# Require model size
: "${MODEL_SIZE:?MODEL_SIZE must be set to 8B or 30B}"

# Launch training and log to /root/train.out
case "${MODEL_SIZE}" in
  8B)
    bash scripts/run-tim.sh &> /root/train.out &
    ;;
  30B)
    bash scripts/run-tim-30B.sh &> /root/train.out &
    ;;
  *)
    echo "[start_training] Invalid MODEL_SIZE='${MODEL_SIZE}'. Expected 8B or 30B." >&2
    exit 1
    ;;
esac
echo $! > /root/train.pid
echo "[start_training] Launched PID $(cat /root/train.pid)"


