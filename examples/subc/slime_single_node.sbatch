#!/bin/bash
#SBATCH -J slime-1n
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 12:00:00
#SBATCH --exclusive

set -euo pipefail

# Single-node launcher for slime RL training on Slurm.
# Runs the same pipeline as skypilot/llm/slime/slime.yaml but on one node.

# Configuration
IMAGE=slimerl/slime:latest
WORKDIR_MOUNT="$PWD"
SUBC_DIR="/workspace/examples/subc"
CONTAINER_NAME="slime-1n-${SLURM_JOB_ID}"
WANDB_KEY=${WANDB_KEY:-}

# Step 1: Pull Docker image
echo "[slime-1n] Pulling docker image: ${IMAGE}"
docker pull "${IMAGE}"

# Step 2: Start container with GPU access
# Using 'sleep infinity' keeps the container alive for subsequent docker exec commands
echo "[slime-1n] Starting container ${CONTAINER_NAME}"
docker run --rm --gpus all --privileged --ipc=host --shm-size=128g \
  --tmpfs /tmp:exec,size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${WORKDIR_MOUNT}":/workspace \
  -e WANDB_KEY="${WANDB_KEY}" \
  --name "${CONTAINER_NAME}" \
  "${IMAGE}" sleep infinity &
CONTPID=$!

# Step 3: Setup environment inside container (install slime, download data/models, convert weights)
echo "[slime-1n] Preparing node inside container"
docker exec "${CONTAINER_NAME}" bash ${SUBC_DIR}/node_setup.sh

# Step 4: Start training with Ray head on localhost
echo "[slime-1n] Starting training inside container"
docker exec -e MASTER_ADDR="127.0.0.1" "${CONTAINER_NAME}" bash ${SUBC_DIR}/start_training.sh

echo "[slime-1n] Training launched in container ${CONTAINER_NAME}. Tail logs with:"
echo "  docker exec ${CONTAINER_NAME} bash -lc 'tail -F /root/train.out'"

# Keep the job alive until container exits
wait ${CONTPID}

echo "[slime-1n] Done."


