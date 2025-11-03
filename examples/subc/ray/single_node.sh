#!/usr/bin/env bash
set -euo pipefail

# Single-node launcher without Slurm, using Ray inside Docker.

#IMAGE=${IMAGE:-slimerl/slime:latest}
IMAGE=${IMAGE:-slimerl/slime:v0.5.0rc0-cu126}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
COMMON_DIR=/workspace/examples/subc/common
# Require a mount suffix and map host /lambda/nfs/<suffix> into the container at /lambda/nfs
if [ $# -lt 1 ]; then
  echo "Usage: $0 <mounted_suffix> [-m 8B|30B]   # mounts host /lambda/nfs/<mounted_suffix> -> container /lambda/nfs" >&2
  exit 1
fi

MOUNT_SUFFIX="$1"; shift
HOST_MOUNT="/lambda/nfs/${MOUNT_SUFFIX}"

# Parse optional args
MODEL_SIZE=""
while [ $# -gt 0 ]; do
  case "$1" in
    -m|--model-size)
      MODEL_SIZE="$2"; shift 2;;
    *)
      echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

if [ -z "${MODEL_SIZE}" ]; then
  echo "[ray-1n] MODEL_SIZE must be provided via -m 8B|30B" >&2
  exit 1
fi

if [ "${MODEL_SIZE}" != "8B" ] && [ "${MODEL_SIZE}" != "30B" ]; then
  echo "[ray-1n] Invalid MODEL_SIZE='${MODEL_SIZE}'. Expected 8B or 30B." >&2
  exit 1
fi

CONTAINER_NAME=slime-ray-1n-$$
WANDB_KEY=${WANDB_KEY:-}
HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}

if [ -z "${WANDB_KEY}" ]; then
  echo "[ray-1n] WANDB_KEY must be set (required by node_setup.sh)" >&2
  exit 1
fi

if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
  echo "[ray-1n] HUGGING_FACE_HUB_TOKEN must be set (required by node_setup.sh)" >&2
  exit 1
fi

echo "[ray-1n] Pulling image ${IMAGE}"
docker pull "$IMAGE"

echo "[ray-1n] Starting container ${CONTAINER_NAME}"
CONTAINER_ID=$(docker run -d --rm --gpus all --privileged --ipc=host --shm-size=128g \
  --tmpfs /tmp:exec,size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${REPO_ROOT}":/workspace \
  -v "${HOST_MOUNT}":/lambda/nfs \
  -e WANDB_KEY="${WANDB_KEY}" \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name "${CONTAINER_NAME}" \
  "$IMAGE" bash -lc 'sleep infinity')

for i in {1..10}; do
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    break
  fi
  sleep 1
done

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[ray-1n] Container ${CONTAINER_NAME} failed to start" >&2
  exit 1
fi

cleanup() {
  echo "[ray-1n] Cleaning up container ${CONTAINER_NAME}"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[ray-1n] Preparing node inside container (MODEL_SIZE=${MODEL_SIZE})"
docker exec -e MODEL_SIZE="${MODEL_SIZE}" "$CONTAINER_NAME" bash ${COMMON_DIR}/node_setup.sh

echo "[ray-1n] Starting training (MODEL_SIZE=${MODEL_SIZE})"
docker exec -e MODEL_SIZE="${MODEL_SIZE}" -e MASTER_ADDR=127.0.0.1 "$CONTAINER_NAME" bash ${COMMON_DIR}/start_training.sh

echo "[ray-1n] Tail logs:"
docker exec "$CONTAINER_NAME" bash -lc 'tail -n +1 -F /root/train.out'

docker wait "$CONTAINER_NAME" >/dev/null 2>&1 || true


