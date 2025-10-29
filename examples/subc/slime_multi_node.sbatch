#!/bin/bash
#SBATCH -J slime-mn
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --gpus-per-node=8
#SBATCH -t 24:00:00
#SBATCH --exclusive

set -euo pipefail

# Multi-node launcher for slime RL training on Slurm.
# Head node runs training; worker nodes join Ray cluster.

# Configuration
IMAGE=slimerl/slime:latest
WORKDIR_MOUNT="$PWD"
SUBC_DIR="/workspace/examples/subc"
JOB_TAG="slime-mn-${SLURM_JOB_ID}"
HEAD_CONT="${JOB_TAG}-head"
WORK_CONT_PREFIX="${JOB_TAG}-w"
WANDB_KEY=${WANDB_KEY:-}

# Get head node info
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_HOST=$(getent hosts "$MASTER_NODE" | awk '{print $1}' | head -n1 || true)
if [ -z "$MASTER_HOST" ]; then MASTER_HOST="$MASTER_NODE"; fi
DASHBOARD_PORT=8265
RAY_GCS_PORT=6379

# Step 1: Pull Docker image on all nodes
echo "[slime-mn] Pulling image ${IMAGE} on all nodes"
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} docker pull "${IMAGE}"

# Step 2: Start one container per node with GPU access
echo "[slime-mn] Starting per-node containers"
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} bash -c '
set -euo pipefail
NAME="'"${WORK_CONT_PREFIX}"'"-${SLURM_PROCID}"
if [ "${SLURM_PROCID}" -eq 0 ]; then NAME="'"${HEAD_CONT}"'"; fi
docker run -d --rm --name "$NAME" \
  --gpus all --privileged --network host --ipc=host --shm-size=128g \
  --tmpfs /tmp:exec,size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "'"${WORKDIR_MOUNT}"'":/workspace \
  -e WANDB_KEY="'"${WANDB_KEY}"'" \
  "'"${IMAGE}"'" sleep infinity
'

# Step 3: Setup environment on all nodes (install slime, download data/models, convert weights)
echo "[slime-mn] Installing slime and preparing data on ALL nodes"
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} bash -c '
set -euo pipefail
NAME="'"${WORK_CONT_PREFIX}"'"-${SLURM_PROCID}
if [ "${SLURM_PROCID}" -eq 0 ]; then NAME="'"${HEAD_CONT}"'"; fi
docker exec "$NAME" bash '"${SUBC_DIR}"'/node_setup.sh
'

# Step 4: Start training on head node only
echo "[slime-mn] Starting training in head container (run-glm4-9B.sh)"
docker exec -e MASTER_ADDR="${MASTER_HOST}" "${HEAD_CONT}" bash ${SUBC_DIR}/start_training.sh

# Step 5: Wait for Ray dashboard to be ready before joining workers
echo "[slime-mn] Waiting for Ray dashboard to be ready on ${MASTER_HOST}:${DASHBOARD_PORT}"
for i in {1..300}; do
  if docker exec "${HEAD_CONT}" bash -lc "curl -sf http://${MASTER_HOST}:${DASHBOARD_PORT}/api/version >/dev/null"; then
    echo "[slime-mn] Ray dashboard is up"
    break
  fi
  sleep 2
  if [ "$i" -eq 300 ]; then echo "[slime-mn] Ray dashboard did not become ready"; exit 1; fi
done

# Step 6: Join worker nodes to Ray cluster
echo "[slime-mn] Joining worker containers to Ray cluster at ${MASTER_HOST}:${RAY_GCS_PORT}"
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} bash -c '
set -euo pipefail
if [ "${SLURM_PROCID}" -ne 0 ]; then
  NAME="'"${WORK_CONT_PREFIX}"'"-${SLURM_PROCID}"
  docker exec "$NAME" ray start --address="'"${MASTER_HOST}"':'"${RAY_GCS_PORT}"'" --num-gpus ${SLURM_GPUS_ON_NODE:-8}
fi
'

# Step 7: Follow training logs
echo "[slime-mn] Following training log from head container (Ctrl-C to detach)"
docker exec "${HEAD_CONT}" bash -lc 'tail -n +1 -F /root/train.out'

# Step 8: Cleanup
echo "[slime-mn] Job complete. Stopping Ray workers and cleaning up containers"
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} bash -c '
set -euo pipefail
NAME="'"${WORK_CONT_PREFIX}"'"-${SLURM_PROCID}"
if [ "${SLURM_PROCID}" -eq 0 ]; then NAME="'"${HEAD_CONT}"'"; fi
docker exec "$NAME" ray stop || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
'

echo "[slime-mn] Done."


