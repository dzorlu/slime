#!/usr/bin/env bash
set -euo pipefail

# Multi-node launcher without Slurm, using SSH fan-out + Ray inside Docker.
# Usage: ./multi_node_hostfile.sh -H hosts.txt [--gpus-per-node 8] [--image slimerl/slime:latest]

HOSTFILE=
GPUS_PER_NODE=8
IMAGE=slimerl/slime:latest
COMMON_DIR=/workspace/examples/subc/common
WANDB_KEY=${WANDB_KEY:-}
# Path to the slime repo on each remote host to mount at /workspace inside the container
# Default assumes same username on remotes and repo at /home/$USER/slime
REPO_PATH=${REPO_PATH:-/home/${USER}/slime}
# Optional: host path to a shared dataset volume to mount at /lambda/nfs in the container
NFS_PATH=${NFS_PATH:-}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -H|--hosts|--hostfile) HOSTFILE="$2"; shift 2;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    --repo|--repo-path) REPO_PATH="$2"; shift 2;;
    --nfs|--nfs-path) NFS_PATH="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [ -z "${HOSTFILE}" ] || [ ! -f "${HOSTFILE}" ]; then
  echo "Hostfile required. Use -H hosts.txt (head host must be first line)."; exit 1
fi

if [ -z "${WANDB_KEY}" ]; then
  echo "WANDB_KEY must be set (required by node_setup.sh)" >&2
  exit 1
fi

readarray -t HOSTS < "${HOSTFILE}"
HEAD=${HOSTS[0]}
WORKERS=("${HOSTS[@]:1}")

echo "[ray-mn] Using image: ${IMAGE} | GPUs per node: ${GPUS_PER_NODE}"

remote() { ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$@"; }

get_ip() { remote "$1" "hostname -I | awk '{print \$1}'"; }

HEAD_IP=$(get_ip "$HEAD")
echo "[ray-mn] Head: ${HEAD} (${HEAD_IP}) | Workers: ${WORKERS[*]:-none}"

start_container() {
  local host="$1"; local name="$2";
  local nfs_mount=""
  if [ -n "${NFS_PATH}" ]; then
    nfs_mount="-v ${NFS_PATH}:/lambda/nfs"
  fi
  remote "$host" "docker pull ${IMAGE} && \
    docker run -d --rm --name ${name} \
      --gpus all --privileged --network host --ipc=host --shm-size=128g \
      --tmpfs /tmp:exec,size=64g \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -v ${REPO_PATH}:/workspace ${nfs_mount} \
      -e WANDB_KEY=${WANDB_KEY} \
      ${IMAGE} sleep infinity"
}

exec_in() { remote "$1" "docker exec $2 bash -lc '$3'"; }

HEAD_CONT="slime-ray-head"
start_container "$HEAD" "$HEAD_CONT"
for w in "${WORKERS[@]}"; do
  start_container "$w" "slime-ray-${w}"
done

echo "[ray-mn] Running node setup on all nodes"
exec_in "$HEAD" "$HEAD_CONT" "bash ${COMMON_DIR}/node_setup.sh"
for w in "${WORKERS[@]}"; do
  exec_in "$w" "slime-ray-${w}" "bash ${COMMON_DIR}/node_setup.sh"
done

echo "[ray-mn] Starting training on head"
# Pass ACTOR_NUM_NODES to the training script to control multi-node usage
NUM_NODES=${#HOSTS[@]}
exec_in "$HEAD" "$HEAD_CONT" "MASTER_ADDR=${HEAD_IP} ACTOR_NUM_NODES=${NUM_NODES} bash ${COMMON_DIR}/start_training.sh"

echo "[ray-mn] Waiting for Ray dashboard on head"
for i in {1..120}; do
  if exec_in "$HEAD" "$HEAD_CONT" "curl -sf http://${HEAD_IP}:8265/api/version >/dev/null"; then
    echo "[ray-mn] Ray dashboard is up"; break
  fi
  sleep 1
done

echo "[ray-mn] Joining workers to Ray at ${HEAD_IP}:6379"
for w in "${WORKERS[@]}"; do
  exec_in "$w" "slime-ray-${w}" "ray start --address=${HEAD_IP}:6379 --num-gpus ${GPUS_PER_NODE}"
done

echo "[ray-mn] Streaming logs (Ctrl-C to detach)"
exec_in "$HEAD" "$HEAD_CONT" "tail -n +1 -F /root/train.out"


