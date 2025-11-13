Slime on Lambda/Slurm
=====================

Slurm launch scripts for slime RL training using Docker. Mirrors the workflow from `skypilot/llm/slime/slime.yaml`.



Files
-----

- `slime_single_node.sbatch`: Single-node training
- `slime_multi_node.sbatch`: Multi-node training with Ray cluster
- `node_setup.sh`: Environment setup script (runs inside containers)
- `start_training.sh`: Training launcher script (runs on head node)

Prerequisites
-------------

- Lambda 1-Click Cluster (for multi-node) or single instance with Slurm
- Docker installed and accessible
- Optional: Set `WANDB_KEY` environment variable

```
sudo usermod -aG docker $USER
newgrp docker
```

Usage
-----
For B200, build a different Docker image.
```
cd ~/slime
DOCKER_BUILDKIT=0 docker build -f docker/Dockerfile_b200 -t slimerl/slime:v0.5.2rc2-cu128-b200 .
```
set the IMAGE env.
```
#export IMAGE=slimerl/slime:v0.5.2rc2-cu128-b200
export IMAGE=dzorlu/slime-b200:latest
```

Prepare AIME eval data (boxed final answer)
```
python examples/subc/update_boxed_simple.py \
  --input /lambda/nfs/aime-2024/aime-2024.jsonl \
  --output /lambda/nfs/aime-2024/aime-2024.boxed.jsonl
```
- Then point your training script to the boxed file:
  - `--eval-prompt-data aime /lambda/nfs/aime-2024/aime-2024.boxed.jsonl`


From the repo root:
```
git clone https://github.com/dzorlu/slime.git
git checkout deniz/tim_30b
sudo usermod -aG docker $USER
newgrp docker

```

```bash
# Single node
export WANDB_KEY=<your-key>
export HUGGING_FACE_HUB_TOKEN=<your-key>
#TODO sbatch slime/examples/subc/slime_single_node.sbatch
# w/o slurm (argument is the mount suffix; mounts /lambda/nfs/<suffix> on host)
bash examples/subc/ray/single_node.sh FS_NAME --model-size 30B

# Multi-node (adjust -N and --gpus-per-node in script as needed)
#TODO sbatch slime/examples/subc/slime_multi_node.sbatch
```

View logs: `tail -f slurm-<jobid>.out`

How it works
------------

**Single-node**: Runs training on one node with Ray head on localhost.

**Multi-node**: 
1. Starts Docker containers on all nodes
2. Runs `node_setup.sh` on all nodes (installs slime, downloads data, converts weights)
3. Starts training on head node via `start_training.sh` (launches Ray head + job)
4. Worker nodes join Ray cluster
5. Streams logs from head container

Notes
-----

- Uses `docker run ... sleep infinity` pattern to maintain stable containers throughout the job
- All setup/training steps run via `docker exec` into these containers
- Default `scripts/run-glm4-9B.sh` uses `--actor-num-nodes 1` and `--colocate` (single-node training)
- For true multi-node actor training, modify the script to use `--actor-num-nodes ${SLURM_NNODES}`
- Ray dashboard available at `<head-ip>:8265` (if firewall allows)
- Debug rollout data is saved to `/lambda/nfs/{run_id}/rollout_{rollout_id}.pt` if enabled in the training script.

TODO
----
- Run single-node training as POC - on actual data and small model.
- Run multi-node training



