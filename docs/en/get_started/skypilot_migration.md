# Migrating to SkyPilot

This guide helps you migrate from manual Ray cluster setup to SkyPilot-managed clusters.

## Overview

SkyPilot provides automated infrastructure management for distributed training jobs. Instead of manually setting up Ray clusters, SSH-ing to worker nodes, and managing cloud resources, SkyPilot handles all of this automatically through declarative YAML configurations.

## Key Benefits

- **Automated cluster provisioning**: No need to manually create VMs or configure networking
- **Multi-cloud support**: Easily switch between AWS, GCP, and Azure
- **Checkpoint persistence**: Automatic sync to cloud storage (S3, GCS, Azure Blob)
- **Cost optimization**: Use spot instances and autostop to reduce costs
- **Smart resumption**: Reuse existing clusters without recreating them
- **Simplified workflows**: Single command to launch, monitor, and manage training jobs

## Key Differences

### Before (Manual Setup)

```bash
# Start Ray cluster manually on head node
export MASTER_ADDR=${MLP_WORKER_0_HOST}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

# SSH to each worker node and start Ray workers
for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
  if [[ "$WORKER_IP" == "$MLP_WORKER_0_HOST" ]]; then
    continue
  fi
  ssh root@"${WORKER_IP}" \
    "ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP}"
done

# Submit job to Ray cluster
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"env_vars": {...}}' \
   -- python train.py --actor-num-nodes 8 ...
```

### After (SkyPilot)

```bash
# Everything automated with a single command
sky launch -c my-training-cluster sky_configs/examples/train_glm4_9b.yaml
```

## Migration Steps

### Step 1: Install SkyPilot

```bash
pip install "skypilot[aws]"  # or [gcp], [azure]
```

### Step 2: Configure Cloud Credentials

```bash
# For AWS
aws configure

# For GCP
gcloud auth application-default login

# For Azure
az login
```

### Step 3: Create Cloud Storage Bucket (Optional but Recommended)

```bash
# AWS S3
aws s3 mb s3://my-slime-checkpoints

# Google Cloud Storage
gsutil mb gs://my-slime-checkpoints

# Azure Blob Storage
az storage container create --name my-slime-checkpoints
```

### Step 4: Convert Your Training Script to SkyPilot YAML

Let's walk through converting the existing `run-glm4-9B.sh` script to a SkyPilot configuration.

#### Original Script Structure

```bash
#!/bin/bash
# run-glm4-9B.sh

# Kill existing processes
pkill -9 sglang
ray stop --force
pkill -9 ray

# Set environment variables
export PYTHONBUFFERED=16
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4-9B.sh"

# Define training arguments
CKPT_ARGS=(...)
ROLLOUT_ARGS=(...)
PERF_ARGS=(...)
# ... more argument arrays

# Start Ray cluster
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8

# Submit training job
ray job submit --runtime-env-json='{...}' \
   -- python train.py ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ...
```

#### Converted SkyPilot YAML

```yaml
# sky_configs/my_glm4_9b.yaml
name: my-glm4-9b-training

resources:
  cloud: aws
  accelerators: H100:8
  use_spot: false

num_nodes: 1

file_mounts:
  /slime_checkpoints:
    name: my-slime-checkpoints
    store: s3
    mode: MOUNT

envs:
  PYTHONPATH: /root/Megatron-LM/
  CUDA_DEVICE_MAX_CONNECTIONS: "1"
  WANDB_API_KEY: ${WANDB_API_KEY}
  
setup: |
  set -ex
  
  cd /root
  git clone https://github.com/THUDM/slime.git || (cd slime && git pull && cd ..)
  cd slime
  pip install -e .
  
  git clone https://github.com/NVIDIA/Megatron-LM.git /root/Megatron-LM || true
  
  # Download model and data
  pip install -U huggingface_hub
  huggingface-cli download zai-org/GLM-Z1-9B-0414 --local-dir /root/GLM-Z1-9B-0414

run: |
  set -ex
  
  cd /root/slime
  source scripts/models/glm4-9B.sh
  
  # SkyPilot automatically sets up Ray, no need for ray start or ray job submit
  python train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --rollout-num-gpus 4 \
    --hf-checkpoint /root/GLM-Z1-9B-0414/ \
    --ref-load /slime_checkpoints/GLM-Z1-9B-0414_torch_dist \
    --save /slime_checkpoints/GLM-Z1-9B-0414_slime/ \
    ${MODEL_ARGS[@]} \
    # ... rest of training arguments from CKPT_ARGS, ROLLOUT_ARGS, etc.
```

### Step 5: Launch Your Training Job

```bash
sky launch -c my-training-cluster sky_configs/my_glm4_9b.yaml
```

## Converting Existing Scripts

Here's a systematic approach to converting your existing training scripts:

### 1. Remove Ray Cluster Setup Commands

**Remove:**
- `ray start --head`
- SSH commands to worker nodes
- `ray stop --force`
- Process killing commands (`pkill`)

**Why:** SkyPilot automatically manages Ray cluster lifecycle.

### 2. Remove `ray job submit` Wrapper

**Before:**
```bash
ray job submit --runtime-env-json='{...}' -- python train.py ...
```

**After:**
```yaml
run: |
  python train.py ...
```

**Why:** SkyPilot directly executes the run command.

### 3. Move Environment Variables to YAML

**Before:**
```bash
export PYTHONPATH=/root/Megatron-LM/
export CUDA_DEVICE_MAX_CONNECTIONS=1
ray job submit --runtime-env-json='{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1"
  }
}' -- python train.py
```

**After:**
```yaml
envs:
  PYTHONPATH: /root/Megatron-LM/
  CUDA_DEVICE_MAX_CONNECTIONS: "1"
```

### 4. Keep All Training Arguments

All `train.py` arguments remain exactly the same:

```yaml
run: |
  python train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --rollout-num-gpus 4 \
    --hf-checkpoint /path/to/checkpoint \
    # ... all other arguments unchanged
```

### 5. Add Checkpoint Persistence

**Before:**
Checkpoints saved to local disk, manually copied to cloud storage

**After:**
```yaml
file_mounts:
  /slime_checkpoints:
    name: my-checkpoint-bucket
    store: s3
    mode: MOUNT
```

Then update checkpoint paths in training arguments:
```yaml
run: |
  python train.py \
    --save /slime_checkpoints/model_name/ \
    --load /slime_checkpoints/model_name/
```

## Example Conversions

### Example 1: Single-Node Training

See the complete example in `sky_configs/examples/train_glm4_9b.yaml`.

### Example 2: Multi-Node Training

For multi-node training like `run-glm4.5-355B-A32B.sh`:

**Key changes:**
- Set `num_nodes: 8` in YAML
- Remove all SSH-based worker setup
- Keep NCCL environment variables in `envs:` section
- Remove MPI-specific environment variables (handled by SkyPilot)

See the complete example in `sky_configs/examples/train_glm4_355b.yaml`.

## Migration Checklist

Use this checklist when migrating a training script:

- [ ] Installed SkyPilot and configured cloud credentials
- [ ] Created cloud storage bucket for checkpoints
- [ ] Created SkyPilot YAML configuration file
- [ ] Moved environment variables to `envs:` section
- [ ] Removed Ray cluster setup commands
- [ ] Removed `ray job submit` wrapper
- [ ] Updated checkpoint paths to use cloud storage mounts
- [ ] Verified all training arguments are preserved
- [ ] Added `setup:` section for dependencies and data download
- [ ] Tested configuration on a small cluster first
- [ ] Updated checkpoint bucket name to your own bucket

## Testing Your Migration

1. **Start with a small configuration:**
   ```yaml
   num_nodes: 1
   resources:
     accelerators: H100:1  # Use fewer GPUs for testing
   ```

2. **Launch and verify:**
   ```bash
   sky launch -c test-migration sky_configs/your_config.yaml
   ```

3. **Monitor logs:**
   ```bash
   sky logs test-migration --follow
   ```

4. **Verify training starts correctly:**
   ```bash
   sky ssh test-migration
   # Inside the cluster, check Ray status
   ray status
   # Check GPU utilization
   nvidia-smi
   ```

5. **Once verified, scale up:**
   Update `num_nodes` and accelerator count, then relaunch.

## Common Migration Issues

### Issue 1: Import Errors

**Symptom:** Python can't find Megatron-LM modules

**Solution:** Ensure `PYTHONPATH: /root/Megatron-LM/` is set in `envs:` section

### Issue 2: Data Not Found

**Symptom:** Training fails because data files don't exist

**Solution:** Add data download to `setup:` section or use `file_mounts:` for large datasets

### Issue 3: Checkpoint Paths

**Symptom:** Model can't find previous checkpoints

**Solution:** Update all checkpoint paths to use `/slime_checkpoints/` (the mounted directory)

### Issue 4: Network Configuration

**Symptom:** NCCL errors or communication timeouts

**Solution:** Ensure all NCCL environment variables are in the `envs:` section

## Best Practices

1. **Test locally first:** Verify your training script works before migrating to SkyPilot
2. **Start small:** Test with 1 node before scaling to multi-node
3. **Use version control:** Keep your SkyPilot YAML files in git
4. **Document customizations:** Add comments to your YAML files explaining custom settings
5. **Monitor costs:** Use `sky status` to track running clusters and set autostop timers
6. **Use meaningful names:** Name clusters descriptively (e.g., `glm4-9b-exp1` instead of `test`)

## Getting Help

If you encounter issues during migration:

1. Check the [SkyPilot documentation](https://skypilot.readthedocs.io/)
2. Review example configurations in `sky_configs/examples/`
3. Use `sky logs <cluster-name>` to debug failures
4. Check the [main SkyPilot README](../../../sky_configs/README.md) for troubleshooting tips

## Next Steps

After successfully migrating:

1. **Optimize costs:** Enable spot instances for non-critical jobs
2. **Set up autostop:** Automatically stop idle clusters
3. **Explore multi-cloud:** Try running on different cloud providers
4. **Automate experiments:** Create multiple YAML configs for different hyperparameters
