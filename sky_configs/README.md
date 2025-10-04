# SkyPilot Integration for slime

This directory contains SkyPilot configurations for launching slime training jobs on cloud providers.

## Overview

SkyPilot automates:
- Cloud cluster provisioning (AWS, GCP, Azure)
- Ray cluster setup (head + worker nodes)
- Checkpoint persistence to cloud storage
- Environment configuration
- Cost optimization with spot instances

## Prerequisites

1. **Install SkyPilot:**
   ```bash
   pip install "skypilot[aws]"  # or [gcp], [azure]
   ```

2. **Configure cloud credentials:**
   ```bash
   # For AWS
   aws configure
   
   # For GCP
   gcloud auth application-default login
   
   # For Azure
   az login
   ```

3. **Set up cloud storage bucket for checkpoints** (optional but recommended):
   - Create an S3/GCS/Azure Blob Storage bucket for checkpoint persistence
   - Ensure your cloud credentials have read/write access to the bucket

## Quick Start

### Option 1: Use example configurations

**Launch a single-node GLM-4 9B training job:**
```bash
sky launch -c slime-glm4-9b sky_configs/examples/train_glm4_9b.yaml
```

**Launch a multi-node GLM-4.5 355B training job:**
```bash
sky launch -c slime-glm4-355b sky_configs/examples/train_glm4_355b.yaml
```

### Option 2: Use the generic template with environment variables

```bash
export SLIME_CLUSTER_NAME=my-training-job
export SLIME_NUM_NODES=2
export SLIME_ACCELERATOR="H100:8"
export SLIME_CHECKPOINT_BUCKET=my-checkpoint-bucket
export WANDB_API_KEY=your_wandb_key

sky launch -c $SLIME_CLUSTER_NAME sky_configs/train.yaml
```

### Option 3: Use the helper launcher script

```bash
./scripts/sky_launch.sh sky_configs/examples/train_glm4_9b.yaml
```

## Customization

You can customize the configurations by:
1. Copying an example YAML file
2. Modifying the resources, environment variables, or training arguments
3. Launching with your custom config

Example:
```bash
cp sky_configs/examples/train_glm4_9b.yaml my_custom_config.yaml
# Edit my_custom_config.yaml with your preferences
sky launch -c my-cluster my_custom_config.yaml
```

## Key Configuration Options

### Resources
```yaml
resources:
  cloud: aws          # aws, gcp, or azure
  accelerators: H100:8  # GPU type and count
  use_spot: false     # Use spot instances for cost savings
  instance_type: p5.48xlarge  # (Optional) specific instance type
```

### File Mounts (Checkpoint Persistence)
```yaml
file_mounts:
  /slime_checkpoints:
    name: my-checkpoint-bucket  # Your S3/GCS/Azure bucket name
    store: s3                    # s3, gcs, or azure
    mode: MOUNT                  # Bidirectional sync
```

### Environment Variables
```yaml
envs:
  PYTHONPATH: /root/Megatron-LM/
  WANDB_API_KEY: ${WANDB_API_KEY}  # Set before launching
  HF_TOKEN: ${HF_TOKEN}            # Set before launching
```

## Monitoring

**View cluster status:**
```bash
sky status
```

**SSH into the cluster:**
```bash
sky ssh slime-glm4-9b
```

**View logs:**
```bash
sky logs slime-glm4-9b --follow
```

**Execute commands on the cluster:**
```bash
sky exec slime-glm4-9b "nvidia-smi"
```

## Stopping and Cleanup

**Stop the cluster** (keeps it for later resume):
```bash
sky stop slime-glm4-9b
```

**Resume a stopped cluster:**
```bash
sky start slime-glm4-9b
```

**Terminate the cluster completely:**
```bash
sky down slime-glm4-9b
```

**List all clusters:**
```bash
sky status
```

## Cost Optimization

### Use Spot Instances
Spot instances can reduce costs by up to 70%:
```yaml
resources:
  use_spot: true
```

Note: Spot instances may be preempted. SkyPilot will automatically retry, and checkpoints will be preserved in cloud storage.

### Multi-Cloud Support
SkyPilot can automatically select the cheapest cloud provider:
```yaml
resources:
  cloud: ${SLIME_CLOUD}  # Leave unset to auto-select cheapest
```

Or specify multiple clouds to choose from:
```bash
sky launch --cloud aws,gcp ...
```

## Checkpoint Management

Checkpoints are automatically synced to cloud storage via `file_mounts`:

1. **Create a cloud storage bucket:**
   ```bash
   # AWS S3
   aws s3 mb s3://my-slime-checkpoints
   
   # Google Cloud Storage
   gsutil mb gs://my-slime-checkpoints
   
   # Azure Blob Storage
   az storage container create --name my-slime-checkpoints
   ```

2. **Update the bucket name in your YAML:**
   ```yaml
   file_mounts:
     /slime_checkpoints:
       name: my-slime-checkpoints
   ```

3. **Checkpoints are automatically synced:**
   - During training: Checkpoints saved to `/slime_checkpoints/` are automatically uploaded
   - On resume: Checkpoints are automatically downloaded from cloud storage

## Differences from Manual Setup

| Manual Setup | SkyPilot |
|-------------|----------|
| Manual `ray start --head` | Automatic Ray cluster setup |
| SSH to workers manually | Automatic worker coordination |
| `ray job submit` | Direct command execution |
| Manual checkpoint management | Automatic cloud storage sync |
| Manual instance provisioning | Automatic cluster provisioning |
| Manual environment setup | Declarative setup in YAML |

## Advanced Features

### Smart Resumption
SkyPilot can reuse existing Ray clusters:
```bash
sky launch -c existing-cluster sky_configs/train.yaml
```

If the cluster already exists and Ray is running, SkyPilot will submit the new job without recreating the cluster.

### Multiple Clouds
Launch jobs on different clouds:
```bash
# Launch on AWS
sky launch -c aws-cluster --cloud aws sky_configs/train.yaml

# Launch on GCP
sky launch -c gcp-cluster --cloud gcp sky_configs/train.yaml
```

### Autostop
Automatically stop clusters after idle time to save costs:
```bash
sky launch -c my-cluster --down-if-idle-minutes 30 sky_configs/train.yaml
```

## Troubleshooting

### Common Issues

**Issue: SkyPilot cannot find credentials**
```bash
# Verify credentials are configured
aws sts get-caller-identity  # For AWS
gcloud auth list             # For GCP
az account show              # For Azure
```

**Issue: Insufficient quota for GPUs**
- Check your cloud provider's quota limits
- Request quota increases if needed
- Try a different region or cloud provider

**Issue: File mount errors**
- Verify bucket exists and credentials have access
- Check bucket name matches in YAML
- Ensure bucket is in the same region as instances (for best performance)

**Issue: Training job fails**
- Check logs: `sky logs <cluster-name> --follow`
- SSH into cluster: `sky ssh <cluster-name>`
- Verify model/data paths are correct
- Check environment variables are set

### Getting Help

1. **Check SkyPilot status:**
   ```bash
   sky status --all
   ```

2. **View detailed logs:**
   ```bash
   sky logs <cluster-name> --follow
   ```

3. **SSH into cluster for debugging:**
   ```bash
   sky ssh <cluster-name>
   ```

4. **Verify cloud credentials:**
   ```bash
   sky check
   ```

## Example Workflows

### Development Workflow
```bash
# Launch small cluster for testing
sky launch -c dev-cluster sky_configs/examples/train_glm4_9b.yaml

# Monitor progress
sky logs dev-cluster --follow

# Stop when done testing
sky stop dev-cluster

# Clean up
sky down dev-cluster
```

### Production Training Workflow
```bash
# Launch large multi-node cluster
export SLIME_CHECKPOINT_BUCKET=production-checkpoints
export WANDB_API_KEY=xxx
sky launch -c prod-training sky_configs/examples/train_glm4_355b.yaml

# Monitor from local machine
sky logs prod-training --follow

# Checkpoints automatically saved to S3/GCS

# Training completes, cluster auto-stops after idle
```

### Multi-Experiment Workflow
```bash
# Launch multiple experiments in parallel
sky launch -c exp1 experiment1.yaml &
sky launch -c exp2 experiment2.yaml &
sky launch -c exp3 experiment3.yaml &

# Monitor all experiments
sky status

# Each experiment saves to separate checkpoint buckets
```

## Best Practices

1. **Use spot instances** for development and non-critical training
2. **Set up checkpoint persistence** to cloud storage for all training runs
3. **Use meaningful cluster names** to track experiments
4. **Set autostop timers** to avoid unnecessary costs
5. **Test configurations on small clusters** before scaling up
6. **Monitor costs** using cloud provider dashboards
7. **Clean up unused clusters** regularly with `sky down`

## Additional Resources

- [SkyPilot Documentation](https://skypilot.readthedocs.io/)
- [SkyPilot GitHub](https://github.com/skypilot-org/skypilot)
- [slime Migration Guide](../docs/en/get_started/skypilot_migration.md)
- [VERL + SkyPilot Example](https://docs.skypilot.co/en/latest/examples/training/verl.html)
