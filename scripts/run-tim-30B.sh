#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# Detect NVLink with a hard timeout; default to 0 if it hangs/unavailable
NVLINK_TOPO="$(timeout 10s nvidia-smi topo -m 2>/dev/null || true)"
if [ -n "$NVLINK_TOPO" ]; then
  NVLINK_COUNT="$(echo "$NVLINK_TOPO" | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)"
else
  NVLINK_COUNT=0
  echo -e "\e[31m\e[1m"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "!! WARNING: nvidia-smi topo -m unavailable or timed out.   !!"
  echo "!! Proceeding without NVLink optimizations. This may       !!"
  echo "!! slightly impact performance but is otherwise safe.        !!"
  echo "!! If this persists, consider rebooting the node.            !!"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo -e "\e[0m"
fi

if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen3-30B-A3B
   --ref-load /lambda/nfs/models/model_torch_dist_30B
   --load /lambda/nfs/checkpoints/
   --save /lambda/nfs/checkpoints/
   --save-interval 40
   --no-save-optim
)

ROLLOUT_ARGS=(
   #--prompt-data /lambda/nfs/OpenR1-Math-220k-formatted/data/train-00000-of-00027.parquet
   --prompt-data /lambda/nfs/OpenR1-Math-220k-formatted/data/train0-5.parquet
   --input-key message
   --label-key answer
   --apply-chat-template
   #--rollout-shuffle
   --rm-type math # accepts a boxed answer anywhere in the response.
   --num-rollout 1000
   --rollout-batch-size 32 # H100: 4, B200: 32
   --n-samples-per-prompt 16 # (rollout-batch-size × n-samples-per-prompt) = (global-batch-size × num-steps-per-rollout)
   --rollout-max-response-len 16384 # H100: 8192, B200: 16384
   --rollout-temperature 0.9 
   #--global-batch-size 256 # H100: 32, B200: 256
   --num-steps-per-rollout 4 # take 4 steps per rollout.
   #--balance-data
)

EVAL_ARGS=(
   --eval-interval 40
   --eval-prompt-data aime /lambda/nfs/aime-2024/aime-2024.jsonl
   --eval-input-key prompt
   --eval-label-key label
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 2

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480 # H100: 8192, B200: 20480
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss # load the rerence model.
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00 # loss.
   --eps-clip 0.2
   --eps-clip-high 0.28
   --calculate-per-token-loss
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

#    --optimizer-cpu-offload
#    --overlap-cpu-optimizer-d2h-h2d
#    --use-precision-aware-optimizer

)

WANDB_ARGS=(
   --wandb-mode online
   --use-wandb
   --wandb-team autonomous-nlp
   --wandb-project OpenR1-Math-220k
   --wandb-group tim-30b
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus 8
   --rollout-num-gpus-per-engine 1 # H100: 4, B200: 1. more parallelization.
   --sglang-mem-fraction-static 0.5
   --sglang-expert-parallel-size 1
   #--sglang-moe-a2a-backend deepep
)

DEBUG_ARGS=(
   #--sglang-enable-metrics
   --save-debug-rollout-data /lambda/nfs/rollouts/{run_id}/rollout_{rollout_id}.pt
   --record-memory-history
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   #--accumulate-allreduce-grads-in-fp32
   #--attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
  --object-store-memory 12884901888 \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"32\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# Wait for dashboard and Jobs API to be ready (inside the container)
for i in {1..60}; do
  if curl -sf http://127.0.0.1:8265/api/version >/dev/null && \
     curl -sf http://127.0.0.1:8265/api/jobs/ >/dev/null; then
    echo "Ray dashboard/Jobs API is ready..."
    break
  fi
  echo "Waiting for Ray dashboard/Jobs API to be ready..."
  sleep 1
done


sleep 60
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes ${ACTOR_NUM_NODES:-1} \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --log-passrate \
   --use-slime-router \
   --custom-generate-function-path examples.subc.generate_with_constraint.generate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${DEBUG_ARGS[@]} \
   ${MISC_ARGS[@]}


