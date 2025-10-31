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
source "${SCRIPT_DIR}/models/tim-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint SubconsciousDev/TIM-8b-long-grpo
   --ref-load /lambda/nfs/models/model_torch_dist
   --load /workspace/checkpoints/
   --save /workspace/checkpoints/
   --save-interval 20
)

ROLLOUT_ARGS=(
   # --prompt-data /root/OpenR1-Math-220k/
   # --input-key problem
   # --label-key answer
   --prompt-data /lambda/nfs/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 10
   --rollout-batch-size 4
   --n-samples-per-prompt 2
   --rollout-max-response-len 16384
   --rollout-temperature 0.8
   --global-batch-size 8 # (rollout-batch-size × n-samples-per-prompt) = (global-batch-size × num-steps-per-rollout)
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /lambda/nfs/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 2

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --wandb-mode online
   --use-wandb
   --wandb-team autonomous-nlp
   --wandb-project slime-dev
   --wandb-group test-tim
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus 4
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.65
)

DEBUG_ARGS=(
   --sglang-enable-metrics
   --save-debug-rollout-data /lambda/nfs/{run_id}/rollout_{rollout_id}.pt
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
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
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
   -- python3 train_async.py \
   --actor-num-nodes ${ACTOR_NUM_NODES:-1} \
   --actor-num-gpus-per-node 4 \
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
