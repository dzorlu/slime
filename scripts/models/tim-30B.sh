NLAYERS=48
arr=()
for ((i=0; i<NLAYERS; i++)); do arr+=(1); done
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

MODEL_ARGS=(
   --num-layers 48
   --hidden-size 2048
   --ffn-hidden-size 6144
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 4
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 10000000
   --vocab-size 151936
   --kv-channels 128
   --untie-embeddings-and-output-weights

   # moe
   --moe-ffn-hidden-size 768
   --moe-shared-expert-intermediate-size 768
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 8
   --moe-layer-freq $MOE_LAYER_FREQ
   --num-experts 128
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
)


