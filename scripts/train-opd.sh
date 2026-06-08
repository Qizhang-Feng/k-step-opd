#!/bin/bash
# Unified OPD training script
# Usage: bash scripts/train-opd.sh <config>
# Example: bash scripts/train-opd.sh configs/phase1v3.env
set -ex

CONFIG=${1:?Usage: bash scripts/train-opd.sh <config-file>}
source "$CONFIG"

# Defaults
TEACHER_PORT=${TEACHER_PORT:-13141}
TEACHER_TP=${TEACHER_TP:-2}
TEACHER_GPUS=${TEACHER_GPUS:-"6,7"}
ACTOR_GPUS=${ACTOR_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-25}
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES=${N_SAMPLES:-4}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
LR=${LR:-5e-7}
OPD_KL_COEF=${OPD_KL_COEF:-1.0}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
OPD_CUMULATIVE=${OPD_CUMULATIVE:-0}
OPD_GAMMA=${OPD_GAMMA:-1.0}
OPD_HORIZON=${OPD_HORIZON:--1}
OPD_DUMP_KL_PATH=${OPD_DUMP_KL_PATH:-}
OPD_DUMP_KL_INTERVAL=${OPD_DUMP_KL_INTERVAL:-1}
OPD_DUMP_KL_MAX_SAMPLES=${OPD_DUMP_KL_MAX_SAMPLES:--1}

mkdir -p $SAVE_DIR

# Build cumulative OPD args
OPD_CUMULATIVE_ARGS=""
if [ "${OPD_CUMULATIVE}" = "1" ]; then
    OPD_CUMULATIVE_ARGS="--opd-cumulative --opd-gamma $OPD_GAMMA --opd-horizon $OPD_HORIZON"
fi

# Build dump KL args
OPD_DUMP_ARGS=""
if [ -n "${OPD_DUMP_KL_PATH}" ]; then
    OPD_DUMP_ARGS="--opd-dump-kl-path ${OPD_DUMP_KL_PATH} --opd-dump-kl-interval ${OPD_DUMP_KL_INTERVAL} --opd-dump-kl-max-samples ${OPD_DUMP_KL_MAX_SAMPLES}"
    mkdir -p "$(dirname "${OPD_DUMP_KL_PATH//\{rollout_id\}/0}")" 2>/dev/null || true
fi

# Start teacher
CUDA_VISIBLE_DEVICES=$TEACHER_GPUS python3 -m sglang.launch_server \
    --model-path $TEACHER_HF \
    --host 0.0.0.0 --port $TEACHER_PORT \
    --tp $TEACHER_TP --chunked-prefill-size 4096 --mem-fraction-static 0.6 \
    > /tmp/sglang_teacher.log 2>&1 &

echo "Starting teacher..."
until curl -sf http://127.0.0.1:$TEACHER_PORT/health_generate > /dev/null; do
    echo "Waiting..."; sleep 5
done
echo "Teacher ready."
sleep 5

# Model config
source /root/slime/scripts/models/${MODEL_CONFIG}.sh

# Ray
rm -rf /tmp/ray 2>/dev/null
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --port 6380

# Train
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"env_vars":{"PYTHONPATH":"/root/Megatron-LM/","CUDA_DEVICE_MAX_CONNECTIONS":"1"}}' \
   -- python3 /root/slime/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node $ACTOR_GPUS \
   --rollout-num-gpus $ROLLOUT_GPUS \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $STUDENT_HF \
   --ref-load $STUDENT_TORCH_DIST \
   --load $STUDENT_TORCH_DIST \
   --save $SAVE_DIR \
   --save-interval $SAVE_INTERVAL \
   --prompt-data $TRAIN_DATA \
   --input-key prompt \
   --apply-chat-template \
   --rollout-shuffle \
   --num-rollout $NUM_ROLLOUT \
   --rollout-batch-size $ROLLOUT_BATCH_SIZE \
   --n-samples-per-prompt $N_SAMPLES \
   --rollout-max-response-len $MAX_RESPONSE_LEN \
   --rollout-temperature $TEMPERATURE \
   --rollout-top-p $TOP_P \
   --rollout-top-k $TOP_K \
   --global-batch-size $GLOBAL_BATCH_SIZE \
   --balance-data \
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func \
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards \
   --rm-url http://127.0.0.1:$TEACHER_PORT/generate \
   --tensor-model-parallel-size ${ACTOR_TP:-2} \
   --sequence-parallel \
   --pipeline-model-parallel-size 1 \
   --context-parallel-size 1 \
   --expert-model-parallel-size 1 \
   --expert-tensor-parallel-size 1 \
   --recompute-granularity full \
   --recompute-method uniform \
   --recompute-num-layers 1 \
   --use-dynamic-batch-size \
   --max-tokens-per-gpu $MAX_TOKENS_PER_GPU \
   --advantage-estimator grpo \
   --use-opd \
   --opd-type sglang \
   --opd-kl-coef $OPD_KL_COEF \
   ${OPD_CUMULATIVE_ARGS} \
   ${OPD_DUMP_ARGS} \
   --use-kl-loss \
   --kl-loss-coef 0.00 \
   --kl-loss-type low_var_kl \
   --entropy-coef 0.00 \
   --optimizer adam \
   --lr $LR \
   --lr-decay-style constant \
   --weight-decay 0.1 \
   --adam-beta1 0.9 \
   --adam-beta2 0.98 \
   --rollout-num-gpus-per-engine 1 \
   --sglang-mem-fraction-static 0.4 \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --attention-backend flash

# Cleanup
ray stop --force 2>/dev/null || true
