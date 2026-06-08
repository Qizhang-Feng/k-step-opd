#!/bin/bash
# Pilot OPD smoke test: Qwen3-4B student + Qwen3-8B teacher
# Run inside the k-step-opd docker container on p5-6
# Usage: bash /workspace/k-step-opd/run-pilot-opd.sh

set -ex

# ============ Paths ============
STUDENT_HF=/root/.cache/huggingface/Qwen3-4B
STUDENT_TORCH_DIST=/root/.cache/huggingface/Qwen3-4B_torch_dist
TEACHER_HF=/root/.cache/huggingface/Qwen3-8B
TRAIN_DATA=/workspace/data/dapo-math-17k/dapo-math-17k.jsonl

# ============ Teacher SGLang Server ============
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141
LOG_FILE="/tmp/sglang_teacher.log"

# Launch teacher on GPU 7 (reserve GPUs 0-6 for training + rollout)
CUDA_VISIBLE_DEVICES=7 python3 -m sglang.launch_server \
    --model-path $TEACHER_HF \
    --host 0.0.0.0 \
    --port $TEACHER_PORT \
    --tp 1 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.6 \
    > "$LOG_FILE" 2>&1 &

echo "Starting teacher SGLang server..."
until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
    echo "Waiting for teacher server..."
    tail -n 5 "$LOG_FILE"
    sleep 5
done
echo "Teacher server ready at $TEACHER_IP:$TEACHER_PORT"
sleep 5

# ============ Model Config (Qwen3-4B) ============
source /root/slime/scripts/models/qwen3-4B.sh

# ============ Args ============
CKPT_ARGS=(
   --hf-checkpoint $STUDENT_HF
   --ref-load $STUDENT_TORCH_DIST
   --load $STUDENT_TORCH_DIST
   --save /workspace/k-step-opd/checkpoints/pilot-opd/
   --save-interval 999999
)

ROLLOUT_ARGS=(
   --prompt-data $TRAIN_DATA
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 50
   --rollout-batch-size 8
   --n-samples-per-prompt 2
   --rollout-max-response-len 2048
   --rollout-temperature 1
   --global-batch-size 16
   --balance-data
)

RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
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
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ============ Launch Ray ============
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# ============ Run Training ============
# 2 GPUs for actor (training), 4 GPUs for rollout (SGLang inference), 1 GPU for teacher
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 /root/slime/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

# ============ Cleanup ============
pkill -9 sglang || true
sleep 3
ray stop --force
pkill -9 ray || true
pkill -9 python || true
sleep 3
