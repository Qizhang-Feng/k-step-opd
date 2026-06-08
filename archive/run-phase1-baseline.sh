#!/bin/bash
# Phase 1: Single-step OPD Baseline
# Qwen3-4B student + Qwen3-8B teacher, pure OPD on dapo-math-17k
# Run inside k-step-opd container on p5-6

set -ex

# ============ Paths ============
STUDENT_HF=/root/.cache/huggingface/Qwen3-4B
STUDENT_TORCH_DIST=/root/.cache/huggingface/Qwen3-4B_torch_dist
TEACHER_HF=/root/.cache/huggingface/Qwen3-8B
TRAIN_DATA=/workspace/data/dapo-math-17k/dapo-math-17k.jsonl
EVAL_DATA=/workspace/data/aime-2024/aime-2024.jsonl
SAVE_DIR=/workspace/k-step-opd/checkpoints/phase1-baseline
EXPERIMENT_NAME="phase1-opd-k1-qwen3-4b-8b"

mkdir -p $SAVE_DIR

# ============ Teacher SGLang Server ============
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141
LOG_FILE="/tmp/sglang_teacher.log"

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
    tail -n 3 "$LOG_FILE"
    sleep 5
done
echo "Teacher server ready."
sleep 5

# ============ Model Config (Qwen3-4B) ============
source /root/slime/scripts/models/qwen3-4B.sh

# ============ Args ============
CKPT_ARGS=(
   --hf-checkpoint $STUDENT_HF
   --ref-load $STUDENT_TORCH_DIST
   --load $STUDENT_TORCH_DIST
   --save $SAVE_DIR
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data $TRAIN_DATA
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 200
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 32
   --balance-data
)

RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)

EVAL_ARGS=(
   # Eval disabled for now - OPD custom RM conflicts with eval reward logging
   # Will run eval separately after training
   # --eval-interval 50
   # --eval-prompt-data math500 /workspace/data/math-500/math-500.jsonl aime $EVAL_DATA
   # --n-samples-per-eval-prompt 4
   # --eval-max-response-len 4096
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

WANDB_ARGS=(
   # Uncomment to enable wandb:
   # --use-wandb
   # --wandb-project k-step-opd
   # --wandb-group phase1-baseline
   # --wandb-key ${WANDB_KEY}
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
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
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
