#!/bin/bash
# Phase 1 v2: Single-step OPD Baseline (positive control)
# Qwen3-1.7B-Base student + Qwen3-8B teacher, pure OPD on dapo-math-17k
#
# Key changes from v1:
#   - Student: 1.7B-Base (weaker, ~60-70% MATH-500) instead of 4B (91.6%)
#   - Teacher: 8B with TP=2 (stronger relative to student)
#   - max_response_len: 8192 (was 4096, reduce truncation)
#   - max_tokens_per_gpu: 8192
#   - lr: 5e-7 (was 1e-6)
#   - sampling: temperature=0.6, top_p=0.95, top_k=20 (Qwen3 thinking mode)
#
# Run inside k-step-opd container on a p5 node.

set -ex

# ============ Paths ============
STUDENT_HF=/root/.cache/huggingface/Qwen3-1.7B-Base
STUDENT_TORCH_DIST=/root/.cache/huggingface/Qwen3-1.7B-Base_torch_dist
TEACHER_HF=/root/.cache/huggingface/Qwen3-8B
TRAIN_DATA=/workspace/data/dapo-math-17k/dapo-math-17k.jsonl
SAVE_DIR=/workspace/k-step-opd/checkpoints/phase1v2-baseline-A
EXPERIMENT_NAME="phase1v2-opd-k1-qwen3-1.7b-base-8b-8192"

mkdir -p $SAVE_DIR

# ============ Teacher SGLang Server (TP=2 on GPU 6,7) ============
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141
LOG_FILE="/tmp/sglang_teacher.log"

CUDA_VISIBLE_DEVICES=6,7 python3 -m sglang.launch_server \
    --model-path $TEACHER_HF \
    --host 0.0.0.0 \
    --port $TEACHER_PORT \
    --tp 2 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.6 \
    > "$LOG_FILE" 2>&1 &

echo "Starting teacher SGLang server (TP=2 on GPU 6,7)..."
until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
    echo "Waiting for teacher server..."
    tail -n 3 "$LOG_FILE"
    sleep 5
done
echo "Teacher server ready."
sleep 5

# ============ Model Config (Qwen3-1.7B) ============
source /root/slime/scripts/models/qwen3-1.7B.sh

# ============ Args ============
CKPT_ARGS=(
   --hf-checkpoint $STUDENT_HF
   --ref-load $STUDENT_TORCH_DIST
   --load $STUDENT_TORCH_DIST
   --save $SAVE_DIR
   --save-interval 25
)

ROLLOUT_ARGS=(
   --prompt-data $TRAIN_DATA
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 200
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 8192
   --rollout-temperature 0.6
   --rollout-top-p 0.95
   --rollout-top-k 20
   --global-batch-size 32
   --balance-data
)

RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)

EVAL_ARGS=(
   # Eval disabled - will run separately after training
   # OPD custom RM conflicts with eval reward logging
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
   --max-tokens-per-gpu 8192
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
   --lr 5e-7
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # Uncomment to enable wandb:
   # --use-wandb
   # --wandb-project k-step-opd
   # --wandb-group phase1v2-baseline
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
sleep 3
