#!/bin/bash
# Phase 1 v3: Single-step OPD Baseline
# Qwen3-1.7B (post-trained) student + Qwen3-8B teacher
# Key difference from v2: student is post-trained (has thinking mode, follows instructions)
set -ex

STUDENT_HF=/root/.cache/huggingface/Qwen3-1.7B
STUDENT_TORCH_DIST=/root/.cache/huggingface/Qwen3-1.7B_torch_dist
TEACHER_HF=/root/.cache/huggingface/Qwen3-8B
TRAIN_DATA=/workspace/data/dapo-math-17k/dapo-math-17k.jsonl
SAVE_DIR=/workspace/k-step-opd/checkpoints/phase1v3-baseline
EXPERIMENT_NAME="phase1v3-opd-k1-qwen3-1.7b-8b"

mkdir -p $SAVE_DIR

# Teacher SGLang Server (TP=2 on GPU 6,7)
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141

CUDA_VISIBLE_DEVICES=6,7 python3 -m sglang.launch_server \
    --model-path $TEACHER_HF \
    --host 0.0.0.0 --port $TEACHER_PORT \
    --tp 2 --chunked-prefill-size 4096 --mem-fraction-static 0.6 \
    > /tmp/sglang_teacher.log 2>&1 &

echo "Starting teacher server..."
until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
    echo "Waiting..."; sleep 5
done
echo "Teacher ready."
sleep 5

# Model Config
source /root/slime/scripts/models/qwen3-1.7B.sh

# Training args
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"env_vars":{"PYTHONPATH":"/root/Megatron-LM/","CUDA_DEVICE_MAX_CONNECTIONS":"1"}}' \
   -- python3 /root/slime/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $STUDENT_HF \
   --ref-load $STUDENT_TORCH_DIST \
   --load $STUDENT_TORCH_DIST \
   --save $SAVE_DIR \
   --save-interval 25 \
   --prompt-data $TRAIN_DATA \
   --input-key prompt \
   --apply-chat-template \
   --rollout-shuffle \
   --num-rollout 200 \
   --rollout-batch-size 8 \
   --n-samples-per-prompt 4 \
   --rollout-max-response-len 8192 \
   --rollout-temperature 0.6 \
   --rollout-top-p 0.95 \
   --rollout-top-k 20 \
   --global-batch-size 32 \
   --balance-data \
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func \
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards \
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate \
   --tensor-model-parallel-size 2 \
   --sequence-parallel \
   --pipeline-model-parallel-size 1 \
   --context-parallel-size 1 \
   --expert-model-parallel-size 1 \
   --expert-tensor-parallel-size 1 \
   --recompute-granularity full \
   --recompute-method uniform \
   --recompute-num-layers 1 \
   --use-dynamic-batch-size \
   --max-tokens-per-gpu 8192 \
   --advantage-estimator grpo \
   --use-opd \
   --opd-type sglang \
   --opd-kl-coef 1.0 \
   --use-kl-loss \
   --kl-loss-coef 0.00 \
   --kl-loss-type low_var_kl \
   --entropy-coef 0.00 \
   --optimizer adam \
   --lr 5e-7 \
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
ray stop --force 2>/dev/null
