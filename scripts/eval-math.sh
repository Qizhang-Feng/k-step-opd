#!/bin/bash
# Unified eval script
# Usage: bash scripts/eval-math.sh <model-path> <output-name> [dp=8] [max-tokens=16384]
# Example:
#   bash scripts/eval-math.sh /root/.cache/huggingface/Qwen3-1.7B student_init_1.7b
#   bash scripts/eval-math.sh /workspace/k-step-opd/checkpoints/phase1v3-baseline/iter_0000199_hf trained_1.7b
set -ex

MODEL_PATH=${1:?Usage: bash scripts/eval-math.sh <model-path> <output-name> [dp] [max-tokens]}
OUTPUT_NAME=${2:?Usage: bash scripts/eval-math.sh <model-path> <output-name> [dp] [max-tokens]}
DP=${3:-8}
MAX_TOKENS=${4:-16384}

MATH500_DATA=/workspace/data/math-500/math-500.jsonl
EVAL_SCRIPT=/workspace/k-step-opd/eval_math.py
OUTPUT_DIR=/workspace/k-step-opd/eval_results
PORT=30000

export PYTHONUNBUFFERED=1
mkdir -p $OUTPUT_DIR

echo "=== Eval: $OUTPUT_NAME (dp=$DP, max_tokens=$MAX_TOKENS) ==="

python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host 0.0.0.0 --port $PORT \
    --dp $DP --tp 1 \
    --chunked-prefill-size 4096 --mem-fraction-static 0.85 \
    > /tmp/sglang_eval.log 2>&1 &
PID=$!

until curl -sf http://127.0.0.1:$PORT/health_generate > /dev/null; do sleep 5; done
echo "Server ready"
sleep 3

python3 $EVAL_SCRIPT \
    --server-url http://127.0.0.1:$PORT \
    --data-path $MATH500_DATA \
    --output-path $OUTPUT_DIR/${OUTPUT_NAME}_math500.json \
    --n-samples 8 --max-tokens $MAX_TOKENS --temperature 0.6 \
    --dataset-name math500

kill -9 $PID 2>/dev/null || true
# Kill all sglang child processes
pkill -9 -f "sglang::scheduler" 2>/dev/null || true
pkill -9 -f "sglang::detokenizer" 2>/dev/null || true
sleep 2
echo "=== Done: $OUTPUT_NAME ==="
