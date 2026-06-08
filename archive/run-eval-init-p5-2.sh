#!/bin/bash
# Eval student init (1.7B-Base) with fixed grading on p5-2, 8-GPU
set -ex

STUDENT_HF=/root/.cache/huggingface/Qwen3-1.7B-Base
MATH500_DATA=/workspace/data/math-500/math-500.jsonl
EVAL_SCRIPT=/workspace/k-step-opd/eval_math.py
OUTPUT_DIR=/workspace/k-step-opd/eval_results_v2
PORT=30000

export PYTHONUNBUFFERED=1
mkdir -p $OUTPUT_DIR

sleep 1

echo "=== Starting 8-GPU server for student init ==="
python3 -m sglang.launch_server \
    --model-path $STUDENT_HF \
    --host 0.0.0.0 --port $PORT \
    --dp 8 --tp 1 \
    --chunked-prefill-size 4096 --mem-fraction-static 0.85 \
    > /tmp/sglang_eval_init.log 2>&1 &
PID=$!

echo "Waiting for server..."
until curl -sf http://127.0.0.1:$PORT/health_generate > /dev/null; do sleep 5; done
echo "Server ready"
sleep 3

echo "=== Evaluating student init (1.7B-Base) on MATH-500 ==="
python3 $EVAL_SCRIPT \
    --server-url http://127.0.0.1:$PORT \
    --data-path $MATH500_DATA \
    --output-path $OUTPUT_DIR/student_init_qwen3-1.7b-base_math500_fixed.json \
    --n-samples 8 --max-tokens 16384 --temperature 0.6 \
    --dataset-name math500

kill -9 $PID 2>/dev/null || true

echo "=== Done ==="
