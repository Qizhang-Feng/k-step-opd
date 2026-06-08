#!/bin/bash
# Eval Phase 1 v2: Student init (1.7B-Base) + Trained student on MATH-500
set -ex

STUDENT_HF=/root/.cache/huggingface/Qwen3-1.7B-Base
TRAINED_HF=/workspace/k-step-opd/checkpoints/phase1v2-baseline-A/iter_0000199_hf
MATH500_DATA=/workspace/data/math-500/math-500.jsonl
EVAL_SCRIPT=/workspace/k-step-opd/eval_math.py
OUTPUT_DIR=/workspace/k-step-opd/eval_results_v2
PORT=30000

export PYTHONUNBUFFERED=1
mkdir -p $OUTPUT_DIR

eval_model() {
    local MODEL_PATH=$1
    local MODEL_NAME=$2

    echo "=== Evaluating: $MODEL_NAME ==="
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 5

    python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --host 0.0.0.0 --port $PORT \
        --tp 1 --chunked-prefill-size 4096 --mem-fraction-static 0.85 \
        > /tmp/sglang_eval_${MODEL_NAME}.log 2>&1 &
    local PID=$!

    until curl -sf http://127.0.0.1:$PORT/health_generate > /dev/null; do sleep 5; done
    echo "Server ready"
    sleep 3

    python3 $EVAL_SCRIPT \
        --server-url http://127.0.0.1:$PORT \
        --data-path $MATH500_DATA \
        --output-path $OUTPUT_DIR/${MODEL_NAME}_math500.json \
        --n-samples 8 --max-tokens 16384 --temperature 0.6 \
        --dataset-name math500

    kill -9 $PID 2>/dev/null || true
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 8
}

eval_model $STUDENT_HF "student_init_qwen3-1.7b-base"
eval_model $TRAINED_HF "trained_student_qwen3-1.7b-base"

echo "=== All done ==="
