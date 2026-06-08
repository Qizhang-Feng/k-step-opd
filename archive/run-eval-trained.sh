#!/bin/bash
# Eval trained student only on MATH-500 and AIME-2024
set -ex

TRAINED_HF=/workspace/k-step-opd/checkpoints/phase1-baseline/iter_0000199_hf
MATH500_DATA=/workspace/data/math-500/math-500.jsonl
AIME_DATA=/workspace/data/aime-2024/aime-2024.jsonl
EVAL_SCRIPT=/workspace/k-step-opd/eval_math.py
OUTPUT_DIR=/workspace/k-step-opd/eval_results
PORT=30000

export PYTHONUNBUFFERED=1

# Kill any leftover processes
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 -f "eval_math" 2>/dev/null || true
sleep 3

echo "GPU memory:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# Start SGLang server for trained student (TP=1, 4B model)
python3 -m sglang.launch_server \
    --model-path $TRAINED_HF \
    --host 0.0.0.0 \
    --port $PORT \
    --tp 1 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.85 \
    > /tmp/sglang_eval_trained.log 2>&1 &
SERVER_PID=$!

echo "Starting SGLang server (PID=$SERVER_PID)..."
MAX_WAIT=300
WAITED=0
until curl -sf http://127.0.0.1:$PORT/health_generate > /dev/null; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: Server failed to start"
        tail -20 /tmp/sglang_eval_trained.log
        exit 1
    fi
    echo "  Waiting... (${WAITED}s)"
done
echo "Server ready!"
sleep 3

# MATH-500
echo "=== Evaluating trained student on MATH-500 ==="
python3 $EVAL_SCRIPT \
    --server-url http://127.0.0.1:$PORT \
    --data-path $MATH500_DATA \
    --output-path $OUTPUT_DIR/trained_student_qwen3-4b_math500.json \
    --n-samples 8 \
    --max-tokens 16384 \
    --temperature 0.6 \
    --dataset-name math500

# AIME-2024
echo "=== Evaluating trained student on AIME-2024 ==="
python3 $EVAL_SCRIPT \
    --server-url http://127.0.0.1:$PORT \
    --data-path $AIME_DATA \
    --output-path $OUTPUT_DIR/trained_student_qwen3-4b_aime.json \
    --n-samples 8 \
    --max-tokens 16384 \
    --temperature 0.6 \
    --dataset-name aime

# Cleanup
kill -9 $SERVER_PID 2>/dev/null || true
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
sleep 3

echo "=== Done ==="
