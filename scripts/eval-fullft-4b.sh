#!/bin/bash
# Eval full FT 4B checkpoints (500, 1000, 1500) on AIME-2024 and AIME-2025
# Run on p5-5 inside k-step-opd container
set -ex

PORT=30010
MAX_TOKENS=30000
N_SAMPLES=1
TEMPERATURE=0.6
EVAL_DIR=/workspace/k-step-opd/eval_results_4b_fullft

mkdir -p $EVAL_DIR

for STEP in 500 1000 1500; do
    MODEL_PATH=/root/.cache/huggingface/sft-qwen3-4b-fullft-${STEP}
    MODEL_NAME="Qwen3-4B-Base-FullFT-${STEP}"

    echo ""
    echo "============================================"
    echo "=== Evaluating $MODEL_NAME ==="
    echo "============================================"

    # Kill any existing server
    pkill -9 -f sglang 2>/dev/null || true
    sleep 5

    # Start server
    python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --port $PORT \
        --tp 8 \
        --trust-remote-code \
        --mem-fraction-static 0.85 \
        --max-running-requests 2 \
        --max-total-tokens 32768 \
        > /tmp/eval_server_${STEP}.log 2>&1 &

    # Wait for server
    echo "Waiting for server..."
    for i in $(seq 1 180); do
        if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            break
        fi
        if [ $i -eq 180 ]; then
            echo "ERROR: Server failed to start"
            cat /tmp/eval_server_${STEP}.log | tail -20
            exit 1
        fi
        sleep 1
    done

    SERVER_URL="http://127.0.0.1:$PORT"

    # AIME-2024
    echo "--- AIME-2024 ---"
    python3 /workspace/k-step-opd/eval_math.py \
        --server-url $SERVER_URL \
        --data-path /workspace/data/aime-2024/aime-2024.jsonl \
        --output-path $EVAL_DIR/aime2024_step${STEP}.json \
        --n-samples $N_SAMPLES \
        --max-tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --dataset-name aime2024 \
        --model-name "$MODEL_NAME" \
        --max-workers 1

    # AIME-2025
    echo "--- AIME-2025 ---"
    python3 /workspace/k-step-opd/eval_math.py \
        --server-url $SERVER_URL \
        --data-path /workspace/data/aime-2025/aime-2025.jsonl \
        --output-path $EVAL_DIR/aime2025_step${STEP}.json \
        --n-samples $N_SAMPLES \
        --max-tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --dataset-name aime2025 \
        --model-name "$MODEL_NAME" \
        --max-workers 1

    # Stop server
    pkill -9 -f sglang 2>/dev/null || true
    sleep 3

    echo "=== $MODEL_NAME done ==="
done

# Summary
echo ""
echo "=========================================="
echo "=== FINAL SUMMARY ==="
echo "=========================================="
for f in $EVAL_DIR/*.json; do
    python3 -c "
import json
with open('$f') as fp:
    d = json.load(fp)
print(f\"{d['model_name']:30s} | {d['dataset_name']:10s} | pass@1={d['pass_at_1']*100:.1f}% | avg_len={d['avg_response_length']:.0f}\")
"
done
echo "=========================================="
