#!/bin/bash
set -ex

MODEL_PATH=${1:-/root/.cache/huggingface/opd-cumulative-hf}
MODEL_NAME=${2:-OPD-Cumulative}
EVAL_DIR=/root/.cache/huggingface/eval_${MODEL_NAME// /_}
PORT=30010

mkdir -p $EVAL_DIR

echo "=== Starting SGLang server for $MODEL_PATH ==="
python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port $PORT \
    --tp 8 \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --max-running-requests 2 \
    --max-total-tokens 32768 \
    > /tmp/eval_server.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
    echo "ERROR: Server failed to start"
    cat /tmp/eval_server.log | tail -30
    exit 1
fi

SERVER_URL="http://127.0.0.1:$PORT"

echo "=== Evaluating AIME-2024 ==="
python3 /workspace/k-step-opd/eval_math.py \
    --server-url $SERVER_URL \
    --data-path /workspace/data/aime-2024/aime-2024.jsonl \
    --output-path $EVAL_DIR/aime2024.json \
    --n-samples 1 \
    --max-tokens 8192 \
    --temperature 0.6 \
    --dataset-name aime2024 \
    --model-name "$MODEL_NAME" \
    --max-workers 1

echo "=== Evaluating AIME-2025 ==="
python3 /workspace/k-step-opd/eval_math.py \
    --server-url $SERVER_URL \
    --data-path /workspace/data/aime-2025/aime-2025.jsonl \
    --output-path $EVAL_DIR/aime2025.json \
    --n-samples 1 \
    --max-tokens 8192 \
    --temperature 0.6 \
    --dataset-name aime2025 \
    --model-name "$MODEL_NAME" \
    --max-workers 1

pkill -f sglang || true
sleep 3

echo ""
echo "=========================================="
echo "=== EVAL RESULTS ==="
echo "=========================================="
for f in $EVAL_DIR/*.json; do
    python3 -c "
import json
with open('$f') as fp:
    d = json.load(fp)
print(f\"{d['dataset_name']:10s} | pass@1={d['pass_at_1']*100:.1f}% | avg_len={d['avg_response_length']:.0f}\")
"
done
echo "=========================================="
