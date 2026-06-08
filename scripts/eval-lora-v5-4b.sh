#!/bin/bash
# Eval LoRA v5 4B merged model on AIME-2024 and AIME-2025
# Run on p5-4 inside k-step-opd container
# Config: TP=1, n=8 (pass@k=8), max_tokens=16384, temperature=0.6
set -ex

PORT=8100
MAX_TOKENS=30000
N_SAMPLES=8
TEMPERATURE=0.6
MODEL_PATH=/root/.cache/huggingface/sft-qwen3-4b-lora-v5-merged
MODEL_NAME="Qwen3-4B-Base-LoRA-v5"
EVAL_DIR=/workspace/k-step-opd/eval_results_4b_lora_v5

mkdir -p $EVAL_DIR

# Kill any existing server
pkill -9 -f sglang 2>/dev/null || true
sleep 5

# Start server (DP=8, TP=1: 8 engines, one per GPU, single port)
python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port $PORT \
    --dp-size 8 \
    --tp 1 \
    --trust-remote-code \
    --max-total-tokens 32768 \
    --host 0.0.0.0 \
    > /tmp/eval_server_lora_v5.log 2>&1 &

# Wait for server
echo "Waiting for server (8 DP engines)..."
for i in $(seq 1 180); do
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: Server failed to start"
        tail -20 /tmp/eval_server_lora_v5.log
        exit 1
    fi
    sleep 1
done

SERVER_URL="http://127.0.0.1:$PORT"

# AIME-2024
echo "--- AIME-2024 (n=$N_SAMPLES, max_tokens=$MAX_TOKENS) ---"
python3 /workspace/k-step-opd/eval_math.py \
    --server-url $SERVER_URL \
    --data-path /workspace/data/aime-2024/aime-2024.jsonl \
    --output-path $EVAL_DIR/aime2024_lora_v5.json \
    --n-samples $N_SAMPLES \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --dataset-name aime2024 \
    --model-name "$MODEL_NAME" \
    --max-workers 8

# AIME-2025
echo "--- AIME-2025 (n=$N_SAMPLES, max_tokens=$MAX_TOKENS) ---"
python3 /workspace/k-step-opd/eval_math.py \
    --server-url $SERVER_URL \
    --data-path /workspace/data/aime-2025/aime-2025.jsonl \
    --output-path $EVAL_DIR/aime2025_lora_v5.json \
    --n-samples $N_SAMPLES \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --dataset-name aime2025 \
    --model-name "$MODEL_NAME" \
    --max-workers 8

# Stop server
pkill -9 -f sglang 2>/dev/null || true

# Summary
echo ""
echo "=========================================="
echo "=== RESULTS ==="
echo "=========================================="
for f in $EVAL_DIR/aime*_lora_v5.json; do
    python3 -c "
import json
with open('$f') as fp:
    d = json.load(fp)
print(f\"{d['model_name']:30s} | {d['dataset_name']:10s} | pass@1={d['pass_at_1']*100:.1f}% | pass@{$N_SAMPLES}={d.get('pass_at_any',0)*100:.1f}% | avg_len={d['avg_response_length']:.0f}\")
"
done
echo "=========================================="
