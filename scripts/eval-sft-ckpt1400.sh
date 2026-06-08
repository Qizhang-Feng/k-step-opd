#!/bin/bash
# Eval 4B Full FT 152K checkpoint-1400 on AIME-2024 + AIME-2025
# Server already running on p5-3 at port 30000 (dp=8)
# Run: ssh p5-3 "docker exec k-step-opd bash /workspace/k-step-opd/scripts/eval-sft-ckpt1400.sh"

set -e

SERVER_URL="http://127.0.0.1:30000"
MODEL_NAME="4B-FullFT-152k-ckpt1400"
EVAL_DIR="/workspace/k-step-opd/eval_results_152k_ckpt1400"
MAX_TOKENS=30000
N_SAMPLES=1
TEMPERATURE=0.6

mkdir -p $EVAL_DIR

echo "=== Waiting for server at $SERVER_URL ==="
for i in $(seq 1 120); do
    if curl -s $SERVER_URL/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: Server not ready after 120s"
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== AIME-2024 ==="
python3 /workspace/k-step-opd/eval_math.py \
    --server-url $SERVER_URL \
    --data-path /workspace/data/aime-2024/aime-2024.jsonl \
    --output-path $EVAL_DIR/aime2024.json \
    --n-samples $N_SAMPLES \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --dataset-name aime2024 \
    --model-name "$MODEL_NAME" \
    --max-workers 1

echo ""
echo "=== AIME-2025 ==="
python3 /workspace/k-step-opd/eval_math.py \
    --server-url $SERVER_URL \
    --data-path /workspace/data/aime-2025/aime-2025.jsonl \
    --output-path $EVAL_DIR/aime2025.json \
    --n-samples $N_SAMPLES \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --dataset-name aime2025 \
    --model-name "$MODEL_NAME" \
    --max-workers 1

echo ""
echo "=========================================="
echo "=== RESULTS: $MODEL_NAME ==="
echo "=========================================="
for f in $EVAL_DIR/*.json; do
    python3 -c "
import json
with open('$f') as fp:
    d = json.load(fp)
print(f\"{d['dataset_name']:10s} | pass@1={d['pass_at_1']*100:.1f}% | avg_len={d['avg_response_length']:.0f} chars\")
"
done
echo "=========================================="
