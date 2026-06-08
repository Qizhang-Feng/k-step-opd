#!/bin/bash
# Eval SFT checkpoint on MATH-500, AIME-2024, AIME-2025
# Starts SGLang server, runs eval, then stops server.
#
# Usage: 
#   docker exec -d k-step-opd bash /workspace/k-step-opd/scripts/eval-sft-checkpoint.sh \
#       > /workspace/k-step-opd/eval_sft.log 2>&1

set -ex

MODEL_PATH=/workspace/k-step-opd/checkpoints/sft-qwen3-8b-base-lora-merged
EVAL_DIR=/workspace/k-step-opd/eval_results_sft
N_SAMPLES=1
MAX_TOKENS=8192
TEMPERATURE=0.6
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
    &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

# Check server is actually up
if ! curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
    echo "ERROR: Server failed to start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Prepare AIME-2025 data if not already in jsonl format
if [ ! -f /workspace/data/aime-2025/aime-2025.jsonl ]; then
    echo "=== Preparing AIME-2025 data ==="
    python3 -c "
import json
from datasets import load_dataset
ds = load_dataset('/workspace/data/aime-2025', split='train')
with open('/workspace/data/aime-2025/aime-2025.jsonl', 'w') as f:
    for row in ds:
        record = {
            'prompt': row['problem'],
            'label': str(row['answer']),
            'reward_model': 'math'
        }
        f.write(json.dumps(record) + '\n')
print(f'Prepared {len(ds)} AIME-2025 problems')
"
fi

SERVER_URL="http://127.0.0.1:$PORT"

# Eval AIME-2024 (first - small, quick sanity check)
echo "=== Evaluating AIME-2024 ==="
python3 /workspace/k-step-opd/eval_math.py \
    --server-url $SERVER_URL \
    --data-path /workspace/data/aime-2024/aime-2024.jsonl \
    --output-path $EVAL_DIR/aime2024_sft.json \
    --n-samples $N_SAMPLES \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --dataset-name aime2024 \
    --model-name "Qwen3-8B-Base-SFT-LoRA-100K" \
    --max-workers 1

# Eval AIME-2025
echo "=== Evaluating AIME-2025 ==="
python3 /workspace/k-step-opd/eval_math.py \
    --server-url $SERVER_URL \
    --data-path /workspace/data/aime-2025/aime-2025.jsonl \
    --output-path $EVAL_DIR/aime2025_sft.json \
    --n-samples $N_SAMPLES \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --dataset-name aime2025 \
    --model-name "Qwen3-8B-Base-SFT-LoRA-100K" \
    --max-workers 1

# Eval MATH-500
echo "=== Evaluating MATH-500 ==="
python3 /workspace/k-step-opd/eval_math.py \
    --server-url $SERVER_URL \
    --data-path /workspace/data/math-500/math-500.jsonl \
    --output-path $EVAL_DIR/math500_sft.json \
    --n-samples $N_SAMPLES \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --dataset-name math500 \
    --model-name "Qwen3-8B-Base-SFT-LoRA-100K" \
    --max-workers 1

# Stop server
echo "=== Stopping server ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

# Print summary
echo ""
echo "=========================================="
echo "=== EVAL SUMMARY ==="
echo "=========================================="
for f in $EVAL_DIR/*_sft.json; do
    python3 -c "
import json
with open('$f') as fp:
    d = json.load(fp)
print(f\"{d['dataset_name']:10s} | pass@1={d['pass_at_1']*100:.1f}% | pass@any={d['pass_at_any']*100:.1f}% | first_boxed@1={d.get('first_boxed_pass_at_1',0)*100:.1f}% | avg_len={d['avg_response_length']:.0f}\")
"
done
echo "=========================================="
echo "=== Done ==="
