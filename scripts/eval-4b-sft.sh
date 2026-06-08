#!/bin/bash
# Merge Qwen3-4B-Base LoRA checkpoint and eval on AIME-2024/2025
# Run on p5-5 inside k-step-opd-sft container (has ms-swift for merge)
# then eval in k-step-opd container (has sglang + slime)
#
# Usage from local:
#   ssh p5-5 "bash /opt/dlami/nvme/qzf/k-step-opd/scripts/eval-4b-sft.sh"

set -ex

# === Config ===
BASE_MODEL=/root/.cache/huggingface/Qwen3-4B-Base
LORA_DIR=/root/.cache/huggingface/sft-qwen3-4b-base-lora
MERGED_DIR=/root/.cache/huggingface/sft-qwen3-4b-base-merged
EVAL_DIR=/workspace/k-step-opd/eval_results_4b_sft
MODEL_NAME="Qwen3-4B-Base-SFT-100K"
PORT=30010
MAX_TOKENS=30000
N_SAMPLES=1
TEMPERATURE=0.6

# === Step 1: Merge LoRA (in k-step-opd-sft container) ===
echo "=== Step 1: Merging LoRA ==="

# Find the latest checkpoint
CKPT=$(docker exec k-step-opd-sft bash -c "ls -d ${LORA_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1")
if [ -z "$CKPT" ]; then
    # Maybe the LoRA dir itself is the checkpoint (no subdirectory)
    CKPT=$LORA_DIR
fi
echo "Using checkpoint: $CKPT"

# Check if already merged
if docker exec k-step-opd-sft test -f ${MERGED_DIR}/config.json; then
    echo "Merged model already exists at $MERGED_DIR, skipping merge."
else
    docker exec k-step-opd-sft bash -c "
        swift export \
            --model $BASE_MODEL \
            --adapters $CKPT \
            --output_dir $MERGED_DIR \
            --torch_dtype bfloat16
    "
    echo "Merge complete: $MERGED_DIR"
fi

# === Step 2: Start SGLang server (in k-step-opd container) ===
echo "=== Step 2: Starting SGLang server ==="

# Kill any existing sglang processes
docker exec k-step-opd bash -c "pkill -9 -f sglang || true"
sleep 3

docker exec -d k-step-opd bash -c "
    python3 -m sglang.launch_server \
        --model-path $MERGED_DIR \
        --port $PORT \
        --tp 8 \
        --trust-remote-code \
        --mem-fraction-static 0.85 \
        --max-running-requests 2 \
        --max-total-tokens 32768 \
        > /tmp/eval_4b_server.log 2>&1
"

# Wait for server
echo "Waiting for server to start..."
for i in $(seq 1 180); do
    if docker exec k-step-opd curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: Server failed to start after 180s"
        docker exec k-step-opd cat /tmp/eval_4b_server.log | tail -30
        exit 1
    fi
    sleep 1
done

# === Step 3: Run Eval ===
echo "=== Step 3: Running Eval ==="

docker exec k-step-opd bash -c "mkdir -p $EVAL_DIR"

# AIME-2024
echo "--- AIME-2024 ---"
docker exec k-step-opd bash -c "
    python3 /workspace/k-step-opd/eval_math.py \
        --server-url http://127.0.0.1:$PORT \
        --data-path /workspace/data/aime-2024/aime-2024.jsonl \
        --output-path $EVAL_DIR/aime2024.json \
        --n-samples $N_SAMPLES \
        --max-tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --dataset-name aime2024 \
        --model-name '$MODEL_NAME' \
        --max-workers 1
"

# AIME-2025
echo "--- AIME-2025 ---"
docker exec k-step-opd bash -c "
    python3 /workspace/k-step-opd/eval_math.py \
        --server-url http://127.0.0.1:$PORT \
        --data-path /workspace/data/aime-2025/aime-2025.jsonl \
        --output-path $EVAL_DIR/aime2025.json \
        --n-samples $N_SAMPLES \
        --max-tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --dataset-name aime2025 \
        --model-name '$MODEL_NAME' \
        --max-workers 1
"

# === Step 4: Cleanup & Summary ===
echo "=== Step 4: Stopping server ==="
docker exec k-step-opd bash -c "pkill -9 -f sglang || true"

echo ""
echo "=========================================="
echo "=== EVAL RESULTS: $MODEL_NAME ==="
echo "=========================================="
docker exec k-step-opd bash -c "
for f in $EVAL_DIR/*.json; do
    python3 -c \"
import json
with open('\$f') as fp:
    d = json.load(fp)
print(f\\\"{d['dataset_name']:10s} | pass@1={d['pass_at_1']*100:.1f}% | avg_len={d['avg_response_length']:.0f}\\\")
\"
done
"
echo "=========================================="
echo "=== Done ==="
