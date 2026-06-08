#!/bin/bash
# n=16 eval on AIME-2024 + AIME-2025 for a single model.
#
# Required env:
#   MODEL_PATH    — HF model dir (must exist)
#   MODEL_NAME    — short name used in output filenames
# Optional:
#   N_SAMPLES     — default 16
#   MAX_TOKENS    — default 30000 (Qwen3-4B-Base context = 32768; leave headroom for prompt)
#   TEMPERATURE   — default 0.6
#   TOP_P         — default 0.95
#   PORT          — default 30040
#   EVAL_DIR      — default /workspace/k-step-opd/eval_results_n16
#   AIME_2024     — default /workspace/data/aime-2024/aime-2024.jsonl
#   AIME_2025     — default /workspace/data/aime-2025/aime-2025.jsonl
#
# Run inside k-step-opd container.  Uses all 8 GPUs (DP=8).
set -e

: "${MODEL_PATH:?MODEL_PATH required}"
: "${MODEL_NAME:?MODEL_NAME required}"

N_SAMPLES=${N_SAMPLES:-16}
MAX_TOKENS=${MAX_TOKENS:-30000}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
PORT=${PORT:-30040}
EVAL_DIR=${EVAL_DIR:-/workspace/k-step-opd/eval_results_n16}
AIME_2024=${AIME_2024:-/workspace/data/aime-2024/aime-2024.jsonl}
AIME_2025=${AIME_2025:-/workspace/data/aime-2025/aime-2025.jsonl}
# eval_math.py splits each problem's n_samples into independent n=1 requests
# in parallel. With max_workers=4 problems × n=16 = 64 sequences in flight,
# distributed across DP=8 → ~8 per replica. KV-cache + continuous batching
# handles preemption when a replica's tokens overflow.
MAX_WORKERS=${MAX_WORKERS:-4}

mkdir -p "$EVAL_DIR"

cleanup_server() {
    pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
    sleep 5
}
trap cleanup_server EXIT

if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: $MODEL_PATH does not look like a HF model dir"
    exit 1
fi

cleanup_server
echo "=== Launching SGLang DP=8 for $MODEL_NAME ==="
nohup python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port $PORT \
    --dp 8 \
    --mem-fraction-static 0.85 \
    --trust-remote-code \
    > /tmp/sglang-n16-${MODEL_NAME}.log 2>&1 &

echo "Waiting for server..."
for i in $(seq 1 360); do
    if curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if [ $i -eq 360 ]; then
        echo "ERROR: server failed to start"
        tail -50 /tmp/sglang-n16-${MODEL_NAME}.log
        exit 1
    fi
    sleep 1
done

SERVER_URL="http://127.0.0.1:$PORT"

run_eval() {
    local data_path="$1"
    local out_name="$2"

    echo ""
    echo "--- $out_name ---"
    python3 /workspace/k-step-opd/eval_math.py \
        --server-url "$SERVER_URL" \
        --data-path "$data_path" \
        --output-path "$EVAL_DIR/${out_name}_${MODEL_NAME}.json" \
        --n-samples "$N_SAMPLES" \
        --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --dataset-name "$out_name" \
        --model-name "$MODEL_NAME" \
        --max-workers "$MAX_WORKERS" \
        || echo "$out_name eval FAILED for $MODEL_NAME"
}

run_eval "$AIME_2024" aime2024
run_eval "$AIME_2025" aime2025

cleanup_server

echo ""
echo "=========================================="
echo "=== $MODEL_NAME — n=$N_SAMPLES summary ==="
echo "=========================================="
for DS in aime2024 aime2025; do
    F="$EVAL_DIR/${DS}_${MODEL_NAME}.json"
    if [ -f "$F" ]; then
        python3 -c "
import json
with open('$F') as fp: d = json.load(fp)
print(f\"{'$DS':10s} | avg_p1={d['avg_pass_at_1']*100:5.1f}%  fb_avg={d['avg_first_boxed_pass_at_1']*100:5.1f}%  pass_any={d['pass_at_any']*100:5.1f}%  avg_len={d['avg_response_length']:.0f}\")
"
    fi
done
echo "=========================================="
