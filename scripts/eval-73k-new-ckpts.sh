#!/bin/bash
# Eval 4B Full FT 73K-new (Greenland) checkpoints across training: 200/400/600/702
# Run inside k-step-opd container on p5-3.
# Usage: ssh p5-3 "docker exec k-step-opd bash /workspace/k-step-opd/scripts/eval-73k-new-ckpts.sh"
set -e

PORT=30000
MAX_TOKENS=30000
TEMPERATURE=0.6
EVAL_DIR=/workspace/k-step-opd/eval_results_73k_new_ckpts
mkdir -p "$EVAL_DIR"

STEPS="${STEPS:-200 400 600 702}"
BASE_PATH="/root/.cache/huggingface/sft-qwen3-4b-full-73k-new"

for STEP in $STEPS; do
    MODEL_PATH="${BASE_PATH}/checkpoint-${STEP}"
    MODEL_NAME="4B-FullFT-73k-new-ckpt${STEP}"

    echo ""
    echo "============================================"
    echo "=== $MODEL_NAME ==="
    echo "============================================"

    # Force-overwrite tokenizer with Base tokenizer (ms-swift writes list-format extra_special_tokens that slime container can't load)
    cp /root/.cache/huggingface/Qwen3-4B-Base/tokenizer.json "$MODEL_PATH/"
    cp /root/.cache/huggingface/Qwen3-4B-Base/tokenizer_config.json "$MODEL_PATH/"

    # Kill prior server
    pkill -9 -f sglang 2>/dev/null || true
    sleep 5

    # Start server
    nohup python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port $PORT \
        --dp 8 \
        --mem-fraction-static 0.80 \
        --max-total-tokens 32768 \
        --trust-remote-code \
        > /tmp/sglang-73k-${STEP}.log 2>&1 &

    # Wait for ready (up to 5 min)
    echo "Waiting for server..."
    for i in $(seq 1 300); do
        if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            break
        fi
        if [ $i -eq 300 ]; then
            echo "ERROR: Server failed to start"
            tail -40 /tmp/sglang-73k-${STEP}.log
            exit 1
        fi
        sleep 1
    done

    echo "--- AIME-2024 ---"
    python3 /workspace/k-step-opd/eval_math.py \
        --server-url http://127.0.0.1:$PORT \
        --data-path /workspace/data/aime-2024/aime-2024.jsonl \
        --output-path "$EVAL_DIR/aime2024_step${STEP}.json" \
        --n-samples 1 \
        --max-tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --dataset-name aime2024 \
        --model-name "$MODEL_NAME" \
        --max-workers 8 || echo "eval failed for $MODEL_NAME"

    pkill -9 -f sglang 2>/dev/null || true
    sleep 5
    echo "=== $MODEL_NAME done ==="
done

echo ""
echo "=========================================="
echo "=== SUMMARY: 4B FullFT 73K-new ckpts ==="
echo "=========================================="
for STEP in $STEPS; do
    F="$EVAL_DIR/aime2024_step${STEP}.json"
    if [ -f "$F" ]; then
        python3 -c "
import json
with open('$F') as fp: d = json.load(fp)
n_close = sum(1 for r in d['details'] if '</think>' in (r.get('response_full','') or ''))
n_box = sum(1 for r in d['details'] if '\\\\boxed{' in (r.get('response_full','') or ''))
print(f\"step ${STEP}: pass@1={d['pass_at_1']*100:.1f}%  fb_p@1={d['first_boxed_pass_at_1']*100:.1f}%  avg_len={d['avg_response_length']:.0f}  </think>={n_close}/30  boxed={n_box}/30  trunc={d['truncation_rate']*100:.0f}%\")
"
    fi
done
echo "=========================================="
