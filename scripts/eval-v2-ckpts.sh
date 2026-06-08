#!/bin/bash
# Eval v2 (79K filtered × 3ep, 16-GPU) intermediate checkpoints: 600/700/759
# This is the "successful" 50% AIME run; we want to see if v2 was already good
# at step 600/700 or only annealed to 50% at the very end.
#
# Run inside k-step-opd container on p5-3 (after symlinks/copies are in place).
set -e

PORT=30000
MAX_TOKENS=30000
TEMPERATURE=0.6
EVAL_DIR=/workspace/k-step-opd/eval_results_v2_ckpts
mkdir -p "$EVAL_DIR"

# v2 used 16 GPU dual-node; on p5-3 single-node we just load the same model
# to evaluate. Tokenizer here is already correct (dict-format extra_special_tokens).
STEPS="${STEPS:-600 700 759}"
for STEP in $STEPS; do
    MODEL_PATH=/root/.cache/huggingface/sft-qwen3-4b-full-v2-ckpt${STEP}
    MODEL_NAME="4B-FullFT-v2-79k-ckpt${STEP}"

    echo ""
    echo "============================================"
    echo "=== $MODEL_NAME ==="
    echo "============================================"

    pkill -9 -f sglang 2>/dev/null || true
    sleep 5

    nohup python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port $PORT \
        --dp 8 \
        --mem-fraction-static 0.80 \
        --max-total-tokens 32768 \
        --trust-remote-code \
        > /tmp/sglang-v2-${STEP}.log 2>&1 &

    echo "Waiting for server..."
    for i in $(seq 1 300); do
        if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            break
        fi
        if [ $i -eq 300 ]; then
            echo "ERROR: Server failed to start"
            tail -40 /tmp/sglang-v2-${STEP}.log
            exit 1
        fi
        sleep 1
    done

    echo "--- AIME-2024 ---"
    python3 /workspace/k-step-opd/eval_math.py \
        --server-url http://127.0.0.1:$PORT \
        --data-path /workspace/data/aime-2024/aime-2024.jsonl \
        --output-path "$EVAL_DIR/aime2024_v2_step${STEP}.json" \
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
echo "=== SUMMARY: v2 (79K) ckpts ==="
echo "=========================================="
for STEP in $STEPS; do
    F="$EVAL_DIR/aime2024_v2_step${STEP}.json"
    if [ -f "$F" ]; then
        python3 -c "
import json
with open('$F') as fp: d = json.load(fp)
n_close = sum(1 for r in d['details'] if '</think>' in (r.get('response_full','') or ''))
n_box = sum(1 for r in d['details'] if '\\\\boxed{' in (r.get('response_full','') or ''))
print(f\"v2 step ${STEP}: pass@1={d['pass_at_1']*100:.1f}%  fb_p@1={d['first_boxed_pass_at_1']*100:.1f}%  avg_len={d['avg_response_length']:.0f}  </think>={n_close}/30  boxed={n_box}/30\")
"
    fi
done
echo "=========================================="
