#!/bin/bash
# Phase 2.5: convert torch_dist OPD checkpoints → HF, then n=16 eval on AIME-2024/2025.
#
# Used for R4 (mean-K=4, p5-3) and R1 (mean-K=8, p5-2).
# Run inside k-step-opd container. Uses all 8 GPUs (DP=8 eval) and 4B converter.
#
# Required env:
#   RUN_NAME    — short tag, e.g. R4-meanK4  (used in HF dir + eval output names)
#   CKPT_ROOT   — torch_dist checkpoint root containing iter_NNNNNNN dirs
#                 (e.g. /root/.cache/huggingface/opd-4b-R4-meanK4)
# Optional:
#   ITERS       — default "99 199 299"
#   ORIGIN_HF   — SFT starting-point HF dir (default sft-qwen3-4b-full-v2-ckpt700)
#   PORT        — default 30050
#   KEEP_HF     — set to 1 to keep converted HF dirs (default: removed after eval)
#
# Baseline to beat (n=16): SFT v2-700 = AIME24 48.8% / AIME25 40.6%
#                          opd-4b-B iter_299 = AIME24 55.8% / AIME25 45.0%
set -e

: "${RUN_NAME:?RUN_NAME required}"
: "${CKPT_ROOT:?CKPT_ROOT required}"

ITERS="${ITERS:-99 199 299}"
ORIGIN_HF="${ORIGIN_HF:-/root/.cache/huggingface/sft-qwen3-4b-full-v2-ckpt700}"
PORT="${PORT:-30050}"
KEEP_HF="${KEEP_HF:-0}"

HF_OUT_ROOT=/root/.cache/huggingface
EVAL_DIR=/workspace/k-step-opd/eval_results_n16
mkdir -p "$EVAL_DIR"

N_SAMPLES=16
MAX_TOKENS=30000
TEMPERATURE=0.6
TOP_P=0.95
MAX_WORKERS=4
AIME_2024=/workspace/data/aime-2024/aime-2024.jsonl
AIME_2025=/workspace/data/aime-2025/aime-2025.jsonl

cleanup_server() {
    pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
    sleep 5
}
trap cleanup_server EXIT

for ITER in $ITERS; do
    PADDED=$(printf "%07d" "$ITER")
    INPUT_DIR="$CKPT_ROOT/iter_${PADDED}"
    HF_OUT="$HF_OUT_ROOT/opd-4b-${RUN_NAME}-iter${ITER}-hf"
    MODEL_NAME="${RUN_NAME}-iter${ITER}"

    echo ""
    echo "============================================================"
    echo "=== ${RUN_NAME} ITER ${ITER}: $INPUT_DIR ==="
    echo "============================================================"

    if [ ! -d "$INPUT_DIR" ]; then
        echo "SKIP: $INPUT_DIR does not exist"
        continue
    fi

    # --- Step 1: Convert torch_dist → HF ---
    if [ -f "$HF_OUT/config.json" ] && [ -f "$HF_OUT/model.safetensors.index.json" ]; then
        echo "HF already converted at $HF_OUT, skipping conversion"
    else
        echo "--- Converting iter_${PADDED} → HF ---"
        cd /root/slime
        # shellcheck disable=SC1091
        source scripts/models/qwen3-4B.sh
        PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
            --input-dir "$INPUT_DIR" \
            --output-dir "$HF_OUT" \
            --origin-hf-dir "$ORIGIN_HF" \
            --vocab-size 151936 \
            -f
        cd - > /dev/null
    fi

    # --- Step 2: Launch SGLang server (DP=8) ---
    cleanup_server
    echo "--- Launching SGLang DP=8 for $MODEL_NAME ---"
    nohup python3 -m sglang.launch_server \
        --model-path "$HF_OUT" \
        --port $PORT \
        --dp 8 \
        --mem-fraction-static 0.85 \
        --trust-remote-code \
        > /tmp/sglang-n16-${MODEL_NAME}.log 2>&1 &

    echo "Waiting for server..."
    ok=0
    for i in $(seq 1 360); do
        if curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            echo "Server ready after ${i}s"; ok=1; break
        fi
        sleep 1
    done
    if [ $ok -ne 1 ]; then
        echo "ERROR: server failed for $MODEL_NAME"
        tail -50 /tmp/sglang-n16-${MODEL_NAME}.log
        cleanup_server
        continue
    fi

    SERVER_URL="http://127.0.0.1:$PORT"

    # --- Step 3: Eval AIME-2024 + AIME-2025 ---
    for pair in "aime2024 $AIME_2024" "aime2025 $AIME_2025"; do
        set -- $pair
        DS="$1"; DATA="$2"
        echo "--- $MODEL_NAME / $DS ---"
        python3 /workspace/k-step-opd/eval_math.py \
            --server-url "$SERVER_URL" \
            --data-path "$DATA" \
            --output-path "$EVAL_DIR/${DS}_${MODEL_NAME}.json" \
            --n-samples $N_SAMPLES \
            --max-tokens $MAX_TOKENS \
            --temperature $TEMPERATURE \
            --dataset-name "$DS" \
            --model-name "$MODEL_NAME" \
            --max-workers $MAX_WORKERS \
            || echo "$DS eval FAILED for $MODEL_NAME"
    done

    cleanup_server

    # --- Step 4: Optionally free NVMe ---
    if [ "$KEEP_HF" != "1" ]; then
        echo "Removing $HF_OUT to save space"
        rm -rf "$HF_OUT"
    fi
    echo "=== ${MODEL_NAME} done ==="
done

# --- Final summary ---
echo ""
echo "=========================================="
echo "=== ${RUN_NAME} — n=16 summary ==="
echo "=========================================="
echo "Baseline v2-700: AIME24 48.8 / AIME25 40.6 ;  opd-4b-B iter299: 55.8 / 45.0"
echo ""
for ITER in $ITERS; do
    for DS in aime2024 aime2025; do
        F="$EVAL_DIR/${DS}_${RUN_NAME}-iter${ITER}.json"
        if [ -f "$F" ]; then
            python3 -c "
import json
with open('$F') as fp: d = json.load(fp)
print(f\"${RUN_NAME}-iter${ITER} ${DS}: avg_p1={d['avg_pass_at_1']*100:5.1f}%  fb_avg={d['avg_first_boxed_pass_at_1']*100:5.1f}%  pass_any={d['pass_at_any']*100:5.1f}%  avg_len={d['avg_response_length']:.0f}\")
"
        fi
    done
done
echo "=========================================="
