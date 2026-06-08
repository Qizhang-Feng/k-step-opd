#!/bin/bash
# Eval 4B instant OPD checkpoints (iter_099, iter_199, iter_299) on AIME-2024.
#
# Pipeline per checkpoint:
#   1. Convert torch_dist → HF (origin = SFT v2 ckpt-700 = the OPD starting point)
#   2. Launch SGLang server (DP=8, max_total_tokens=32768)
#   3. Run AIME-2024 eval (max_tokens=30000, temp=0.6, n=1, max_workers=8)
#   4. Run AIME-2025 eval (same settings)
#   5. Stop server, clean HF dir to save NVMe
#
# Run on p5-3 inside k-step-opd container:
#   docker exec -d k-step-opd bash /workspace/k-step-opd/scripts/eval-opd-4b-instant-ckpts.sh \
#     > /workspace/k-step-opd/eval_opd_4b.log 2>&1
#
# Baseline to beat: SFT v2 ckpt-700 = 60% AIME-2024, 47% AIME-2025
set -e

# Iterations to evaluate (override with: ITERS="99 199 299" bash ...)
ITERS="${ITERS:-99 199 299}"

# Paths (inside container: /opt/dlami/nvme/qzf/models is mounted at /root/.cache/huggingface)
CKPT_ROOT=/root/.cache/huggingface/opd-4b-v2-ckpt700-instant
ORIGIN_HF=/root/.cache/huggingface/sft-qwen3-4b-full-v2-ckpt700
HF_OUT_ROOT=/root/.cache/huggingface
EVAL_DIR=/workspace/k-step-opd/eval_results_opd_4b_instant
mkdir -p "$EVAL_DIR"

# SGLang serve / eval config
PORT=30020
MAX_TOKENS=30000
TEMPERATURE=0.6
N_SAMPLES=1
MAX_WORKERS=8

cleanup_server() {
    pkill -9 -f sglang 2>/dev/null || true
    sleep 5
}
trap cleanup_server EXIT

for ITER in $ITERS; do
    PADDED=$(printf "%07d" "$ITER")
    INPUT_DIR=$CKPT_ROOT/iter_${PADDED}
    HF_OUT=$HF_OUT_ROOT/opd-4b-v2-ckpt700-instant-iter${ITER}-hf
    MODEL_NAME="OPD-4B-Instant-iter${ITER}"

    echo ""
    echo "============================================================"
    echo "=== ITER ${ITER}: $INPUT_DIR ==="
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
        cd -
    fi

    # --- Step 2: Launch SGLang server (DP=8 for parallel decoding) ---
    cleanup_server
    echo "--- Launching SGLang server (DP=8) ---"
    nohup python3 -m sglang.launch_server \
        --model-path "$HF_OUT" \
        --port $PORT \
        --dp 8 \
        --mem-fraction-static 0.80 \
        --max-total-tokens 32768 \
        --trust-remote-code \
        > /tmp/sglang-opd-iter${ITER}.log 2>&1 &

    echo "Waiting for server..."
    for i in $(seq 1 300); do
        if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            break
        fi
        if [ $i -eq 300 ]; then
            echo "ERROR: Server failed to start"
            tail -40 /tmp/sglang-opd-iter${ITER}.log
            cleanup_server
            continue 2
        fi
        sleep 1
    done

    SERVER_URL="http://127.0.0.1:$PORT"

    # --- Step 3: Eval AIME-2024 ---
    echo "--- AIME-2024 ---"
    python3 /workspace/k-step-opd/eval_math.py \
        --server-url $SERVER_URL \
        --data-path /workspace/data/aime-2024/aime-2024.jsonl \
        --output-path "$EVAL_DIR/aime2024_iter${ITER}.json" \
        --n-samples $N_SAMPLES \
        --max-tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --dataset-name aime2024 \
        --model-name "$MODEL_NAME" \
        --max-workers $MAX_WORKERS || echo "AIME-2024 eval failed for iter${ITER}"

    # --- Step 4: Eval AIME-2025 ---
    echo "--- AIME-2025 ---"
    python3 /workspace/k-step-opd/eval_math.py \
        --server-url $SERVER_URL \
        --data-path /workspace/data/aime-2025/aime-2025.jsonl \
        --output-path "$EVAL_DIR/aime2025_iter${ITER}.json" \
        --n-samples $N_SAMPLES \
        --max-tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --dataset-name aime2025 \
        --model-name "$MODEL_NAME" \
        --max-workers $MAX_WORKERS || echo "AIME-2025 eval failed for iter${ITER}"

    cleanup_server
    echo "=== iter_${ITER} done ==="
done

# --- Final summary ---
echo ""
echo "=========================================="
echo "=== OPD 4B Instant — Summary ==="
echo "=========================================="
echo "Baseline: SFT v2 ckpt-700 = 60% AIME-2024, 47% AIME-2025"
echo ""
for ITER in $ITERS; do
    for DS in aime2024 aime2025; do
        F="$EVAL_DIR/${DS}_iter${ITER}.json"
        if [ -f "$F" ]; then
            python3 -c "
import json
with open('$F') as fp: d = json.load(fp)
n_close = sum(1 for r in d['details'] if '</think>' in (r.get('response_full','') or ''))
n_box = sum(1 for r in d['details'] if '\\\\boxed{' in (r.get('response_full','') or ''))
total = len(d['details'])
print(f\"iter${ITER} ${DS}: pass@1={d['pass_at_1']*100:.1f}%  fb_p@1={d.get('first_boxed_pass_at_1',0)*100:.1f}%  avg_len={d['avg_response_length']:.0f}  </think>={n_close}/{total}  boxed={n_box}/{total}\")
"
        fi
    done
done
echo "=========================================="
