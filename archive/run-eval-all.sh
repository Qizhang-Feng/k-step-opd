#!/bin/bash
# Eval three models on MATH-500 and AIME-2024:
#   1. Teacher: Qwen3-8B
#   2. Student init: Qwen3-4B (before training)
#   3. Trained student: Phase 1 checkpoint
#
# Uses SGLang for generation + slime's deepscaler reward for grading.
# Run inside k-step-opd container on p5-6.

set -ex

# ============ Config ============
TEACHER_HF=/root/.cache/huggingface/Qwen3-8B
STUDENT_HF=/root/.cache/huggingface/Qwen3-4B
TRAINED_CKPT=/workspace/k-step-opd/checkpoints/phase1-baseline/iter_0000199
TRAINED_HF=/workspace/k-step-opd/checkpoints/phase1-baseline/iter_0000199_hf

MATH500_DATA=/workspace/data/math-500/math-500.jsonl
AIME_DATA=/workspace/data/aime-2024/aime-2024.jsonl

EVAL_SCRIPT=/workspace/k-step-opd/eval_math.py
OUTPUT_DIR=/workspace/k-step-opd/eval_results

N_SAMPLES=8          # pass@1 from N samples (majority vote / greedy)
MAX_RESPONSE_LEN=16384
TEMPERATURE=0.6
PORT=30000

mkdir -p $OUTPUT_DIR

export PYTHONUNBUFFERED=1

# ============ Step 0: Convert trained checkpoint to HF ============
if [ ! -d "$TRAINED_HF" ]; then
    echo "=== Converting trained checkpoint to HF format ==="
    cd /root/slime
    source scripts/models/qwen3-4B.sh
    PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
        --input-dir $TRAINED_CKPT \
        --output-dir $TRAINED_HF \
        --origin-hf-dir $STUDENT_HF \
        -a -f
    echo "=== Conversion done ==="
else
    echo "=== Trained HF checkpoint already exists, skipping conversion ==="
fi

# ============ Helper: run eval for one model ============
eval_model() {
    local MODEL_PATH=$1
    local MODEL_NAME=$2
    local TP=$3

    echo ""
    echo "=========================================="
    echo "  Evaluating: $MODEL_NAME"
    echo "  Model path: $MODEL_PATH"
    echo "  TP: $TP"
    echo "=========================================="

    # Kill any existing sglang server
    pkill -9 -f "sglang.launch_server" || true
    pkill -9 -f "eval_math" || true
    sleep 5

    # Verify GPU is free
    echo "GPU memory before starting $MODEL_NAME:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

    # Start SGLang server
    python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --host 0.0.0.0 \
        --port $PORT \
        --tp $TP \
        --chunked-prefill-size 4096 \
        --mem-fraction-static 0.85 \
        > /tmp/sglang_eval_${MODEL_NAME}.log 2>&1 &
    local SERVER_PID=$!

    echo "Starting SGLang server for $MODEL_NAME..."
    local MAX_WAIT=300
    local WAITED=0
    until curl -sf http://127.0.0.1:$PORT/health_generate > /dev/null; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: SGLang server failed to start for $MODEL_NAME"
            tail -n 20 /tmp/sglang_eval_${MODEL_NAME}.log
            pkill -9 -f "sglang.launch_server" || true
            return 1
        fi
        echo "  Waiting... (${WAITED}s)"
    done
    echo "Server ready for $MODEL_NAME"
    sleep 3

    # Run eval on MATH-500
    echo "--- Evaluating $MODEL_NAME on MATH-500 ---"
    python3 $EVAL_SCRIPT \
        --server-url http://127.0.0.1:$PORT \
        --data-path $MATH500_DATA \
        --output-path $OUTPUT_DIR/${MODEL_NAME}_math500.json \
        --n-samples $N_SAMPLES \
        --max-tokens $MAX_RESPONSE_LEN \
        --temperature $TEMPERATURE \
        --dataset-name math500

    # Run eval on AIME-2024
    echo "--- Evaluating $MODEL_NAME on AIME-2024 ---"
    python3 $EVAL_SCRIPT \
        --server-url http://127.0.0.1:$PORT \
        --data-path $AIME_DATA \
        --output-path $OUTPUT_DIR/${MODEL_NAME}_aime.json \
        --n-samples $N_SAMPLES \
        --max-tokens $MAX_RESPONSE_LEN \
        --temperature $TEMPERATURE \
        --dataset-name aime

    # Shutdown server
    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "sglang.launch_server" || true
    sleep 8
    echo "=== Done evaluating $MODEL_NAME ==="
}

# ============ Run evals ============
# Teacher: Qwen3-8B (TP=2 for 8B model)
eval_model $TEACHER_HF "teacher_qwen3-8b" 2

# Student init: Qwen3-4B (TP=1)
eval_model $STUDENT_HF "student_init_qwen3-4b" 1

# Trained student: Phase 1 checkpoint (TP=1)
eval_model $TRAINED_HF "trained_student_qwen3-4b" 1

# ============ Summary ============
echo ""
echo "=========================================="
echo "  All evaluations complete!"
echo "=========================================="
echo ""
python3 -c "
import json, glob, os

results_dir = '$OUTPUT_DIR'
files = sorted(glob.glob(os.path.join(results_dir, '*.json')))

print(f\"{'Model':<35} {'Dataset':<10} {'pass@1':>8} {'maj@{$N_SAMPLES}':>10} {'Avg Len':>8} {'N':>5}\")
print('-' * 80)

for f in files:
    with open(f) as fh:
        data = json.load(fh)
    model = data.get('model_name', os.path.basename(f))
    dataset = data.get('dataset_name', '?')
    pass1 = data.get('pass_at_1', 0) * 100
    majN = data.get('majority_vote', 0) * 100
    avg_len = data.get('avg_response_length', 0)
    n = data.get('n_problems', 0)
    print(f'{model:<35} {dataset:<10} {pass1:>7.1f}% {majN:>9.1f}% {avg_len:>8.0f} {n:>5}')
"

echo ""
echo "Detailed results in: $OUTPUT_DIR/"
