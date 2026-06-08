#!/bin/bash
# Convert 3 Lightning-recipe OPD checkpoints to HF format on p5-3.
#
# Inputs:  /root/.cache/huggingface/opd-4b-lightning-recipe/iter_NNNNNNN
# Origin:  /root/.cache/huggingface/sft-qwen3-4b-full-v2-ckpt700  (ckpt SFT 起点)
# Output:  /root/.cache/huggingface/opd-4b-lightning-iterNN-hf
set -e

ITERS="${ITERS:-99 299 599}"
CKPT_ROOT=/root/.cache/huggingface/opd-4b-lightning-recipe
ORIGIN_HF=/root/.cache/huggingface/sft-qwen3-4b-full-v2-ckpt700

cd /root/slime
# shellcheck disable=SC1091
source scripts/models/qwen3-4B.sh

for ITER in $ITERS; do
    PADDED=$(printf "%07d" "$ITER")
    INPUT="$CKPT_ROOT/iter_${PADDED}"
    HF_OUT="/root/.cache/huggingface/opd-4b-lightning-iter${ITER}-hf"

    if [ -f "$HF_OUT/config.json" ] && [ -f "$HF_OUT/model.safetensors.index.json" ]; then
        echo "iter_${ITER} already converted at $HF_OUT, skipping"
        continue
    fi

    if [ ! -d "$INPUT" ]; then
        echo "SKIP iter_${ITER}: $INPUT not found"
        continue
    fi

    echo "============================================================"
    echo "Converting iter_${ITER}"
    echo "============================================================"
    PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
        --input-dir "$INPUT" \
        --output-dir "$HF_OUT" \
        --origin-hf-dir "$ORIGIN_HF" \
        --vocab-size 151936 \
        -f
    echo "iter_${ITER} converted → $HF_OUT"
done

echo
echo "=== ALL CONVERSIONS DONE ==="
ls -lah /root/.cache/huggingface/opd-4b-lightning-iter*-hf 2>/dev/null | head -10
