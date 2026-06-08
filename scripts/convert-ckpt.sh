#!/bin/bash
# Convert torch_dist checkpoint to HF format
# Usage: bash scripts/convert-ckpt.sh <model-config> <input-dir> <output-dir> <origin-hf-dir>
# Example:
#   bash scripts/convert-ckpt.sh qwen3-1.7B \
#     /workspace/k-step-opd/checkpoints/phase1v3-baseline/iter_0000199 \
#     /workspace/k-step-opd/checkpoints/phase1v3-baseline/iter_0000199_hf \
#     /root/.cache/huggingface/Qwen3-1.7B
set -ex

MODEL_CONFIG=${1:?Usage: bash scripts/convert-ckpt.sh <model-config> <input> <output> <origin-hf>}
INPUT_DIR=${2:?}
OUTPUT_DIR=${3:?}
ORIGIN_HF=${4:?}

cd /root/slime
source scripts/models/${MODEL_CONFIG}.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir $INPUT_DIR \
    --output-dir $OUTPUT_DIR \
    --origin-hf-dir $ORIGIN_HF \
    -a -f

echo "=== Converted: $OUTPUT_DIR ==="
