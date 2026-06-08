#!/bin/bash
# OPD on 4B SFT v2 ckpt-700 (60% AIME) with per-token KL dump for offline analysis.
#
# Steps:
#   1. (one-time) Convert HF ckpt-700 → torch_dist for slime/Megatron actor
#   2. Launch instant OPD with KL dump every 10 rollouts × 8 samples/rank
#
# Run inside the `k-step-opd` container on p5-3.
#   docker exec -it k-step-opd bash /workspace/k-step-opd/_opd_run/run-opd-4b-v2-ckpt700-with-kl-dump.sh
set -ex

CONFIG=${CONFIG:-/workspace/k-step-opd/_opd_run/opd-4b-v2-ckpt700-instant.env}
source "$CONFIG"

# --- Step 1: convert HF → torch_dist if not already done ---
if [ ! -f "${STUDENT_TORCH_DIST}/latest_checkpointed_iteration.txt" ]; then
    echo "=== Converting ${STUDENT_HF} → ${STUDENT_TORCH_DIST} ==="
    cd /root/slime
    source scripts/models/${MODEL_CONFIG}.sh
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint ${STUDENT_HF} \
        --save ${STUDENT_TORCH_DIST}
    echo "=== Conversion done ==="
fi

# --- Step 2: launch OPD ---
exec bash /workspace/k-step-opd/_opd_run/train-opd.sh "$CONFIG"
