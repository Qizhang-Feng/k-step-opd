#!/bin/bash
# Run on p5-3: baseline (sft v2-700) + OPD iter_099 sequentially
set -e

cd /workspace/k-step-opd

echo "=== [1/2] BASELINE: sft-qwen3-4b-full-v2-ckpt700 ==="
MODEL_PATH=/root/.cache/huggingface/sft-qwen3-4b-full-v2-ckpt700 \
MODEL_NAME=v2-ckpt700-baseline \
PORT=30040 \
bash scripts/eval-aime-n16.sh 2>&1 | tee logs/eval_n16_baseline.log

sleep 10

echo "=== [2/2] OPD iter_099 ==="
MODEL_PATH=/root/.cache/huggingface/opd-4b-v2-ckpt700-instant-iter99-hf \
MODEL_NAME=opd-iter099 \
PORT=30040 \
bash scripts/eval-aime-n16.sh 2>&1 | tee logs/eval_n16_iter099.log

echo "=== p5-3 BOTH DONE ==="
