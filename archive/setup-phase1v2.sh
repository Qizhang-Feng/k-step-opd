#!/bin/bash
# Setup for Phase 1 v2: Download Qwen3-1.7B-Base and convert to torch_dist
# Run inside k-step-opd container
set -ex

echo "=== Downloading Qwen3-1.7B-Base ==="
pip install -q huggingface_hub[cli] 2>/dev/null
huggingface-cli download Qwen/Qwen3-1.7B-Base --local-dir /root/.cache/huggingface/Qwen3-1.7B-Base

echo "=== Converting to torch_dist ==="
cd /root/slime
source scripts/models/qwen3-1.7B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/.cache/huggingface/Qwen3-1.7B-Base \
    --save /root/.cache/huggingface/Qwen3-1.7B-Base_torch_dist

echo "=== Verify ==="
ls -la /root/.cache/huggingface/Qwen3-1.7B-Base/*.safetensors | wc -l
ls /root/.cache/huggingface/Qwen3-1.7B-Base_torch_dist/ | head -5
echo "=== Done ==="
