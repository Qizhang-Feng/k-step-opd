#!/bin/bash
# Download Greenland instant OPD checkpoint, convert to HF, eval AIME
set -ex

S3_PATH="s3://delphi-greenland-res-alpha/qzf/outputs/opd-f3c2c9f1-3921-4374-8d05-add8273c56e4#0/opd-checkpoint/iter_0000199/"
LOCAL_CKPT=/root/.cache/huggingface/opd-instant-iter199
HF_OUTPUT=/root/.cache/huggingface/opd-instant-hf
ORIGIN_HF=/root/.cache/huggingface/sft-100k-merged

echo "=== Step 1: Download checkpoint from S3 ==="
mkdir -p $LOCAL_CKPT
aws s3 sync "$S3_PATH" "$LOCAL_CKPT/" --region us-east-2
echo "Download complete: $(ls $LOCAL_CKPT | wc -l) files"

echo "=== Step 2: Convert torch_dist to HF ==="
cd /root/slime
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir $LOCAL_CKPT \
    --output-dir $HF_OUTPUT \
    --origin-hf-dir $ORIGIN_HF \
    -a -f
echo "Conversion complete"

echo "=== Step 3: Eval AIME ==="
bash /workspace/k-step-opd/scripts/eval-aime.sh $HF_OUTPUT OPD-Instant

echo "=== ALL DONE ==="
