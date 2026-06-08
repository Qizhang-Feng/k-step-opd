#!/bin/bash
# =============================================================================
# Bootstrap script for Slime OPD training on Greenland
# =============================================================================
# Downloads code/data/model/checkpoint from S3, then runs slime OPD.
# =============================================================================

set -euo pipefail

# ─── Trap: upload checkpoint on exit (even on failure) ───
UPLOAD_DONE=0
upload_checkpoint() {
    [ "$UPLOAD_DONE" = "1" ] && return 0
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  [TRAP] Uploading checkpoint to S3 (exit handler)..."
    echo "════════════════════════════════════════════════════════"
    local _output_dir="${NVME:-/tmp/instance_storage}/outputs"
    local _bucket="${S3_BUCKET:-delphi-greenland-res-alpha}"
    local _region="${S3_REGION:-us-east-2}"
    local _job_id="${AWS_BATCH_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
    if [ -d "$_output_dir" ] && [ "$(ls -A $_output_dir 2>/dev/null)" ]; then
        aws s3 sync "$_output_dir/" "s3://${_bucket}/qzf/outputs/opd-${_job_id}/" \
            --region "$_region" 2>&1 || echo "  [WARN] S3 upload failed"
        echo "  Uploaded to: s3://${_bucket}/qzf/outputs/opd-${_job_id}/"
    else
        echo "  No outputs to upload"
    fi
}
trap upload_checkpoint EXIT

# ─── Configuration (from environment variables) ───
S3_BUCKET="${S3_BUCKET:-qzf-k-step-opd-us-east-2}"
S3_REGION="${S3_REGION:-us-east-2}"
NVME="/tmp/instance_storage"
WORKSPACE="${NVME}/workspace"
MODEL_DIR="${NVME}/models"
DATA_DIR="${NVME}/data"
OUTPUT_DIR="${NVME}/outputs"

# S3 paths
S3_STUDENT_KEY="${S3_STUDENT_KEY:-checkpoints/sft-qwen3-8b-base-lora-merged_torch_dist}"
S3_TEACHER_KEY="${S3_TEACHER_KEY:-models/Qwen3-8B}"
S3_DATA_KEY="${S3_DATA_KEY:-data/dapo-math-17k.jsonl}"
S3_CODE_KEY="${S3_CODE_KEY:-code/k-step-opd.tar.gz}"

# Training params
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train-opd.sh}"
NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
LR="${LR:-5e-7}"

echo "════════════════════════════════════════════════════════"
echo "  K-Step OPD: Slime OPD Bootstrap"
echo "════════════════════════════════════════════════════════"
echo "  S3 Bucket:    $S3_BUCKET"
echo "  Student:      $S3_STUDENT_KEY"
echo "  Teacher:      $S3_TEACHER_KEY"
echo "  Data:         $S3_DATA_KEY"
echo "  Script:       $TRAIN_SCRIPT"
echo "  num_rollout:  $NUM_ROLLOUT"
echo "  batch_size:   $GLOBAL_BATCH_SIZE"
echo "════════════════════════════════════════════════════════"

# ─── Step 1: Prepare directories ───
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 1: Prepare directories"
echo "════════════════════════════════════════════════════════"
mkdir -p "$WORKSPACE" "$MODEL_DIR" "$DATA_DIR" "$OUTPUT_DIR"

# Symlink HF cache
mkdir -p "${NVME}/cache/huggingface"
ln -sfn "${NVME}/cache/huggingface" /root/.cache/huggingface 2>/dev/null || true
echo "  Done"

# ─── Step 2: Download student checkpoint (torch_dist) ───
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 2: Download student checkpoint"
echo "════════════════════════════════════════════════════════"
STUDENT_PATH="${MODEL_DIR}/student_torch_dist"
if [ "${SKIP_MODEL_DOWNLOAD:-0}" = "1" ]; then
    echo "  Skipped"
else
    aws s3 sync "s3://${S3_BUCKET}/${S3_STUDENT_KEY}/" "$STUDENT_PATH/" --region "$S3_REGION"
    echo "  Student: $STUDENT_PATH"
fi

# ─── Step 3: Download teacher model (HF format for SGLang) ───
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 3: Download teacher model"
echo "════════════════════════════════════════════════════════"
TEACHER_PATH="${MODEL_DIR}/$(basename $S3_TEACHER_KEY)"
if [ "${SKIP_MODEL_DOWNLOAD:-0}" = "1" ]; then
    echo "  Skipped"
else
    aws s3 sync "s3://${S3_BUCKET}/${S3_TEACHER_KEY}/" "$TEACHER_PATH/" --region "$S3_REGION"
    echo "  Teacher: $TEACHER_PATH ($(ls $TEACHER_PATH/*.safetensors 2>/dev/null | wc -l) safetensors)"
fi

# ─── Step 4: Download data ───
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 4: Download training data"
echo "════════════════════════════════════════════════════════"
LOCAL_DATA="${DATA_DIR}/$(basename $S3_DATA_KEY)"
aws s3 cp "s3://${S3_BUCKET}/${S3_DATA_KEY}" "$LOCAL_DATA" --region "$S3_REGION"
echo "  Data: $LOCAL_DATA ($(wc -l < $LOCAL_DATA) lines)"

# ─── Step 5: Download code ───
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 5: Download code"
echo "════════════════════════════════════════════════════════"
if [ "${SKIP_CODE_DOWNLOAD:-0}" = "1" ]; then
    echo "  Skipped"
else
    aws s3 cp "s3://${S3_BUCKET}/${S3_CODE_KEY}" "/tmp/code.tar.gz" --region "$S3_REGION"
    tar -xzf /tmp/code.tar.gz -C "$WORKSPACE"
    echo "  Code: $WORKSPACE"
fi

# ─── Step 6: Verify environment ───
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 6: Verify environment"
echo "════════════════════════════════════════════════════════"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA GPUs: {torch.cuda.device_count()}')
import sglang
print(f'  SGLang: {sglang.__version__}')
import slime
print(f'  Slime: OK')
import torchao
print(f'  torchao: {torchao.__version__}')
"

# ─── Step 7: Run OPD training ───
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 7: Run Slime OPD training"
echo "════════════════════════════════════════════════════════"

# Generate config file for train-opd.sh
cat > /tmp/opd_config.env << EOF
MODEL_CONFIG=qwen3-8B
STUDENT_HF=$TEACHER_PATH
STUDENT_TORCH_DIST=$STUDENT_PATH
TEACHER_HF=$TEACHER_PATH
TRAIN_DATA=$LOCAL_DATA
SAVE_DIR=$OUTPUT_DIR/opd-checkpoint

TEACHER_TP=2
TEACHER_GPUS="6,7"
ACTOR_GPUS=4
ACTOR_TP=4
ROLLOUT_GPUS=2

LR=${LR:-5e-7}
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
N_SAMPLES=4
MAX_RESPONSE_LEN=8192
OPD_KL_COEF=${OPD_KL_COEF:-1.0}
SAVE_INTERVAL=25
MAX_TOKENS_PER_GPU=4096
EOF

# Override STUDENT_HF to point to merged HF checkpoint (for rollout)
# The torch_dist is for actor/ref, HF is for rollout SGLang engines
sed -i "s|STUDENT_HF=.*|STUDENT_HF=${NVME}/models/student_hf|" /tmp/opd_config.env

# Download merged HF checkpoint for rollout
echo "  Downloading merged HF checkpoint for rollout..."
S3_STUDENT_HF_KEY="${S3_STUDENT_HF_KEY:-qzf/checkpoints/sft-100k-merged}"
mkdir -p "${NVME}/models/student_hf"
python3 << PYEOF
import boto3, os
s3 = boto3.client('s3', region_name='$S3_REGION')
paginator = s3.get_paginator('list_objects_v2')
count = 0
for page in paginator.paginate(Bucket='$S3_BUCKET', Prefix='${S3_STUDENT_HF_KEY}/'):
    for obj in page.get('Contents', []):
        key = obj['Key']
        if '/.cache/' in key or '/.git/' in key:
            continue
        rel_path = key[len('${S3_STUDENT_HF_KEY}/'):]
        if not rel_path:
            continue
        local_path = os.path.join('${NVME}/models/student_hf', rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file('$S3_BUCKET', key, local_path)
        count += 1
print(f'  Downloaded {count} files for student HF model')
PYEOF

cd "$WORKSPACE"
bash "$TRAIN_SCRIPT" /tmp/opd_config.env

# ─── Step 8: Upload results ───
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 8: Upload results to S3"
echo "════════════════════════════════════════════════════════"
if [ "${UPLOAD_OUTPUTS:-1}" = "1" ]; then
    JOB_ID="${AWS_BATCH_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
    S3_OUTPUT="s3://${S3_BUCKET}/qzf/outputs/opd-${JOB_ID}/"
    aws s3 sync "$OUTPUT_DIR/" "$S3_OUTPUT" --region "$S3_REGION"
    echo "  Uploaded to: $S3_OUTPUT"
    UPLOAD_DONE=1
else
    echo "  Skipped (UPLOAD_OUTPUTS=0)"
    UPLOAD_DONE=1
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅ OPD training complete!"
echo "════════════════════════════════════════════════════════"
