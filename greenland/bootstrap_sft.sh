#!/bin/bash
# =============================================================================
# Greenland Bootstrap: SFT Training
# =============================================================================
# Based on proven greenland/bootstrap.sh pattern.
# Uses boto3 for S3 (no aws cli dependency).
# =============================================================================

set -uo pipefail

# =============================================================================
# Configuration
# =============================================================================
S3_BUCKET="${S3_BUCKET:-qzf-k-step-opd-us-east-2}"
S3_REGION="${S3_REGION:-us-east-2}"
S3_CODE_KEY="${S3_CODE_KEY:-code/k-step-opd.tar.gz}"
S3_MODEL_KEY="${S3_MODEL_KEY:-models/Qwen3-8B-Base}"
S3_DATA_KEY="${S3_DATA_KEY:-data/sft_math_100k_v2.jsonl}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/run-sft-lora.sh}"
UPLOAD_OUTPUTS="${UPLOAD_OUTPUTS:-1}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
SKIP_CODE_DOWNLOAD="${SKIP_CODE_DOWNLOAD:-0}"

NVME="/tmp/instance_storage"
NVME_WORKSPACE="${NVME}/workspace"
NVME_DATA="${NVME}/data"
NVME_MODELS="${NVME}/models"
NVME_OUTPUTS="${NVME}/outputs"
NVME_CACHE_HF="${NVME}/cache/hf"
NVME_CACHE_TORCH="${NVME}/cache/torch"
NVME_CACHE_TRITON="${NVME}/cache/triton"

JOB_NAME="${AWS_BATCH_JOB_ID:-sft-$(date +%Y%m%d_%H%M%S)}"
S3_OUTPUT_PREFIX="${S3_OUTPUT_PREFIX:-outputs/${JOB_NAME}/}"

# =============================================================================
# Logging
# =============================================================================
log_info()    { echo -e "[bootstrap] $1"; }
log_success() { echo -e "[bootstrap ✅] $1"; }
log_error()   { echo -e "[bootstrap ❌] $1"; }
log_section() { echo -e "\n════════════════════════════════════════════════════════"; echo -e "  $1"; echo -e "════════════════════════════════════════════════════════\n"; }

# =============================================================================
# S3 helpers (boto3, no aws cli needed)
# =============================================================================
s3_download_file() {
    local s3_key="$1"
    local local_path="$2"
    python3 -c "
import boto3, os
os.makedirs(os.path.dirname('$local_path') or '.', exist_ok=True)
s3 = boto3.client('s3', region_name='$S3_REGION')
s3.download_file('$S3_BUCKET', '$s3_key', '$local_path')
print(f'  Downloaded: $s3_key -> $local_path')
"
}

s3_download_prefix() {
    local prefix="$1"
    local local_dir="$2"
    python3 << PYEOF
import boto3, os
s3 = boto3.client('s3', region_name='$S3_REGION')
paginator = s3.get_paginator('list_objects_v2')
count = 0
total_bytes = 0
for page in paginator.paginate(Bucket='$S3_BUCKET', Prefix='$prefix'):
    for obj in page.get('Contents', []):
        key = obj['Key']
        size = obj['Size']
        if '/.cache/' in key or '/.git/' in key:
            continue
        rel_path = key[len('$prefix'):]
        if not rel_path:
            continue
        local_path = os.path.join('$local_dir', rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file('$S3_BUCKET', key, local_path)
        count += 1
        total_bytes += size
print(f'  Downloaded {count} files ({total_bytes / (1024**3):.1f} GB)')
PYEOF
}

s3_upload_dir() {
    local local_dir="$1"
    local prefix="$2"
    python3 << PYEOF
import boto3, os
s3 = boto3.client('s3', region_name='$S3_REGION')
count = 0
for root, dirs, files in os.walk('$local_dir'):
    for fname in files:
        local_path = os.path.join(root, fname)
        rel_path = os.path.relpath(local_path, '$local_dir')
        s3_key = '$prefix' + rel_path
        s3.upload_file(local_path, '$S3_BUCKET', s3_key)
        count += 1
print(f'  Uploaded {count} files to s3://$S3_BUCKET/$prefix')
PYEOF
}

# =============================================================================
# Step 0: Show config
# =============================================================================
log_section "Greenland SFT Bootstrap"
echo "  Job:    $JOB_NAME"
echo "  Bucket: $S3_BUCKET"
echo "  Model:  $S3_MODEL_KEY"
echo "  Data:   $S3_DATA_KEY"
echo "  Script: $TRAIN_SCRIPT"

# =============================================================================
# Step 1: Prepare NVMe + /dev/shm
# =============================================================================
log_section "Step 1: Prepare NVMe"

mkdir -p "$NVME_WORKSPACE" "$NVME_DATA" "$NVME_MODELS" "$NVME_OUTPUTS"
mkdir -p "$NVME_CACHE_HF" "$NVME_CACHE_TORCH" "$NVME_CACHE_TRITON"

# Fix /dev/shm for NCCL
SHM_SIZE=$(df -h /dev/shm 2>/dev/null | tail -1 | awk '{print $2}')
log_info "Current /dev/shm: $SHM_SIZE"
mount -o remount,size=64G /dev/shm 2>/dev/null || \
    log_info "Cannot remount /dev/shm (using linuxParameters.sharedMemorySize instead)"
df -h /dev/shm | tail -1

log_success "NVMe ready"
df -h "$NVME" | tail -1

# =============================================================================
# Step 2: Symlinks
# =============================================================================
log_section "Step 2: Cache symlinks"

rm -rf /root/.cache/huggingface 2>/dev/null
mkdir -p /root/.cache
ln -sf "$NVME_CACHE_HF" /root/.cache/huggingface
ln -sf "$NVME_CACHE_TORCH" /root/.cache/torch 2>/dev/null || true
ln -sf "$NVME_CACHE_TRITON" /root/.cache/triton 2>/dev/null || true

log_success "Symlinks created"

# =============================================================================
# Step 3: Download code
# =============================================================================
log_section "Step 3: Download code"

if [[ "$SKIP_CODE_DOWNLOAD" == "1" ]]; then
    log_info "Skipped"
else
    s3_download_file "$S3_CODE_KEY" "/tmp/code.tar.gz"
    tar -xzf /tmp/code.tar.gz -C "$NVME_WORKSPACE"
    rm -f /tmp/code.tar.gz
    log_success "Code extracted to $NVME_WORKSPACE"
    ls "$NVME_WORKSPACE/"
fi

# =============================================================================
# Step 4: Download model
# =============================================================================
log_section "Step 4: Download model"

MODEL_NAME=$(basename "${S3_MODEL_KEY%/}")
MODEL_LOCAL="${NVME_MODELS}/${MODEL_NAME}"

if [[ "$SKIP_MODEL_DOWNLOAD" == "1" ]]; then
    log_info "Skipped"
else
    mkdir -p "$MODEL_LOCAL"
    s3_download_prefix "${S3_MODEL_KEY}/" "$MODEL_LOCAL/"
    log_success "Model: $MODEL_LOCAL"
    ls "$MODEL_LOCAL/" | head -10
fi

# =============================================================================
# Step 5: Download data
# =============================================================================
log_section "Step 5: Download data"

DATA_NAME=$(basename "$S3_DATA_KEY")
DATA_LOCAL="${NVME_DATA}/${DATA_NAME}"

s3_download_file "$S3_DATA_KEY" "$DATA_LOCAL"
log_success "Data: $DATA_LOCAL ($(wc -l < $DATA_LOCAL) lines)"

# =============================================================================
# Step 6: Set environment
# =============================================================================
log_section "Step 6: Environment"

export MODEL_PATH="$MODEL_LOCAL"
export DATASET_PATH="$DATA_LOCAL"
export OUTPUT_PATH="$NVME_OUTPUTS/sft-checkpoint"

export HF_HOME="$NVME_CACHE_HF"
export TRANSFORMERS_CACHE="$NVME_CACHE_HF"
export HF_DATASETS_CACHE="${NVME_CACHE_HF}/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_MODE="disabled"

log_info "MODEL_PATH=$MODEL_PATH"
log_info "DATASET_PATH=$DATASET_PATH"
log_info "OUTPUT_PATH=$OUTPUT_PATH"

# =============================================================================
# Step 7: Validate
# =============================================================================
log_section "Step 7: Validate"

FULL_TRAIN_SCRIPT="${NVME_WORKSPACE}/${TRAIN_SCRIPT}"
if [[ -f "$FULL_TRAIN_SCRIPT" ]]; then
    log_success "Train script: $FULL_TRAIN_SCRIPT"
else
    log_error "Train script not found: $FULL_TRAIN_SCRIPT"
    find "$NVME_WORKSPACE" -name "*.sh" | head -10
    exit 1
fi

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
log_success "GPUs: $GPU_COUNT"

python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.device_count()} GPUs')
import swift
print(f'  ms-swift: OK')
" || { log_error "Python imports failed"; exit 1; }

log_success "Validation passed"

# =============================================================================
# Step 8: Run training
# =============================================================================
log_section "Step 8: Run SFT training"

log_info "Executing: bash $FULL_TRAIN_SCRIPT"
TRAIN_START=$(date +%s)

cd "$NVME_WORKSPACE"
bash "$FULL_TRAIN_SCRIPT"
TRAIN_EXIT=$?

TRAIN_END=$(date +%s)
TRAIN_DURATION=$(( (TRAIN_END - TRAIN_START) / 60 ))

if [[ $TRAIN_EXIT -eq 0 ]]; then
    log_success "Training completed in ${TRAIN_DURATION} minutes"
else
    log_error "Training failed (exit $TRAIN_EXIT) after ${TRAIN_DURATION} minutes"
fi

# =============================================================================
# Step 9: Upload outputs
# =============================================================================
log_section "Step 9: Upload outputs"

if [[ "$UPLOAD_OUTPUTS" == "1" && -d "$NVME_OUTPUTS" ]]; then
    s3_upload_dir "$NVME_OUTPUTS" "$S3_OUTPUT_PREFIX"
    log_success "Uploaded to s3://${S3_BUCKET}/${S3_OUTPUT_PREFIX}"
else
    log_info "Skipped"
fi

# =============================================================================
# Done
# =============================================================================
log_section "Done"
echo "  Exit: $TRAIN_EXIT"
echo "  Duration: ${TRAIN_DURATION}m"
echo "  Output: s3://${S3_BUCKET}/${S3_OUTPUT_PREFIX}"

exit $TRAIN_EXIT
