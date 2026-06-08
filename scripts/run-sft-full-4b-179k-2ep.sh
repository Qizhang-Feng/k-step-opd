#!/bin/bash
# Full FT: Qwen3-4B-Base + teacher_sft_179k_merged (179K)
# Single node p5-4, 8×H100, bs=8, accum=4, global batch=256
# 2 epochs, ~680 steps, ~5h
# Usage: docker exec k-step-opd-sft bash /workspace/k-step-opd/scripts/run-sft-full-4b-179k-2ep.sh
set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/root/.cache/huggingface/tmp
export TEMP=/root/.cache/huggingface/tmp
export TMP=/root/.cache/huggingface/tmp
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets_cache
export XDG_CACHE_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

torchrun --nproc_per_node 8 \
    /opt/conda/lib/python3.11/site-packages/swift/cli/sft.py \
    --model /root/.cache/huggingface/Qwen3-4B-Base \
    --dataset /workspace/data/teacher_sft_179k_merged.jsonl \
    --output_dir /root/.cache/huggingface/sft-qwen3-4b-full-179k-2ep \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --learning_rate 8e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_length 16384 \
    --packing true \
    --packing_num_proc 8 \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --load_from_cache_file false \
    --save_steps 200 \
    --save_total_limit 5 \
    --save_only_model true \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero1 \
    --report_to none
