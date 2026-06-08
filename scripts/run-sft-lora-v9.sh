#!/bin/bash
# LoRA v9: Qwen3-4B-Base + teacher_sft_filtered 79K
# Based on v8, only 3 changes:
#   1. lora_alpha: 256 → 32 (Tinker standard, scaling=0.25)
#   2. learning_rate: 5e-4 → 3e-4 (Tinker 4B sweep optimal)
#   3. data: sft_math_100k_v2 → teacher_sft_filtered (teacher-consistent)
#
# Single node p5-4, 8×H100, DDP
# Usage: bash run-sft-lora-v9.sh
set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/root/.cache/huggingface/tmp
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets_cache
export XDG_CACHE_HOME=/root/.cache/huggingface
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    /opt/conda/lib/python3.11/site-packages/swift/cli/sft.py \
    --model /root/.cache/huggingface/Qwen3-4B-Base \
    --dataset /workspace/data/teacher_sft_filtered.jsonl \
    --output_dir /root/.cache/huggingface/sft-qwen3-4b-lora-v9 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --lora_rank 128 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --target_modules all-linear \
    --learning_rate 3e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_length 16384 \
    --packing true \
    --packing_num_proc 8 \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --load_from_cache_file false \
    --save_steps 390 \
    --save_total_limit 4 \
    --save_only_model true \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --report_to none
