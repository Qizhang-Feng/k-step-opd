#!/bin/bash
# Qwen3-4B LoRA v5: lm_head trainable, 16K, lr=3e-5
# Key changes: modules_to_save=lm_head, lower lr
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/root/.cache/huggingface/tmp
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets_cache
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

swift sft \
    --model /root/.cache/huggingface/Qwen3-4B-Base \
    --dataset /workspace/data/sft_math_100k_v2.jsonl \
    --output_dir /root/.cache/huggingface/sft-qwen3-4b-lora-v5 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --lora_rank 128 \
    --lora_alpha 128 \
    --lora_dropout 0.0 \
    --target_modules all-linear \
    --modules_to_save lm_head \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --max_steps 800 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_length 16384 \
    --packing true \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --load_from_cache_file false \
    --save_steps 200 \
    --save_total_limit 4 \
    --save_only_model true \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --report_to none

echo "=== LoRA v5 done ==="
