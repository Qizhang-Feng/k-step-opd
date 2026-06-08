#!/bin/bash
# Full fine-tuning SFT: Qwen3-4B-Base + OpenThoughts3 100K
# DeepSpeed ZeRO-1 + Liger Kernel + gradient checkpointing
# Global batch = 8 GPUs * 1 bs * 32 accum = 256
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/root/.cache/huggingface/tmp
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets_cache
mkdir -p $TMPDIR $HF_DATASETS_CACHE

swift sft \
    --model /root/.cache/huggingface/Qwen3-4B-Base \
    --dataset /workspace/data/sft_math_100k_v2.jsonl \
    --output_dir /root/.cache/huggingface/sft-qwen3-4b-base-fullft-final \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 8e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_steps 3000 \
    --max_length 16384 \
    --packing true \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --deepspeed /workspace/k-step-opd/configs/ds_zero1.json \
    --load_from_cache_file false \
    --save_steps 500 \
    --save_total_limit 2 \
    --save_only_model true \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --report_to none

echo "=== Full FT SFT complete ==="
