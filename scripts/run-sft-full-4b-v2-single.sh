#!/bin/bash
# v2 reproduction: Qwen3-4B-Base full FT, 79K filtered teacher data, 3 epochs
# Single-node 8 GPU on p5-3 (originally trained on 16 GPU dual-node)
#
# Goal: reproduce v2's 50-60% AIME result with same global batch size (256)
# but on a single node, to verify the v2 success isn't multi-node-specific.
#
# Global batch math:
#   per_device=8, accum=4, GPUs=8 -> global batch = 8*4*8 = 256 (same as v2 16-GPU)
#   ~79K samples / 256 = ~310 steps per epoch -> ~930 steps for 3ep with ~25% packing
#   But v2 saw 759 steps total so packing/length filter reduced effective samples.
#
# Save every 100 steps so we capture the same 600/700/759 sweet-spot region as v2.
#
# Usage: ssh p5-3 "docker exec -d k-step-opd-sft bash /workspace/k-step-opd/scripts/run-sft-full-4b-v2-single.sh"
set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/root/.cache/huggingface/tmp
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
    --dataset /workspace/data/teacher_sft_filtered.jsonl \
    --output_dir /root/.cache/huggingface/sft-qwen3-4b-full-v2-single \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --learning_rate 8e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_length 16384 \
    --packing true \
    --packing_num_proc 8 \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --load_from_cache_file false \
    --save_steps 100 \
    --save_total_limit 10 \
    --save_only_model true \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero1 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --report_to none
