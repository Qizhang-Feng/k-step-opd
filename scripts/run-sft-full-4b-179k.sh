#!/bin/bash
# Full FT: Qwen3-4B-Base + teacher_sft_179k_merged (179K, unfiltered)
# Following Lightning-OPD config: full FT, ZeRO-0, liger, packing, lr=8e-5
# Single node 8×H100, global batch = 8*4*2 = 64 (vs Lightning-OPD's 256 on 32 GPU)
# To match Lightning-OPD's effective training: max_steps=3000 (same as them)
# Usage: docker exec k-step-opd-sft bash /workspace/k-step-opd/scripts/run-sft-full-4b-179k.sh
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

# Use swift CLI directly (works regardless of python path)
torchrun --nproc_per_node 8 \
    $(python3 -c "import swift; import os; print(os.path.join(os.path.dirname(swift.__file__), 'cli', 'sft.py'))") \
    --model ${MODEL_PATH:-/root/.cache/huggingface/Qwen3-4B-Base} \
    --dataset ${DATASET_PATH:-/workspace/data/teacher_sft_179k_merged.jsonl} \
    --output_dir ${OUTPUT_PATH:-/root/.cache/huggingface/sft-qwen3-4b-full-179k} \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --learning_rate 8e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_length 16384 \
    --packing true \
    --packing_num_proc 8 \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --load_from_cache_file false \
    --save_steps 500 \
    --save_total_limit 5 \
    --save_only_model true \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero1 \
    --report_to none
