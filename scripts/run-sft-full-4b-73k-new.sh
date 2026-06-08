#!/bin/bash
# Full FT: Qwen3-4B-Base + teacher_extra_100k_filtered (73K, NEW shard only)
# v2 recipe (lr=8e-5, cosine, 3 epochs, ZeRO-1, packing, liger)
# Single node 8×H100, global batch = 4*8*8 = 256 (matches v2)
#
# Goal: Test if the NEW 73K shard alone trains a working 4B SFT (like v2 with 79K),
# or if it triggers degeneration. This tells us whether the new shard has a
# systematic issue (e.g. length distribution) vs. simply being "extra data".
#
# Expected: ~855 steps for 3 epochs (73K samples / 256 global batch)
#
# Usage (Greenland): set TRAIN_SCRIPT="scripts/run-sft-full-4b-73k-new.sh"
#                    set S3_DATA_KEY="qzf/data/teacher_extra_100k_filtered.jsonl"
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
    $(python3 -c "import swift; import os; print(os.path.join(os.path.dirname(swift.__file__), 'cli', 'sft.py'))") \
    --model ${MODEL_PATH:-/root/.cache/huggingface/Qwen3-4B-Base} \
    --dataset ${DATASET_PATH:-/workspace/data/teacher_extra_100k_filtered.jsonl} \
    --output_dir ${OUTPUT_PATH:-/root/.cache/huggingface/sft-qwen3-4b-full-73k-new} \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --learning_rate 8e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
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
