#!/bin/bash
# Full FT: Qwen3-4B-Base + 152K filtered (mixed teacher data)
# but capped at 700 steps with cosine schedule = same budget as v2 79K × 3ep.
#
# Goal: test whether the 152K failure is "long training" or "data mix itself".
#   If 152K @ 700 steps works (>30%) → long training is the trigger
#   If 152K @ 700 steps still fails (<10%) → mix itself is toxic
#
# At step 700, model has seen ~700/595 = ~1.18 epoch of 152K data
# (which means ~25-30% of all (prompt, response) pairs once).
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
    --dataset ${DATASET_PATH:-/workspace/data/teacher_sft_179k_thinkfilter.jsonl} \
    --output_dir ${OUTPUT_PATH:-/root/.cache/huggingface/sft-qwen3-4b-full-152k-700steps} \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --learning_rate 8e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_steps 700 \
    --num_train_epochs 99 \
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
