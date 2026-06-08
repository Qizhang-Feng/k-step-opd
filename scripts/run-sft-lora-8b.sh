#!/bin/bash
# SFT: Qwen3-8B-Base + OpenThoughts3 math 100K LoRA
# Following TML recipe: LoRA rank 128, lr 1e-3, batch 128, linear schedule
#
# Run on p5-5 inside k-step-opd container:
#   docker exec k-step-opd bash /workspace/k-step-opd/scripts/run-sft-lora.sh
#
# Or from local:
#   ssh p5-5 "docker exec k-step-opd bash /workspace/k-step-opd/scripts/run-sft-lora.sh"

set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8

MODEL=${MODEL_PATH:-/root/.cache/huggingface/Qwen3-8B-Base}
# v2 data: token-level filtered, think tags complete, boxed within limit
DATASET=${DATASET_PATH:-/workspace/data/sft_math_100k_v2.jsonl}
OUTPUT_DIR=${OUTPUT_PATH:-/workspace/k-step-opd/checkpoints/sft-qwen3-8b-base-lora}

# Global batch = NPROC_PER_NODE * per_device_batch * grad_accum = 8 * 1 * 16 = 128
swift sft \
    --model $MODEL \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --tuner_type lora \
    --lora_rank 128 \
    --lora_alpha 256 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-3 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.05 \
    --max_length 16384 \
    --packing true \
    --load_from_cache_file true \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --gradient_checkpointing true \
    --report_to none \
    --attn_impl flash_attn

echo "=== SFT LoRA training complete ==="
echo "Output: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Merge LoRA: swift export --model $MODEL --adapters $OUTPUT_DIR/checkpoint-xxx --output_dir ${OUTPUT_DIR}_merged"
echo "  2. Eval: python eval_math.py --model ${OUTPUT_DIR}_merged ..."
echo "  3. Convert to slime: convert_hf_to_torch_dist.py"
