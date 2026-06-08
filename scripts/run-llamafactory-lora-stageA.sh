#!/bin/bash
# LlamaFactory LoRA Stage A: 4B-Base, template=default, lm_head, 8K cutoff
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/k-step-opd

llamafactory-cli train configs/llamafactory/qwen3-4b-lora-stageA.yaml \
    dataset_dir=/workspace/k-step-opd/configs/llamafactory

echo "=== Stage A complete ==="
