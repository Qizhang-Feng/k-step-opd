#!/bin/bash
# Full FT 4B multi-node: p5-3 (master) + p5-4 (worker)
# Teacher-consistent data (filtered: has </think> + \boxed{}, 79K), 3 epochs
# Following Lightning-OPD config: full FT, ZeRO-1, liger, packing
# 16 GPUs, bs=8, accum=2, global batch = 256
# Usage:
#   Master (p5-4): bash run-sft-full-4b-multinode.sh master
#   Worker (p5-2): bash run-sft-full-4b-multinode.sh worker
set -ex

ROLE=${1:?Usage: $0 <master|worker>}
MASTER_ADDR=172.31.12.111
MASTER_PORT=29501
NUM_NODES=2
NUM_GPUS=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=enp71s0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/root/.cache/huggingface/tmp
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets_cache
export XDG_CACHE_HOME=/root/.cache/huggingface
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

if [ "$ROLE" = "master" ]; then
    NODE_RANK=0
elif [ "$ROLE" = "worker" ]; then
    NODE_RANK=1
else
    echo "ERROR: role must be master or worker"
    exit 1
fi

torchrun \
    --nnodes $NUM_NODES \
    --nproc_per_node $NUM_GPUS \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    /opt/conda/lib/python3.11/site-packages/swift/cli/sft.py \
    --model /root/.cache/huggingface/Qwen3-4B-Base \
    --dataset /workspace/data/teacher_sft_filtered.jsonl \
    --output_dir /root/.cache/huggingface/sft-qwen3-4b-full-teacher-v2 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --learning_rate 8e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_length 16384 \
    --packing true \
    --packing_num_proc 8 \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --load_from_cache_file false \
    --save_steps 100 \
    --save_total_limit 3 \
    --save_only_model true \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero1 \
    --report_to none
