#!/bin/bash
# LLaMAFactory LoRA v5 multi-node: p5-5 master + p5-3 worker
# Usage:
#   Master: bash run-lf-lora-v5-multinode.sh master
#   Worker: bash run-lf-lora-v5-multinode.sh worker
set -ex

ROLE=${1:?Usage: $0 <master|worker>}

MASTER_ADDR=172.31.6.60
MASTER_PORT=29501
NUM_NODES=2
NUM_GPUS=8

if [ "$ROLE" = "master" ]; then
    NODE_RANK=0
elif [ "$ROLE" = "worker" ]; then
    NODE_RANK=1
else
    echo "ERROR: role must be master or worker"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

# TCP over ethernet
export NCCL_SOCKET_IFNAME=enp71s0
export GLOO_SOCKET_IFNAME=enp71s0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
    --nnodes "$NUM_NODES" \
    --nproc_per_node "$NUM_GPUS" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    -m llamafactory.launcher \
    /workspace/k-step-opd/configs/llamafactory/qwen3_4b_lora_v5.yaml
