#!/bin/bash
# LlamaFactory LoRA + lm_head: 2-node (p5-5 master, p5-3 worker)
# Usage:
#   Master (p5-5): bash run-llamafactory-lora-multinode.sh master
#   Worker (p5-3): bash run-llamafactory-lora-multinode.sh worker
set -e

ROLE=${1:?Usage: $0 <master|worker>}
MASTER_ADDR=172.31.6.60
MASTER_PORT=29500
NUM_NODES=2
NUM_GPUS=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=enp71s0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

if [ "$ROLE" = "master" ]; then
    NODE_RANK=0
elif [ "$ROLE" = "worker" ]; then
    NODE_RANK=1
else
    echo "ERROR: role must be master or worker"
    exit 1
fi

cd /workspace/k-step-opd

torchrun \
    --nnodes $NUM_NODES \
    --nproc_per_node $NUM_GPUS \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    -m llamafactory.launcher \
    configs/llamafactory/qwen3-4b-lora-stageA.yaml \
    dataset_dir=/workspace/k-step-opd/configs/llamafactory

echo "=== Node $NODE_RANK done ==="
