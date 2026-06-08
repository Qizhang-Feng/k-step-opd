#!/bin/bash
# Setup ms-swift SFT environment on a p5 node
# Usage: bash setup-sft-env.sh <host>
# Creates a separate container for SFT (doesn't pollute slime env)
set -ex

HOST=${1:?Usage: bash setup-sft-env.sh <host>}

echo "=== Setting up SFT env on $HOST ==="

# Remove old sft container if exists
ssh -o ConnectTimeout=5 $HOST "docker rm -f sft-env 2>/dev/null || true"

# Start new container with ms-swift
ssh -o ConnectTimeout=5 $HOST "docker run --gpus all --ipc=host --net=host --privileged --init \
  -v /opt/dlami/nvme/qzf/k-step-opd:/workspace/k-step-opd \
  -v /opt/dlami/nvme/qzf/models:/root/.cache/huggingface \
  -v /opt/dlami/nvme/qzf/data:/workspace/data \
  --name sft-env \
  -d pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel sleep infinity"

# Install ms-swift
ssh -o ConnectTimeout=5 $HOST "docker exec sft-env bash -c '
pip install ms-swift[all] -q 2>&1 | tail -5
pip install flash-attn --no-build-isolation -q 2>&1 | tail -3
echo done
'"

echo "=== SFT env ready on $HOST ==="
