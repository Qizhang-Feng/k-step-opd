#!/bin/bash
# Download extra models not in original setup-node.sh
# Usage: bash setup-extra-models.sh <host>
# Example: bash setup-extra-models.sh p5-5

set -ex
HOST=${1:?Usage: bash setup-extra-models.sh <host>}

echo "=== Downloading extra models on $HOST ==="
ssh -o ConnectTimeout=5 $HOST "docker exec k-step-opd bash -c '
pip install -q huggingface_hub[cli] 2>/dev/null
huggingface-cli download Qwen/Qwen3-1.7B-Base --local-dir /root/.cache/huggingface/Qwen3-1.7B-Base
huggingface-cli download Qwen/Qwen3-1.7B --local-dir /root/.cache/huggingface/Qwen3-1.7B
echo done
'"

echo "=== $HOST extra models ready ==="
