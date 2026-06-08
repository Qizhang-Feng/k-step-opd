#!/bin/bash
# Setup a p5 node with code repo, docker image, models, and data
# Usage: bash setup-node.sh <host>
# Example: bash setup-node.sh p5-2

set -ex
HOST=$1
if [ -z "$HOST" ]; then echo "Usage: bash setup-node.sh <host>"; exit 1; fi

echo "=== Setting up $HOST ==="

# 1. Create directories
ssh -o ConnectTimeout=5 $HOST "mkdir -p /opt/dlami/nvme/qzf/{k-step-opd,models,data}"

# 2. Sync slime code
rsync -avz --exclude='.git' /Volumes/workplace/k-step-opd/slime/ $HOST:/opt/dlami/nvme/qzf/k-step-opd/slime/

# 3. Sync scripts and data prep
rsync -avz run-phase1-baseline.sh run-pilot-opd.sh check-gpu.sh prepare_eval_data.py prepare_math500_eval.py $HOST:/opt/dlami/nvme/qzf/k-step-opd/

# 4. Pull docker image
ssh -o ConnectTimeout=5 $HOST "docker pull slimerl/slime:latest"

# 5. Remove old container if exists, start new one
ssh -o ConnectTimeout=5 $HOST "docker rm -f k-step-opd 2>/dev/null; docker run --gpus all --ipc=host --net=host --privileged \
  -v /opt/dlami/nvme/qzf/k-step-opd:/workspace/k-step-opd \
  -v /opt/dlami/nvme/qzf/models:/root/.cache/huggingface \
  -v /opt/dlami/nvme/qzf/data:/workspace/data \
  --name k-step-opd \
  -d slimerl/slime:latest sleep infinity"

# 6. Download models inside container
ssh -o ConnectTimeout=5 $HOST "docker exec k-step-opd bash -c '
  pip install -q huggingface_hub[cli] 2>/dev/null
  huggingface-cli download Qwen/Qwen3-4B --local-dir /root/.cache/huggingface/Qwen3-4B
  huggingface-cli download Qwen/Qwen3-8B --local-dir /root/.cache/huggingface/Qwen3-8B
  huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /workspace/data/dapo-math-17k
  huggingface-cli download --repo-type dataset HuggingFaceH4/MATH-500 --local-dir /workspace/data/math-500
  huggingface-cli download --repo-type dataset Maxwell-Jia/AIME_2024 --local-dir /workspace/data/aime-2024
'"

# 7. Prepare eval data
ssh -o ConnectTimeout=5 $HOST "docker exec k-step-opd python3 /workspace/k-step-opd/prepare_eval_data.py"
ssh -o ConnectTimeout=5 $HOST "docker exec k-step-opd python3 /workspace/k-step-opd/prepare_math500_eval.py"

# 8. Convert student model to torch_dist
ssh -o ConnectTimeout=5 $HOST "docker exec k-step-opd bash -c '
  cd /root/slime && source scripts/models/qwen3-4B.sh && \
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    \${MODEL_ARGS[@]} \
    --hf-checkpoint /root/.cache/huggingface/Qwen3-4B \
    --save /root/.cache/huggingface/Qwen3-4B_torch_dist
'"

# 9. Verify
ssh -o ConnectTimeout=5 $HOST "docker exec k-step-opd bash -c '
  echo \"=== Verify ===\"
  python -c \"import torch; print(f\\\"PyTorch {torch.__version__}, CUDA {torch.cuda.device_count()} GPUs\\\")\"
  python -c \"import sglang; print(f\\\"SGLang {sglang.__version__}\\\")\"
  python -c \"import slime; print(\\\"slime OK\\\")\"
  ls /root/.cache/huggingface/Qwen3-4B/*.safetensors | wc -l
  ls /root/.cache/huggingface/Qwen3-8B/*.safetensors | wc -l
  ls /root/.cache/huggingface/Qwen3-4B_torch_dist/ | head -3
  wc -l /workspace/data/dapo-math-17k/dapo-math-17k.jsonl
  wc -l /workspace/data/math-500/math-500.jsonl
  wc -l /workspace/data/aime-2024/aime-2024.jsonl
  echo \"=== $HOSTNAME ready ===\"
'"

echo "=== $HOST setup complete ==="
