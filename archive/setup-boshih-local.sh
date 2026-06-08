#!/bin/bash
# Setup slime container for boshih — run directly on p5-2
# Usage: bash setup-boshih-local.sh

set -ex

echo "=== Setting up boshih's slime container ==="

# 1. Create directories
mkdir -p /opt/dlami/nvme/boshih/{slime-work,models,data}

# 2. Remove old container if exists, start new one
docker rm -f boshih_slime 2>/dev/null
docker run --gpus all --ipc=host --net=host --privileged \
  -v /opt/dlami/nvme/boshih/slime-work:/workspace/slime-work \
  -v /opt/dlami/nvme/boshih/models:/root/.cache/huggingface \
  -v /opt/dlami/nvme/boshih/data:/workspace/data \
  --name boshih_slime \
  -d slimerl/slime:latest sleep infinity

# 3. Verify
docker exec boshih_slime bash -c '
  echo "=== Verify ==="
  python -c "import torch; print(f\"PyTorch {torch.__version__}, CUDA {torch.cuda.device_count()} GPUs\")"
  python -c "import sglang; print(f\"SGLang {sglang.__version__}\")"
  python -c "import slime; print(\"slime OK\")"
  echo "=== boshih_slime ready ==="
'

echo "=== Done. Enter with: docker exec -it boshih_slime bash ==="
