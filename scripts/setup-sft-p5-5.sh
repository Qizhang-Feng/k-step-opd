#!/bin/bash
# Setup and run SFT on p5-5
# Qwen3-8B-Base + OpenThoughts3 math 100K → ms-swift full fine-tune
#
# Prerequisites:
#   - p5-5 is accessible via ssh
#   - S3 bucket has the SFT data (or we download fresh)
#
# Usage: bash scripts/setup-sft-p5-5.sh [step]
#   step: all (default), setup, data, train, eval

set -ex

HOST="p5-5"
STEP=${1:-all}

# ============================================================
# Step 1: Setup directories and SFT container
# ============================================================
setup_env() {
    echo "=== [1/4] Setting up SFT environment on $HOST ==="

    # Create directories
    ssh -o ConnectTimeout=5 $HOST "mkdir -p /opt/dlami/nvme/qzf/{k-step-opd,models,data}"

    # Sync scripts
    rsync -avz prepare_sft_data.py eval_math.py prepare_math500_eval.py \
        $HOST:/opt/dlami/nvme/qzf/k-step-opd/

    # Remove old sft container if exists, start new one
    ssh -o ConnectTimeout=5 $HOST "docker rm -f sft-env 2>/dev/null || true"
    ssh -o ConnectTimeout=5 $HOST "docker run --gpus all --ipc=host --net=host --privileged --init \
      -v /opt/dlami/nvme/qzf/k-step-opd:/workspace/k-step-opd \
      -v /opt/dlami/nvme/qzf/models:/root/.cache/huggingface \
      -v /opt/dlami/nvme/qzf/data:/workspace/data \
      --name sft-env \
      -d pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel sleep infinity"

    # Install ms-swift + flash-attn
    ssh -o ConnectTimeout=5 $HOST "docker exec sft-env bash -c '
pip install ms-swift[all] -q 2>&1 | tail -5
pip install flash-attn --no-build-isolation -q 2>&1 | tail -3
pip install huggingface_hub[cli] -q
echo \"=== ms-swift installed ===\"
swift --version
'"
    echo "=== SFT env ready ==="
}

# ============================================================
# Step 2: Download model + prepare data
# ============================================================
prepare_data() {
    echo "=== [2/4] Downloading model and preparing data ==="

    # Download Qwen3-8B-Base
    ssh -o ConnectTimeout=5 $HOST "docker exec sft-env bash -c '
echo \"--- Downloading Qwen3-8B-Base ---\"
huggingface-cli download Qwen/Qwen3-8B-Base \
    --local-dir /root/.cache/huggingface/Qwen3-8B-Base \
    --resume-download
echo \"--- Model download done ---\"
ls /root/.cache/huggingface/Qwen3-8B-Base/*.safetensors | wc -l
'"

    # Download OpenThoughts3 math data (or use S3)
    ssh -o ConnectTimeout=5 $HOST "docker exec sft-env bash -c '
if [ -f /workspace/data/sft_math_100k_messages.jsonl ]; then
    echo \"SFT data already exists\"
    wc -l /workspace/data/sft_math_100k_messages.jsonl
else
    echo \"--- Downloading OpenThoughts3 ---\"
    pip install datasets -q
    python3 -c \"
import json
from datasets import load_dataset

print(\\\"Loading OpenThoughts3...\\\")
ds = load_dataset(\\\"open-thoughts/OpenThoughts3-1.2M\\\", split=\\\"train\\\", streaming=True)

count = 0
target = 100000
with open(\\\"/workspace/data/sft_math_100k_messages.jsonl\\\", \\\"w\\\") as f:
    for row in ds:
        # Filter: math domain, has boxed answer, reasonable length
        if row.get(\\\"domain\\\", \\\"\\\") != \\\"math\\\":
            continue
        response = row.get(\\\"response\\\", \\\"\\\") or row.get(\\\"deepseek_solution\\\", \\\"\\\")
        prompt = row.get(\\\"problem\\\", \\\"\\\") or row.get(\\\"prompt\\\", \\\"\\\")
        if not response or not prompt:
            continue
        if \\\"\\\\\\\\boxed\\\" not in response:
            continue
        if len(response) > 60000:  # skip very long
            continue
        
        record = {
            \\\"messages\\\": [
                {\\\"role\\\": \\\"user\\\", \\\"content\\\": prompt},
                {\\\"role\\\": \\\"assistant\\\", \\\"content\\\": response},
            ]
        }
        f.write(json.dumps(record, ensure_ascii=False) + \\\"\\\\n\\\")
        count += 1
        if count >= target:
            break
        if count % 10000 == 0:
            print(f\\\"  {count}/{target}...\\\")

print(f\\\"Done: {count} samples saved\\\")
\"
fi
'"

    # Also download MATH-500 for eval
    ssh -o ConnectTimeout=5 $HOST "docker exec sft-env bash -c '
if [ -d /workspace/data/math-500 ]; then
    echo \"MATH-500 already exists\"
else
    huggingface-cli download --repo-type dataset HuggingFaceH4/MATH-500 \
        --local-dir /workspace/data/math-500
fi
'"

    echo "=== Data ready ==="
}

# ============================================================
# Step 3: Run SFT training
# ============================================================
run_train() {
    echo "=== [3/4] Starting SFT training ==="

    ssh -o ConnectTimeout=5 $HOST "docker exec sft-env bash -c '
cd /workspace/k-step-opd

# ms-swift SFT command
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /root/.cache/huggingface/Qwen3-8B-Base \
    --dataset /workspace/data/sft_math_100k_messages.jsonl \
    --output_dir /workspace/k-step-opd/checkpoints/sft-qwen3-8b-base \
    --deepspeed zero3 \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_length 16384 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing true \
    --report_to none \
    --attn_impl flash_attention_2
'"

    echo "=== SFT training complete ==="
}

# ============================================================
# Step 4: Eval SFT checkpoint
# ============================================================
run_eval() {
    echo "=== [4/4] Evaluating SFT checkpoint ==="

    # Find latest checkpoint
    ssh -o ConnectTimeout=5 $HOST "docker exec sft-env bash -c '
CKPT_DIR=\$(ls -td /workspace/k-step-opd/checkpoints/sft-qwen3-8b-base/checkpoint-* 2>/dev/null | head -1)
if [ -z \"\$CKPT_DIR\" ]; then
    CKPT_DIR=/workspace/k-step-opd/checkpoints/sft-qwen3-8b-base
fi
echo \"Evaluating: \$CKPT_DIR\"

pip install sglang[all] -q 2>&1 | tail -3
pip install vllm -q 2>&1 | tail -3

# Quick eval with vLLM/SGLang
python3 /workspace/k-step-opd/eval_math.py \
    --model \$CKPT_DIR \
    --data /workspace/data/math-500 \
    --n-samples 1 \
    --max-tokens 16384 \
    --temperature 0.6 \
    --output /workspace/k-step-opd/eval_results_sft/math500_sft.jsonl
'"

    echo "=== Eval complete ==="
}

# ============================================================
# Main
# ============================================================
case $STEP in
    all)
        setup_env
        prepare_data
        run_train
        run_eval
        ;;
    setup)
        setup_env
        ;;
    data)
        prepare_data
        ;;
    train)
        run_train
        ;;
    eval)
        run_eval
        ;;
    *)
        echo "Unknown step: $STEP"
        echo "Usage: bash scripts/setup-sft-p5-5.sh [all|setup|data|train|eval]"
        exit 1
        ;;
esac
