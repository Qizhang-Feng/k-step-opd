#!/bin/bash
export SHARD_ID=0
export NUM_SHARDS=2
cd /workspace/k-step-opd
# Fix prompts path for p5-5
sed 's|PROMPTS=/workspace/data/dapo-math-17k.jsonl|PROMPTS=/workspace/data/dapo-math-17k/dapo-math-17k.jsonl|' scripts/collect-rollouts-8b-dp.sh > /tmp/collect-dp-fixed.sh
rm -f /root/.cache/huggingface/rollouts-8b-sft/rollouts_shard0.jsonl
exec bash /tmp/collect-dp-fixed.sh
