#!/bin/bash
export SHARD_ID=1
export NUM_SHARDS=2
cd /workspace/k-step-opd
rm -f /root/.cache/huggingface/rollouts-8b-sft/rollouts_shard1.jsonl
exec bash scripts/collect-rollouts-8b-dp.sh
