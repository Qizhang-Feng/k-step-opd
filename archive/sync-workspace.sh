#!/bin/bash
# Sync workspace (checkpoints + results + scripts) between nodes
# Usage: bash sync-workspace.sh <from-host> <to-host>
# Example: bash sync-workspace.sh p5-6 p5-2
#
# Syncs: checkpoints, eval_results, scripts (not models/data - those are per-node)

set -e
FROM=${1:?Usage: bash sync-workspace.sh <from> <to>}
TO=${2:?Usage: bash sync-workspace.sh <from> <to>}

WORKSPACE=/opt/dlami/nvme/qzf/k-step-opd

echo "=== Syncing workspace: $FROM → $TO ==="

# Get IPs (needed because nodes can't resolve each other's hostnames)
TO_IP=$(ssh -o ConnectTimeout=5 $TO "hostname -I | awk '{print \$1}'")
echo "Target IP: $TO_IP"

# Sync checkpoints (largest, most important)
echo "--- Syncing checkpoints ---"
ssh -o ConnectTimeout=5 $FROM "rsync -avz --progress \
  -e 'ssh -o StrictHostKeyChecking=no' \
  $WORKSPACE/checkpoints/ \
  ubuntu@${TO_IP}:$WORKSPACE/checkpoints/"

# Sync eval results
echo "--- Syncing eval results ---"
ssh -o ConnectTimeout=5 $FROM "rsync -avz \
  -e 'ssh -o StrictHostKeyChecking=no' \
  $WORKSPACE/eval_results*/ \
  ubuntu@${TO_IP}:$WORKSPACE/"

# Sync scripts
echo "--- Syncing scripts ---"
ssh -o ConnectTimeout=5 $FROM "rsync -avz \
  -e 'ssh -o StrictHostKeyChecking=no' \
  $WORKSPACE/*.sh $WORKSPACE/*.py \
  ubuntu@${TO_IP}:$WORKSPACE/"

echo "=== Done: $FROM → $TO ==="
