#!/bin/bash
# Sync checkpoints between nodes via local machine
# Usage: bash sync-via-local.sh <from-host> <to-host> [subpath]
# Example: bash sync-via-local.sh p5-6 p5-2 checkpoints/phase1v2-baseline-A/iter_0000199_hf
#
# Default syncs all checkpoints. Specify subpath to sync only one.

set -e
FROM=${1:?Usage: bash sync-via-local.sh <from> <to> [subpath]}
TO=${2:?Usage: bash sync-via-local.sh <from> <to> [subpath]}
SUBPATH=${3:-checkpoints}

REMOTE_BASE=/opt/dlami/nvme/qzf/k-step-opd
LOCAL_TMP=/tmp/k-step-opd-sync

echo "=== Syncing $SUBPATH: $FROM → local → $TO ==="

mkdir -p $LOCAL_TMP

# Pull from source
echo "--- Pulling from $FROM ---"
rsync -avz --progress $FROM:$REMOTE_BASE/$SUBPATH/ $LOCAL_TMP/$SUBPATH/

# Push to target
echo "--- Pushing to $TO ---"
rsync -avz --progress $LOCAL_TMP/$SUBPATH/ $TO:$REMOTE_BASE/$SUBPATH/

echo "=== Done ==="
