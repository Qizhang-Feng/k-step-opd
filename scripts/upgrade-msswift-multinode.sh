#!/bin/bash
# Upgrade ms-swift to 4.2 on both p5-5 and p5-3
# Fixes over 4.1.3:
#   1. packing + multi-node deadlock (packing_cache support)
#   2. FSDP create_optimizer() incompatibility with transformers 5.x
#
# Prerequisites:
#   - k-step-opd-sft container running on both nodes
#   - EFS mounted at /mnt/wutianyi-efs on host (shared between p5-5 and p5-3)
#
# Usage: bash scripts/upgrade-msswift-multinode.sh
set -ex

# Nodes
MASTER=p5-5   # 172.31.6.60
WORKER=p5-3   # 172.31.12.111
CONTAINER=k-step-opd-sft

# EFS path for packing cache (shared between nodes)
EFS_HOST=/mnt/wutianyi-efs

upgrade_node() {
    local HOST=$1
    echo "=== Upgrading ms-swift on $HOST ==="

    # Setup EFS packing cache symlink (host NVMe dir is mounted in container)
    ssh -o ConnectTimeout=10 $HOST "
        # Ensure EFS is accessible
        ls $EFS_HOST/ > /dev/null 2>&1 || { echo 'ERROR: EFS not mounted at $EFS_HOST'; exit 1; }

        # Create packing cache dir on EFS
        mkdir -p $EFS_HOST/packing_cache

        # Create symlink in the NVMe workspace (mounted as /workspace/k-step-opd in container)
        ln -sfn $EFS_HOST/packing_cache /opt/dlami/nvme/qzf/k-step-opd/packing_cache
    "

    # Upgrade ms-swift inside container
    ssh -o ConnectTimeout=10 $HOST "docker exec $CONTAINER bash -c '
        set -ex
        echo \"Current ms-swift version:\"
        pip show ms-swift 2>/dev/null | grep Version || echo \"not installed\"

        # Upgrade to 4.2
        pip install \"ms-swift[all]>=4.2\" -U -q 2>&1 | tail -10

        echo \"\"
        echo \"Updated ms-swift version:\"
        pip show ms-swift | grep Version

        # Quick sanity check
        python -c \"from swift.cli.sft import main; print(\\\"ms-swift import OK\\\")\"
    '"

    echo "=== Done: $HOST ==="
}

# Upgrade both nodes
upgrade_node $MASTER
upgrade_node $WORKER

echo ""
echo "=== Both nodes upgraded to ms-swift 4.2 ==="
echo "Packing cache: /workspace/k-step-opd/packing_cache -> EFS (shared)"
echo ""
echo "Next: run scripts/run-sft-lora-v5-multinode.sh with --packing_cache /workspace/k-step-opd/packing_cache"
