#!/bin/bash
# Sync files via S3 between nodes
# Usage:
#   bash s3-sync.sh push <type> [name]    # Push to S3
#   bash s3-sync.sh pull <type> [name]    # Pull from S3
#   bash s3-sync.sh ls [type]             # List S3 contents
#
# Types:
#   model <name>       - HuggingFace models (e.g. Qwen3-1.7B-Base)
#   ckpt <name>        - Training checkpoints (e.g. phase1v2-baseline-A/iter_0000199_hf)
#   results [name]     - Eval results (e.g. eval_results_v2)
#   scripts            - All .sh and .py scripts
#
# Examples:
#   bash s3-sync.sh push model Qwen3-1.7B-Base
#   bash s3-sync.sh pull model Qwen3-1.7B-Base
#   bash s3-sync.sh push ckpt phase1v2-baseline-A/iter_0000199_hf
#   bash s3-sync.sh pull ckpt phase1v2-baseline-A/iter_0000199_hf
#   bash s3-sync.sh push results eval_results_v2
#   bash s3-sync.sh pull results eval_results_v2
#   bash s3-sync.sh push scripts
#   bash s3-sync.sh ls model

set -e

S3_BUCKET="s3://qzf-k-step-opd-us-east-2"
WORKSPACE="/opt/dlami/nvme/qzf/k-step-opd"
MODELS="/opt/dlami/nvme/qzf/models"

ACTION=${1:?Usage: bash s3-sync.sh <push|pull|ls> <type> [name]}
TYPE=${2:-""}
NAME=${3:-""}

get_paths() {
    case $TYPE in
        model)
            LOCAL="${MODELS}/${NAME}"
            S3="${S3_BUCKET}/models/${NAME}"
            ;;
        ckpt|checkpoint)
            LOCAL="${WORKSPACE}/checkpoints/${NAME}"
            S3="${S3_BUCKET}/checkpoints/${NAME}"
            ;;
        results)
            NAME=${NAME:-eval_results_v2}
            LOCAL="${WORKSPACE}/${NAME}"
            S3="${S3_BUCKET}/${NAME}"
            ;;
        scripts)
            LOCAL="${WORKSPACE}"
            S3="${S3_BUCKET}/scripts"
            ;;
        *)
            echo "Unknown type: $TYPE"
            echo "Types: model, ckpt, results, scripts"
            exit 1
            ;;
    esac
}

case $ACTION in
    push)
        get_paths
        echo "=== Push: ${LOCAL} → ${S3} ==="
        if [ "$TYPE" = "scripts" ]; then
            aws s3 sync "${LOCAL}/" "${S3}/" --exclude "*" --include "*.sh" --include "*.py"
        else
            aws s3 sync "${LOCAL}/" "${S3}/"
        fi
        echo "=== Done ==="
        ;;
    pull)
        get_paths
        echo "=== Pull: ${S3} → ${LOCAL} ==="
        mkdir -p "${LOCAL}"
        if [ "$TYPE" = "scripts" ]; then
            aws s3 sync "${S3}/" "${LOCAL}/" --exclude "*" --include "*.sh" --include "*.py"
        else
            aws s3 sync "${S3}/" "${LOCAL}/"
        fi
        echo "=== Done ==="
        ;;
    ls)
        if [ -z "$TYPE" ]; then
            echo "=== S3 top level ==="
            aws s3 ls "${S3_BUCKET}/" --human-readable
        else
            get_paths
            echo "=== S3: ${S3} ==="
            aws s3 ls "${S3}/" --human-readable
        fi
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Usage: bash s3-sync.sh <push|pull|ls> <type> [name]"
        exit 1
        ;;
esac
