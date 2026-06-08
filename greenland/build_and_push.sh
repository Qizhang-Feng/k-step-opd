#!/bin/bash
# =============================================================================
# Build and Push K-Step OPD Images for Greenland
# =============================================================================
# Builds two images: SFT (ms-swift) and Slime (OPD).
#
# Usage:
#   bash greenland/build_and_push.sh [sft|slime|both]
#
# Run on a P5 machine that has docker + AWS credentials.
# =============================================================================

set -euo pipefail

ECR_REGISTRY="654654486179.dkr.ecr.us-east-2.amazonaws.com"
ECR_REGION="us-east-2"
VERSION="${IMAGE_VERSION:-v1}"

TARGET="${1:-both}"

echo "=============================================="
echo "🐳 Build & Push K-Step OPD Images"
echo "=============================================="
echo "  Target:  $TARGET"
echo "  Version: $VERSION"
echo "  ECR:     $ECR_REGISTRY"
echo "=============================================="
echo ""

# ECR Login
echo "Step 1: ECR Login..."
aws ecr get-login-password --region "$ECR_REGION" | \
    docker login --username AWS --password-stdin "$ECR_REGISTRY"
echo ""

# Ensure ECR repos exist
for repo in k-step-opd-sft k-step-opd-slime; do
    aws ecr describe-repositories --repository-names "$repo" --region "$ECR_REGION" 2>/dev/null || \
        aws ecr create-repository --repository-name "$repo" --region "$ECR_REGION" 2>/dev/null || true
done

build_sft() {
    echo "=============================================="
    echo "  Building SFT image..."
    echo "=============================================="
    docker build \
        -t "k-step-opd-sft:greenland-${VERSION}" \
        -f greenland/Dockerfile.sft \
        .
    
    echo "  Testing offline..."
    bash greenland/test_offline.sh sft
    echo "  ✅ Offline test passed"
    
    docker tag "k-step-opd-sft:greenland-${VERSION}" \
        "${ECR_REGISTRY}/k-step-opd-sft:greenland-${VERSION}"
    
    echo "  Pushing..."
    docker push "${ECR_REGISTRY}/k-step-opd-sft:greenland-${VERSION}"
    echo "  ✅ SFT image: ${ECR_REGISTRY}/k-step-opd-sft:greenland-${VERSION}"
    echo ""
}

build_slime() {
    echo "=============================================="
    echo "  Building Slime OPD image..."
    echo "=============================================="
    docker build \
        -t "k-step-opd-slime:greenland-${VERSION}" \
        -f greenland/Dockerfile.slime \
        .
    
    echo "  Testing offline..."
    bash greenland/test_offline.sh slime
    echo "  ✅ Offline test passed"
    
    docker tag "k-step-opd-slime:greenland-${VERSION}" \
        "${ECR_REGISTRY}/k-step-opd-slime:greenland-${VERSION}"
    
    echo "  Pushing..."
    docker push "${ECR_REGISTRY}/k-step-opd-slime:greenland-${VERSION}"
    echo "  ✅ Slime image: ${ECR_REGISTRY}/k-step-opd-slime:greenland-${VERSION}"
    echo ""
}

case "$TARGET" in
    sft)
        build_sft
        ;;
    slime)
        build_slime
        ;;
    both)
        build_sft
        build_slime
        ;;
    *)
        echo "Usage: bash greenland/build_and_push.sh [sft|slime|both]"
        exit 1
        ;;
esac

echo "=============================================="
echo "✅ Done!"
echo "=============================================="
echo ""
echo "Images:"
if [[ "$TARGET" == "sft" || "$TARGET" == "both" ]]; then
    echo "  SFT:   ${ECR_REGISTRY}/k-step-opd-sft:greenland-${VERSION}"
fi
if [[ "$TARGET" == "slime" || "$TARGET" == "both" ]]; then
    echo "  Slime: ${ECR_REGISTRY}/k-step-opd-slime:greenland-${VERSION}"
fi
echo ""
echo "Next: upload data/models to S3, then submit Greenland job."
