#!/bin/bash
# Pull all cellsimbench model Docker images from DockerHub and re-tag for local use
# Usage: ./scripts/pull_all_models.sh

set -e

DOCKERHUB_USER="millerh1"

# Note: presage is excluded due to Genentech Non-Commercial Software License.
# Users must build presage locally: bash docker/presage/build.sh
MODELS=(
    "sclambda"
    "scgpt"
    "gears"
    "fmlp"
)

echo "Pulling cellsimbench model images from DockerHub (${DOCKERHUB_USER})..."

for model in "${MODELS[@]}"; do
    REMOTE_IMAGE="${DOCKERHUB_USER}/cellsimbench-${model}:latest"
    LOCAL_IMAGE="cellsimbench/${model}:latest"
    
    echo ""
    echo "=== Processing ${model} ==="
    
    # Pull from DockerHub
    echo "Pulling ${REMOTE_IMAGE}"
    docker pull "${REMOTE_IMAGE}"
    
    # Re-tag for local use with the framework
    echo "Tagging ${REMOTE_IMAGE} -> ${LOCAL_IMAGE}"
    docker tag "${REMOTE_IMAGE}" "${LOCAL_IMAGE}"
    
    echo "Done with ${model}"
done

echo ""
echo "All images pulled and tagged successfully!"
echo "Images are now available as cellsimbench/<model>:latest"
echo ""
echo "NOTE: presage must be built manually due to licensing restrictions:"
echo "  bash docker/presage/build.sh"
