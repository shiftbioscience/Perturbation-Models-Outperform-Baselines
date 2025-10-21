#!/bin/bash

# Build fMLP Docker image

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building fMLP Docker image...${NC}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Go to the project root (two directories up from docker/fmlp/)
PROJECT_ROOT="$SCRIPT_DIR/../.."
cd "$PROJECT_ROOT"

# Build the Docker image
docker build \
    -f docker/fmlp/Dockerfile \
    -t cellsimbench/fmlp:latest \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully built fMLP Docker image${NC}"
    echo -e "${GREEN}Image tag: cellsimbench/fmlp:latest${NC}"
    
    # Show image info
    echo -e "\n${YELLOW}Image information:${NC}"
    docker images cellsimbench/fmlp:latest
    
    echo -e "\n${YELLOW}To run the container:${NC}"
    echo "Training:"
    echo "  docker run --gpus all -v /data:/data -v /output:/model_output cellsimbench/fmlp:latest train /config.json"
    echo ""
    echo "Prediction:"
    echo "  docker run --gpus all -v /data:/data -v /model:/model_path -v /output:/output cellsimbench/fmlp:latest predict /config.json"
else
    echo -e "${RED}✗ Failed to build fMLP Docker image${NC}"
    exit 1
fi
