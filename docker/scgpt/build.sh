#!/bin/bash

# Build script for scGPT Docker container

set -e

echo "Building scGPT Docker container..."

# Build the Docker image
docker build -f docker/scgpt/Dockerfile -t cellsimbench/scgpt:latest .

echo "Docker image built successfully!"
echo "Image: cellsimbench/scgpt:latest" 