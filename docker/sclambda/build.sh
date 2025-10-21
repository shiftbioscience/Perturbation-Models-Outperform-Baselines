#!/bin/bash

# Build script for scLambda Docker container

set -e

echo "Building scLambda Docker container..."

# Check if .env file exists
if [ -f "docker/sclambda/.env" ]; then
    echo "Found .env file - environment variables will be available for container runs"
    echo "Make sure your .env file contains: OPENAI_API_KEY=your-key-here"
else
    echo "WARNING: No .env file found. Create one with your OpenAI API key:"
    echo "  echo 'OPENAI_API_KEY=your-key-here' > docker/sclambda/.env"
fi

# Build the container (assuming we're at project root)
docker build -f docker/sclambda/Dockerfile -t cellsimbench/sclambda:latest .

echo "Build completed successfully!"
echo "Container tagged as: cellsimbench/sclambda:latest"

# Optional: Test the container
if [ "$1" == "--test" ]; then
    echo "Testing container..."
    
    if [ -f "docker/sclambda/.env" ]; then
        echo "Testing with .env file..."
        docker run --rm --env-file docker/sclambda/.env cellsimbench/sclambda:latest python -c "
import os
import torch
import openai
print('Dependencies OK')
if os.getenv('OPENAI_API_KEY'):
    print('OpenAI API key loaded successfully')
    # Test API connection
    try:
        client = openai.OpenAI()
        # Just test client creation, don't make API call
        print('OpenAI client creation successful')
    except Exception as e:
        print(f'OpenAI client error: {e}')
else:
    print('WARNING: No OpenAI API key found')
"
    else
        echo "Testing without .env file (API key not available)..."
        docker run --rm cellsimbench/sclambda:latest python -c "import torch; import openai; print('Dependencies OK (no API key)')"
    fi
    
    echo "Container test completed!"
fi

echo ""
echo "Usage examples:"
echo "  # Local development with .env file:"
echo "  docker run --rm --env-file docker/sclambda/.env -v \$(pwd)/data:/data cellsimbench/sclambda:latest train /config.json"
echo ""
echo "  # Production with explicit environment variable:"
echo "  docker run --rm -e OPENAI_API_KEY=your-key cellsimbench/sclambda:latest train /config.json" 