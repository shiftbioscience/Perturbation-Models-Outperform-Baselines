#!/bin/bash
set -e

ORIGIN_DIR=$(pwd)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments
FORCE_REBUILD=false
for arg in "$@"; do
    case $arg in
        --force|-f)
            FORCE_REBUILD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force, -f    Force rebuild without cache"
            echo "  --help, -h     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Building PRESAGE Docker image..."
if [ "$FORCE_REBUILD" = true ]; then
    echo "Force rebuild enabled - building without cache"
fi

# Change to the script directory
cd "$SCRIPT_DIR"

# # Check if PRESAGE reference exists
# if [ ! -d "ref/PRESAGE" ]; then
#     echo "Error: PRESAGE reference not found at $SCRIPT_DIR/ref/PRESAGE"
#     echo "Please clone PRESAGE repository first:"
#     echo "cd $SCRIPT_DIR && git clone https://github.com/genentech/PRESAGE.git ref/PRESAGE"
#     exit 1
# fi

# Build the Docker image - use project root as context
PROJECT_ROOT="$SCRIPT_DIR/../.."
if [ "$FORCE_REBUILD" = true ]; then
    docker build --no-cache -t cellsimbench/presage:latest -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_ROOT"
else
    docker build -t cellsimbench/presage:latest -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_ROOT"
fi

cd $ORIGIN_DIR

echo "PRESAGE Docker image built successfully!"
echo "Image: cellsimbench/presage:latest"
