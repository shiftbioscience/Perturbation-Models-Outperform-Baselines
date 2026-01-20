#!/bin/bash
# Pull all processed datasets and model files from S3
# Usage: ./scripts/pull_all_datasets.sh

set -e

S3_BUCKET="s3://public-access-bucket-i9sdj34j"
S3_PREFIX="cellsimbench-data"

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Pulling datasets and model files from S3..."
echo "Bucket: ${S3_BUCKET}/${S3_PREFIX}/"
echo ""

# List of datasets to download
DATASETS=(
    "adamson16"
    "frangieh21"
    "kaden25fibroblast"
    "kaden25rpe1"
    "nadig25hepg2"
    "nadig25jurkat"
    "norman19"
    "replogle22k562"
    "replogle22k562gwps"
    "replogle22rpe1"
    "sunshine23"
    "tian21crispra"
    "tian21crispri"
    "wessels23"
)

# Download processed dataset files for each dataset
echo "=== Downloading processed dataset files ==="
for dataset in "${DATASETS[@]}"; do
    LOCAL_FILE="data/${dataset}/${dataset}_processed_complete.h5ad"
    echo "Downloading: ${LOCAL_FILE}"
    aws s3 cp --no-sign-request "${S3_BUCKET}/${S3_PREFIX}/${LOCAL_FILE}" "${LOCAL_FILE}"
done

# Download model files explicitly (no sync due to bucket policy)
echo ""
echo "=== Downloading model files ==="
MODEL_FILES=(
    "data/models/scgpt/args.json"
    "data/models/scgpt/vocab.json"
    "data/models/scgpt/best_model.pt"
)
for file in "${MODEL_FILES[@]}"; do
    echo "Downloading: ${file}"
    aws s3 cp --no-sign-request "${S3_BUCKET}/${S3_PREFIX}/${file}" "${file}"
done

echo ""
echo "All files downloaded successfully!"
