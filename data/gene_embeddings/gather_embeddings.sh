#!/bin/bash
set -e  # Exit on error

# ============================================================================
# Gene Embeddings Generation Script
# ============================================================================
# This script generates foundation model gene embeddings (GenePT, Geneformer, ESM2)
# for a given input dataset.
#
# Usage: bash data/gene_embeddings/gather_embeddings.sh INPUT_DIR OUTPUT_DIR [REFERENCE_ADATA]
#
# Arguments:
#   INPUT_DIR        - Path to input .h5ad file
#   OUTPUT_DIR       - Path to output .h5ad file with embeddings
#   REFERENCE_ADATA  - (Optional) Reference .h5ad file to transfer embeddings from
# ============================================================================

# Get the directory where this script is located (allows running from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

# ============================================================================
# Argument Parsing and Validation
# ============================================================================

# Parse optional flags
FORCE_REGENERATE=false
for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
        FORCE_REGENERATE=true
    fi
done

# Remove --force from arguments for positional parsing
ARGS=()
for arg in "$@"; do
    if [ "$arg" != "--force" ]; then
        ARGS+=("$arg")
    fi
done

if [ ${#ARGS[@]} -lt 2 ]; then
    log_error "Insufficient arguments provided"
    echo "Usage: bash data/gene_embeddings/gather_embeddings.sh INPUT_DIR OUTPUT_DIR [REFERENCE_ADATA] [--force]"
    echo ""
    echo "Arguments:"
    echo "  INPUT_DIR        - Path to input .h5ad file"
    echo "  OUTPUT_DIR       - Path to output .h5ad file with embeddings"
    echo "  REFERENCE_ADATA  - (Optional) Reference .h5ad file to transfer embeddings from"
    echo ""
    echo "Options:"
    echo "  --force          - Force regeneration, ignoring cached intermediate results"
    exit 1
fi

INPUT_DIR="${ARGS[0]}"
OUTPUT_DIR="${ARGS[1]}"
REFERENCE_ADATA="${ARGS[2]:-}"

log "============================================================================"
log "Starting Gene Embeddings Generation"
log "============================================================================"
log "Script directory: $SCRIPT_DIR"
log "Input file: $INPUT_DIR"
log "Output file: $OUTPUT_DIR"
if [ -n "$REFERENCE_ADATA" ]; then
    log "Reference file: $REFERENCE_ADATA"
fi
if [ "$FORCE_REGENERATE" = true ]; then
    log "Force mode: ENABLED (ignoring cached results)"
else
    log "Force mode: DISABLED (using cached results when available)"
fi
log ""

# Validate input file exists
if [ ! -f "$INPUT_DIR" ]; then
    log_error "Input file does not exist: $INPUT_DIR"
    exit 1
fi
log "✓ Input file exists"

# Create output directory if it doesn't exist
OUTPUT_DIR_PATH="$(dirname "$OUTPUT_DIR")"
if [ ! -d "$OUTPUT_DIR_PATH" ]; then
    log "Creating output directory: $OUTPUT_DIR_PATH"
    mkdir -p "$OUTPUT_DIR_PATH"
fi

# Define stage-specific intermediate files for caching
INPUT_BASE="$(dirname "$INPUT_DIR")/$(basename "$INPUT_DIR" .h5ad)"
GENEPT_CACHE="${INPUT_BASE}.genept.h5ad"
SCGPT_CACHE="${INPUT_BASE}.scgpt.h5ad"
GENEFORMER_CACHE="${INPUT_BASE}.geneformer.h5ad"

# Define temporary file paths for atomic writes
GENEPT_TMP="${INPUT_BASE}.genept.tmp.h5ad"
SCGPT_TMP="${INPUT_BASE}.scgpt.tmp.h5ad"
GENEFORMER_TMP="${INPUT_BASE}.geneformer.tmp.h5ad"
OUTPUT_TMP="${OUTPUT_DIR%.h5ad}.tmp.h5ad"

log "Stage-specific cache files:"
log "  GenePT cache: $GENEPT_CACHE"
log "  scGPT cache: $SCGPT_CACHE"
log "  Geneformer cache: $GENEFORMER_CACHE"
log "  Final output: $OUTPUT_DIR"
log ""

# ============================================================================
# Reference Transfer Mode (if REFERENCE_ADATA provided)
# ============================================================================

if [ -n "$REFERENCE_ADATA" ]; then
    log "Running in reference transfer mode"
    
    if [ ! -f "$REFERENCE_ADATA" ]; then
        log_error "Reference file does not exist: $REFERENCE_ADATA"
        exit 1
    fi
    log "✓ Reference file exists"
    
    log "Transferring embeddings from reference dataset..."
    uv run python "$SCRIPT_DIR/transfer_reference_gene_embeddings.py" \
        --input "$INPUT_DIR" \
        --reference_adata "$REFERENCE_ADATA" \
        --output "$OUTPUT_DIR"
    
    log "✓ Embeddings transferred successfully"
    log "Output saved to: $OUTPUT_DIR"
    exit 0
fi

# ============================================================================
# Environment Validation
# ============================================================================

log "Validating conda environments..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    log_error "conda not found in PATH. Please install conda first."
    exit 1
fi
log "✓ conda is available"

# Source conda to enable conda commands
eval "$(conda shell.bash hook)"

# Required conda environments
REQUIRED_ENVS=("src-scgpt" "src-geneformer" "src-esm2")

for ENV_NAME in "${REQUIRED_ENVS[@]}"; do
    if ! conda env list | grep -q "^${ENV_NAME}\s"; then
        log_error "Conda environment '$ENV_NAME' not found"
        log_error "Please create it using: conda env create -f $SCRIPT_DIR/envs/${ENV_NAME}.yaml"
        exit 1
    fi
    log "✓ Environment '$ENV_NAME' exists"
done
log ""

# ============================================================================
# Model Directories Configuration
# ============================================================================

log "Configuring model directories..."

# Fixed models directories (relative to script directory)
SCGPT_MODEL_DIR="$SCRIPT_DIR/scgpt"
SCGPT_SPATIAL_MODEL_DIR="$SCRIPT_DIR/scgpt_spatial"
GENEFORMER_MODEL_DIR="$SCRIPT_DIR/Geneformer/Geneformer-V2-104M"
GENEFORMER_TOKENIZER_DIR="$SCRIPT_DIR/Geneformer/geneformer/token_dictionary_gc104M.pkl"
GENEFORMER_GENE_NAME_DICT_DIR="$SCRIPT_DIR/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
PRESAGE_EMBEDDINGS_DIR="$SCRIPT_DIR/PRESAGE/cache"
ESM2_CACHE_DIR="$SCRIPT_DIR/esm2/cache"

# ESM2 model configuration
# Available models (each has different embedding dimensions):
#   - facebook/esm2_t30_150M_UR50D (640 dims, fastest, default)
#   - facebook/esm2_t36_3B_UR50D   (2560 dims, balanced)
#   - facebook/esm2_t48_15B_UR50D  (5120 dims, most accurate but slowest)
ESM2_MODEL="facebook/esm2_t48_15B_UR50D"
log "ESM2 model: $ESM2_MODEL"

# Download Geneformer model and install package if not present
if [ ! -f "${GENEFORMER_MODEL_DIR}/model.safetensors" ]; then
    log "Geneformer model not found, cloning from HuggingFace..."
    cd "$SCRIPT_DIR"
    git clone https://huggingface.co/ctheodoris/Geneformer
    cd - > /dev/null
    log "✓ Geneformer model downloaded"
    
    log "Installing geneformer package into src-geneformer environment..."
    conda run -n src-geneformer pip install -e "$SCRIPT_DIR/Geneformer"
    if [ $? -ne 0 ]; then
        log_error "Failed to install geneformer package"
        exit 1
    fi
    log "✓ Geneformer package installed"
else
    log "✓ Geneformer model already exists"
    
    # Check if geneformer package is installed
    if ! conda run -n src-geneformer python -c "import geneformer" 2>/dev/null; then
        log "Geneformer package not found in environment, installing..."
        conda run -n src-geneformer pip install -e "$SCRIPT_DIR/Geneformer"
        if [ $? -ne 0 ]; then
            log_error "Failed to install geneformer package"
            exit 1
        fi
        log "✓ Geneformer package installed"
    else
        log "✓ Geneformer package already installed"
    fi
fi
log ""

# ============================================================================
# Generate Embeddings
# ============================================================================

log "============================================================================"
log "Step 1/4: Generating GenePT embeddings"
log "============================================================================"

if [ -f "$GENEPT_CACHE" ] && [ "$FORCE_REGENERATE" = false ]; then
    log "✓ GenePT cache found, skipping generation"
    log "Using cached file: $GENEPT_CACHE"
else
    if [ -f "$GENEPT_CACHE" ] && [ "$FORCE_REGENERATE" = true ]; then
        log "GenePT cache exists but force mode enabled, regenerating..."
        rm -f "$GENEPT_CACHE"
    else
        log "Generating GenePT embeddings..."
    fi
    
    # Clean up any existing temporary file
    rm -f "$GENEPT_TMP"
    
    # Generate to temporary file
    uv run python "$SCRIPT_DIR/generate_genept_gene_embeddings.py" \
        --input "$INPUT_DIR" \
        --output "$GENEPT_TMP"
    if [ $? -ne 0 ]; then
        log_error "GenePT embedding generation failed"
        rm -f "$GENEPT_TMP"
        exit 1
    fi
    
    # Atomic move to final location
    mv "$GENEPT_TMP" "$GENEPT_CACHE"
    log "✓ GenePT embeddings generated successfully"
    log "Cached to: $GENEPT_CACHE"
fi
log ""

log "============================================================================"
log "Step 2/4: Generating scGPT embeddings"
log "============================================================================"

if [ -f "$SCGPT_CACHE" ] && [ "$FORCE_REGENERATE" = false ]; then
    log "✓ scGPT cache found, skipping generation"
    log "Using cached file: $SCGPT_CACHE"
else
    if [ -f "$SCGPT_CACHE" ] && [ "$FORCE_REGENERATE" = true ]; then
        log "scGPT cache exists but force mode enabled, regenerating..."
        rm -f "$SCGPT_CACHE"
    else
        log "Generating scGPT embeddings..."
    fi
    
    # Clean up any existing temporary file
    rm -f "$SCGPT_TMP"
    
    # Generate to temporary file
    conda run --no-capture-output -n src-scgpt python "$SCRIPT_DIR/generate_scgpt_gene_embeddings.py" \
        --input "$GENEPT_CACHE" \
        --model_dir "$SCGPT_MODEL_DIR" \
        --output "$SCGPT_TMP" \
        --gdown_path gdown
    if [ $? -ne 0 ]; then
        log_error "scGPT embedding generation failed"
        rm -f "$SCGPT_TMP"
        exit 1
    fi
    
    # Atomic move to final location
    mv "$SCGPT_TMP" "$SCGPT_CACHE"
    log "✓ scGPT embeddings generated successfully"
    log "Cached to: $SCGPT_CACHE"
fi
log ""

log "============================================================================"
log "Step 3/4: Generating Geneformer embeddings"
log "============================================================================"

if [ -f "$GENEFORMER_CACHE" ] && [ "$FORCE_REGENERATE" = false ]; then
    log "✓ Geneformer cache found, skipping generation"
    log "Using cached file: $GENEFORMER_CACHE"
else
    if [ -f "$GENEFORMER_CACHE" ] && [ "$FORCE_REGENERATE" = true ]; then
        log "Geneformer cache exists but force mode enabled, regenerating..."
        rm -f "$GENEFORMER_CACHE"
    else
        log "Generating Geneformer embeddings..."
    fi
    
    # Clean up any existing temporary file
    rm -f "$GENEFORMER_TMP"
    
    # Generate to temporary file
    conda run --no-capture-output -n src-geneformer python "$SCRIPT_DIR/generate_geneformer_gene_embeddings.py" \
        --input "$SCGPT_CACHE" \
        --output "$GENEFORMER_TMP" \
        --model_dir "$GENEFORMER_MODEL_DIR" \
        --tokenizer_dir "$GENEFORMER_TOKENIZER_DIR" \
        --gene_name_dict_dir "$GENEFORMER_GENE_NAME_DICT_DIR"
    if [ $? -ne 0 ]; then
        log_error "Geneformer embedding generation failed"
        rm -f "$GENEFORMER_TMP"
        exit 1
    fi
    
    # Atomic move to final location
    mv "$GENEFORMER_TMP" "$GENEFORMER_CACHE"
    log "✓ Geneformer embeddings generated successfully"
    log "Cached to: $GENEFORMER_CACHE"
fi
log ""

# log "============================================================================"
# log "Generating PRESAGE embeddings"
# log "============================================================================"
# conda run -n presage python "$SCRIPT_DIR/generate_presage_embeddings.py" \
#     --input "$TEMP_DIR" \
#     --output "$TEMP_DIR" \
#     --embeddings_dir "$PRESAGE_EMBEDDINGS_DIR"
# if [ $? -ne 0 ]; then
#     log_error "PRESAGE embedding generation failed"
#     rm -f "$TEMP_DIR"
#     exit 1
# fi
# log "✓ PRESAGE embeddings generated successfully"
# log ""

log "============================================================================"
log "Step 4/4: Generating ESM2 embeddings (Model: ${ESM2_MODEL})"
log "============================================================================"

if [ -f "$OUTPUT_DIR" ] && [ "$FORCE_REGENERATE" = false ]; then
    log "✓ Final output already exists, skipping ESM2 generation"
    log "Using existing file: $OUTPUT_DIR"
else
    if [ -f "$OUTPUT_DIR" ] && [ "$FORCE_REGENERATE" = true ]; then
        log "Output file exists but force mode enabled, regenerating..."
        rm -f "$OUTPUT_DIR"
    else
        log "Generating ESM2 embeddings..."
    fi
    
    # Clean up any existing temporary file
    rm -f "$OUTPUT_TMP"
    
    # Generate to temporary file
    conda run --no-capture-output -n src-esm2 python "$SCRIPT_DIR/generate_esm2_gene_embeddings.py" \
        --input "$GENEFORMER_CACHE" \
        --output "$OUTPUT_TMP" \
        --cache_dir "$ESM2_CACHE_DIR" \
        --model "$ESM2_MODEL"
    if [ $? -ne 0 ]; then
        log_error "ESM2 embedding generation failed"
        rm -f "$OUTPUT_TMP"
        exit 1
    fi
    
    # Atomic move to final location
    mv "$OUTPUT_TMP" "$OUTPUT_DIR"
    log "✓ ESM2 embeddings generated successfully"
fi
log ""

# ============================================================================
# Cleanup and Completion
# ============================================================================

# Clean up any remaining temporary files
log "Cleaning up temporary files..."
rm -f "$GENEPT_TMP" "$SCGPT_TMP" "$GENEFORMER_TMP" "$OUTPUT_TMP"
log "✓ Temporary files cleaned up"
log ""

log "============================================================================"
log "SUCCESS: Gene embeddings generated successfully!"
log "============================================================================"
log "Output file: $OUTPUT_DIR"
log "Total embeddings: GenePT, scGPT, Geneformer, ESM2"
log ""
log "Intermediate cache files (kept for future re-runs):"
log "  - $GENEPT_CACHE"
log "  - $SCGPT_CACHE"
log "  - $GENEFORMER_CACHE"
log ""
log "To force regeneration, use: bash $0 ${ARGS[0]} ${ARGS[1]} --force"
log "============================================================================"
