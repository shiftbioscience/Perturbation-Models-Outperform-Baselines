#!/bin/bash

# Define all datasets to process
DATASETS=(
    'norman19' 'nadig25hepg2' 'nadig25jurkat'
    'replogle22rpe1' 'replogle22k562' 'replogle22k562gwps' 'adamson16' 'frangieh21' 'tian21crispri' 'tian21crispra'
    'kaden25rpe1' 'kaden25fibroblast' 'sunshine23' 'wessels23'
)

# DATASETS=(
#     'nadig25jurkat'
#     'replogle22rpe1' 'replogle22k562' 'adamson16' 'frangieh21' 'tian21crispri' 'tian21crispra'
#     'kaden25rpe1' 'kaden25fibroblast' 'sunshine23' 'wessels23'
# )

# DATASETS=(
#     'replogle22k562gwps'
# )


# Maximum number of parallel jobs (adjust based on your system)
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-14}

# Set custom output directory
OUTPUT_BASE_DIR="analyses/calibration/baseline_outputs"

# Create base directory if it doesn't exist
mkdir -p "${OUTPUT_BASE_DIR}"

# Create a directory for log files
LOG_DIR="${OUTPUT_BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Function to run a single dataset
run_dataset() {
    local dataset=$1
    local log_file="${LOG_DIR}/${dataset}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${dataset}..."
    
    # Run benchmark with custom output directory
    uv run cellsimbench benchmark \
        dataset=${dataset} \
        model=baselines \
        +run_centroid_analysis=true \
        hydra.run.dir="${OUTPUT_BASE_DIR}/${dataset}/${timestamp}" \
        > "${log_file}" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Successfully completed ${dataset}"
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Failed ${dataset} (exit code: $exit_code, see ${log_file})"
        return 1
    fi
}

echo "========================================="
echo "Starting parallel baseline runs"
echo "Datasets: ${#DATASETS[@]}"
echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}"
echo "Output directory: ${OUTPUT_BASE_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Centroid analysis: ENABLED"
echo "========================================="
echo ""

# Track PIDs and dataset names
declare -A JOB_PIDS
declare -A PID_DATASETS

# Track results
SUCCESSFUL=()
FAILED=()

# Function to wait for a job slot to become available
wait_for_job_slot() {
    while [ ${#JOB_PIDS[@]} -ge $MAX_PARALLEL_JOBS ]; do
        # Check for completed jobs
        for pid in "${!JOB_PIDS[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                # Job has finished
                wait $pid
                exit_code=$?
                dataset="${PID_DATASETS[$pid]}"
                
                if [ $exit_code -eq 0 ]; then
                    SUCCESSFUL+=("$dataset")
                else
                    FAILED+=("$dataset")
                fi
                
                # Remove from tracking arrays
                unset JOB_PIDS[$pid]
                unset PID_DATASETS[$pid]
            fi
        done
        
        # Brief sleep to avoid busy waiting
        sleep 0.5
    done
}

# Start jobs
for dataset in "${DATASETS[@]}"; do
    # Wait for available job slot
    wait_for_job_slot
    
    # Start new job in background
    run_dataset "$dataset" &
    pid=$!
    
    # Track the job
    JOB_PIDS[$pid]=1
    PID_DATASETS[$pid]="$dataset"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched job for ${dataset} (PID: $pid)"
done

echo ""
echo "All jobs launched. Waiting for completion..."
echo ""

# Wait for all remaining jobs to complete
for pid in "${!JOB_PIDS[@]}"; do
    wait $pid
    exit_code=$?
    dataset="${PID_DATASETS[$pid]}"
    
    if [ $exit_code -eq 0 ]; then
        SUCCESSFUL+=("$dataset")
    else
        FAILED+=("$dataset")
    fi
done

echo ""
echo "========================================="
echo "SUMMARY"
echo "========================================="
echo "Successful runs (${#SUCCESSFUL[@]}/${#DATASETS[@]}): ${SUCCESSFUL[*]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed runs (${#FAILED[@]}/${#DATASETS[@]}): ${FAILED[*]}"
    echo ""
    echo "Check log files in ${LOG_DIR} for error details"
fi
echo "========================================="