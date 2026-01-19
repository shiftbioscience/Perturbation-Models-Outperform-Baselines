# CellSimBench Analysis Pipeline

This directory contains scripts for analyzing perturbation prediction performance using the Replogle K562 GWPS dataset.

## Overview

The analysis pipeline consists of three main components:
1. Computing MSE comparisons between baselines
2. Generating GSEA input files for transcription factor analysis
3. Creating figures and running statistical analyses

## Prerequisites

### R Environment (tested with R 4.4.1)
R dependencies are managed via `renv`. Restore the environment from the lockfile:
```bash
Rscript -e "renv::restore()"
```

### Copy main results in this directory
```bash
cp ../calibration/results/per_perturbation_results.csv .
```

### Required Input Data
The following data files must be present before running the pipeline:

```
../../data/replogle22k562gwps/
└── replogle22k562gwps_processed_complete.h5ad

./
├── ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X.txt
└── per_perturbation_results.csv
```

## Pipeline Steps

### Step 1: Compute MSE Comparisons

**Script:** `compute_replogle_mse.py`

**Purpose:** Computes Mean Squared Error for three baseline methods:
- Technical Duplicate (TD)
- Mean Baseline (MB)  
- Interpolated Duplicate (ID)

**Run:**
```bash
uv run python compute_replogle_mse.py
```

**Input:**
- `../../data/replogle22k562gwps/replogle22k562gwps_processed_complete.h5ad`

**Output:**
- `mse_comparison_data_replogle22k562gwps.csv` - Contains MSE values and DEG counts for all perturbations

### Step 2: Create GSEA Input Files

**Script:** `create_gsea_files.py`

**Purpose:** Generates ranked gene lists with differential expression scores for each transcription factor perturbation that overlaps with ChEA/ENCODE gene sets.

**Run:**
```bash
uv run python create_gsea_files.py
```

**Inputs:**
- `../../data/replogle22k562gwps/replogle22k562gwps_processed_complete.h5ad`
- `ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X.txt`

**Outputs:**
- `gsea_input_for_R/{GENE}_ranked_genes.csv` - One file per TF with ranked gene lists and scores
- `gsea_input_for_R/summary.csv` - Summary of all generated files

### Step 3: Generate Figures and Run Statistical Analysis

**Script:** `cellsimbench_figs_R.rmd`

**Purpose:** Creates all main and supplementary figures, performs statistical tests, and runs GSEA self-enrichment analysis.

**Run:**
```bash
# Restore R environment (first time only)
Rscript -e "renv::restore()"

# Render the R Markdown file
Rscript -e "rmarkdown::render('cellsimbench_figs_R.rmd')"
```

**Inputs:**
- `mse_comparison_data_replogle22k562gwps.csv` (from Step 1)
- `per_perturbation_results.csv` (copied from ../calibration/results/)
- `ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X.txt`
- `gsea_input_for_R/{GENE}_ranked_genes.csv` (from Step 2)

**Outputs:**

Main figures in `figures/`:
- `fS1e_S6a.png` - MSE comparison scatter plots (TD and ID vs MB)
- `f2d.png` - Dataset performance - DRF distribution by metric (MSE vs WMSE)
- `fS8.png` - DRF heatmap (technical duplicate baseline)
- `f2e.png` - DRF heatmap (interpolated baseline)
- `fS7_assembly.png` - Supplementary figure assembly with GSEA results

GSEA results:
- `gsea_input_for_R/fgsea_all_results.csv` - Complete GSEA results for all TFs (included in repo for reproducibility)
- `gsea_input_for_R/fgsea_all_results.rds` - GSEA results in R format
- `gsea_input_for_R/fgsea_summary.csv` - Self-enrichment summary

HTML report:
- `cellsimbench_figs_R.html` - Complete analysis report with all figures and statistics

## Analysis Sections

The R Markdown file contains the following analyses:

1. **Supplementary Figure 1E & 6A**: Mean Squared Error comparison between Technical Duplicate and Mean Baseline vs Interpolated Duplicate and Mean Baseline
2. **Figure 2D**: Dataset performance - DRF distribution by metric (MSE vs WMSE)
3. **Supplementary Figure**: Mean DRF across datasets with standard error
4. **Supplementary Figure 8**: DRF heatmap across datasets and metrics (Technical Duplicate baseline)
5. **Figure 2E**: DRF heatmap across datasets and metrics (Interpolated baseline)
6. **Supplementary Figure**: DRF comparison between Replogle K562 datasets (GWPS vs Essential)
7. **GSEA Analysis**: Transcription factor target self-enrichment analysis
8. **Supplementary**: NES ranking plots for top self-enriched TFs
9. **Supplementary**: DRF vs DEG count highlighting top self-enriched TFs
10. **Supplementary Figure 7**: Assembly of multiple analysis panels

## Quick Start

```bash
# Restore R environment (first time only)
Rscript -e "renv::restore()"

# Run Python preprocessing (Steps 1-2)
uv run python compute_replogle_mse.py
uv run python create_gsea_files.py

# Run R analysis (Step 3)
Rscript -e "rmarkdown::render('cellsimbench_figs_R.rmd')"
```



