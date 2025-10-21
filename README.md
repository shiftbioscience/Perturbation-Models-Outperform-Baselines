# Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines on Well-Calibrated Metrics

**Preprint:** https://www.biorxiv.org/content/10.1101/2025.10.20.683304v1

This repository contains code to reproduce the analyses from our preprint. We introduce a framework for evaluating benchmark metric calibration using positive and negative controls, and demonstrate that deep learning perturbation models outperform uninformative baselines when evaluated with well-calibrated metrics.


## Licensing

The core framework is released under the MIT License.  
Certain benchmark components use third-party models under their original terms:
- `docker/presage/` – Genentech Non-Commercial Software License v1.0  
- `docker/sclambda/` – GNU General Public License v3.0 (derivative work)

Note that the PRESAGE component is redistributed for **academic
use only** under the terms of the **Genentech Non-Commercial Software
License v1.0 (2022)**. See `docker/presage/LICENSE` and `docker/presage/NOTICE`
for full text and attribution. 


## Installation

```bash
# Clone the repository
git clone https://github.com/shiftbioscience/Perturbation-Models-Outperform-Baselines
cd Perturbation-Models-Outperform-Baselines

# Install using uv (recommended) or pip
uv pip install -e .
# OR
pip install -e .
```

**Requirements:** Python ≥3.12

## Data Preparation

Process all 14 Perturb-seq datasets (~2.5M cells across 9 cell types):

```bash
# Step 1: Download and preprocess all datasets
# This runs each dataset's get_data.py script in parallel
python data/run_all_get_data.py --workers 4

# Step 2: Calculate ground truth DEGs (first half of technical duplicates)
python data/add_ground_truth_degs.py --all --workers 4

# Step 3: Add interpolated duplicate baseline (positive control)
python data/add_interpolated_baseline.py --all --workers 4

# Step 4: Generate foundation model embeddings (ESM2, Geneformer, GenePT) for Norman19 dataset
# See data/gene_embeddings/gene_embeddings_README.md for environment setup
# This step is required before transferring embeddings to other datasets
bash data/gene_embeddings/gather_embeddings.sh \
    data/norman19/norman19_processed_complete.h5ad \
    data/norman19/norman19_processed_complete.embeddings.h5ad

# Step 5: Transfer gene embeddings across all datasets
# Uses norman19 as reference for ESM2, Geneformer, GenePT embeddings
python data/batch_transfer_embeddings.py --workers 4
```

**Note:** The full pipeline may take several hours depending on your system. Each script supports `--force` to recompute existing data.

**What these scripts do:**
- `run_all_get_data.py` - Downloads raw data from sources, performs QC, creates train/test splits
- `add_ground_truth_degs.py` - Calculates DEGs from first half of tech duplicate splits for calibration
- `add_interpolated_baseline.py` - Creates interpolated duplicate baseline (key positive control)
- `gather_embeddings.sh` - Generates foundation model gene embeddings (ESM2, Geneformer, GenePT)
- `batch_transfer_embeddings.py` - Transfers embeddings from reference dataset to all other datasets

## Building Model Docker Containers

Each model runs in an isolated Docker container. To reproduce model benchmarks, build the required containers:

```bash
# Build all models
bash docker/fmlp/build.sh
bash docker/gears/build.sh
bash docker/sclambda/build.sh
bash docker/scgpt/build.sh
bash docker/presage/build.sh
bash docker/geneformer/build.sh

# Or build individually as needed
bash docker/<model_name>/build.sh
```

**Note:** Docker must be installed and running. Building all containers may take 30-60 minutes.

## Running Model Benchmarks

After building containers and downloading data, train and evaluate models:

```bash
# Train a model on a dataset
uv run cellsimbench train model=fmlp_esm2 dataset=norman19

# Run benchmark (prediction + evaluation)
uv run cellsimbench benchmark model=fmlp_esm2 dataset=norman19

# Train on all datasets (for multi-dataset analysis)
for dataset in adamson16 norman19 replogle22k562 replogle22rpe1; do
    uv run cellsimbench train model=fmlp_esm2 dataset=$dataset
    uv run cellsimbench benchmark model=fmlp_esm2 dataset=$dataset
done
```

**Available models:** `fmlp_esm2`, `fmlp_geneformer`, `fmlp_genept`, `gears`, `sclambda`, `scgpt`, `presage`, `geneformer`

## Reproducing the Analyses

### 1. Calibration Analysis (Figure 1B, Main Results)

This analysis computes the Dynamic Range Fraction (DRF) and other calibration metrics across all datasets and evaluation metrics.

```bash
# Run baseline calculations for all datasets
bash analyses/calibration/run_all_dataset_baselines.sh

# Generate calibration plots and statistics
python analyses/calibration/calibration_analysis.py
```

**Outputs:**
- `analyses/calibration/baseline_outputs/*/` - Per-dataset baseline predictions and metrics
- `analyses/calibration/results/` - Calibration plots and summary statistics

### 2. Metric Problem Analyses (Figures 2-4)

These scripts demonstrate specific pathologies in common evaluation metrics:

```bash
cd analyses/metric_problems

# Control bias analysis (Supplementary Figure)
python control_bias.py

# Signal dilution analysis (Figure 3)
python signal_dilution.py

# DEG sparsity and coherence (Figure 4)
python effect_sparsity.py
python deg_coherence.py

# Expression-effect relationship
python expression_effect.py

# Cell titration experiments
python cell_titration.py

# Generate main paper plots
python paper_plots.py
```

### 3. Multi-Model Comparison Plots

After running model benchmarks, generate summary visualizations:

```bash
python scripts/plot_multimodel_summary.py outputs/benchmark_*/detailed_metrics.csv
```

## Analysis Scripts Overview

**Calibration Analysis** (`analyses/calibration/`):
- Computes positive controls (interpolated duplicate, technical duplicate) and negative controls (mean baseline, control baseline)
- Calculates Dynamic Range Fraction (DRF) for 13 evaluation metrics across 14 datasets
- Generates calibration heatmaps and per-dataset comparisons

**Metric Problems** (`analyses/metric_problems/`):
- `control_bias.py` - Demonstrates systematic differences between control and perturbed cells
- `signal_dilution.py` - Shows how sparse DEG signals are diluted in unweighted metrics
- `effect_sparsity.py` - Analyzes the relationship between DEG count and metric calibration
- `deg_coherence.py` - Examines consistency of DEG detection across replicates
- `cell_titration.py` - Tests metric behavior with varying cell counts
- `paper_plots.py` - Generates publication-ready figures

## Key Metrics Evaluated

- **Poorly calibrated**: MSE, Pearson(Δctrl), R²(Δctrl)
- **Well calibrated**: WMSE, Weighted R²(Δ), Normalized Inverse Rank (NIR)

See the preprint for full details on metric definitions and calibration assessment.

## Citation

If you use this code or data, please cite:

```bibtex
@article{miller2025perturbation,
  title={Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines on Well-Calibrated Metrics},
  author={Miller, Henry E. and Mejia, Gabriel M. and Leblanc, Francis J. A. and Wang, Bo and Swain, Brendan and Camillo, Lucas Paulo de Lima},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.10.20.683304}
}
```

## License

This project is made available under the CC-BY 4.0 International license.

## Contact

For questions, contact: henry@shiftbioscience.com



