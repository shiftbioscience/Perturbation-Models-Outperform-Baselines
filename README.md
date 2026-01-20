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
uv sync
# OR
pip install -e .

# Activate the environment
. ./.venv/bin/activate

```

**Requirements:** 

1. Environment with python ≥3.12
2. Open AI API key (for running scLambda). Create a file (`.env`) with the line `OPENAI_API_KEY=<your_api_key>`. 



## Data Preparation

Process all 14 Perturb-seq datasets:

```bash
# Step 1: Download and preprocess all datasets
# This runs each dataset's get_data.py script in parallel
python data/run_all_get_data.py --workers 4

# Step 2: Calculate ground truth DEGs (first half of technical duplicates)
python data/add_ground_truth_degs.py --all --workers 4

# Step 3: Add interpolated duplicate baseline (positive control)
python data/add_interpolated_baseline.py --all --workers 4

# Step 4: Generate foundation model embeddings (ESM2, Geneformer, GenePT, scGPT) for Norman19 dataset
# OPTIONAL unless you plan to run the fMLP models.
# First, create the required conda environments:
conda env create -f data/gene_embeddings/envs/src-geneformer.yaml
conda env create -f data/gene_embeddings/envs/src-esm2.yaml
conda env create -f data/gene_embeddings/envs/src-scgpt.yaml

# Then generate embeddings (this step is required before transferring embeddings to other datasets)
bash data/gene_embeddings/gather_embeddings.sh \
    data/norman19/norman19_processed_complete.h5ad \
    data/norman19/norman19_processed_complete.embeddings.h5ad

# Step 5: Transfer gene embeddings across all datasets
# Uses norman19 as reference for ESM2, Geneformer, GenePT embeddings
python data/batch_transfer_embeddings.py --workers 4
```

**Note:** The full pipeline may take several hours depending on your system. Each script supports `--force` to recompute existing data.

**Requirements for Step 4:**
- Conda must be installed and available in PATH
- The conda environments (`src-scgpt`, `src-geneformer`, `src-esm2`) must be created before running `gather_embeddings.sh`
- The script can be run from the project root: `bash data/gene_embeddings/gather_embeddings.sh ...`
- The geneformer package will be automatically installed from HuggingFace when the script runs

**Caching behavior:**
- The script caches intermediate results for each stage (GenePT, scGPT, Geneformer, ESM2)
- If a stage has already been completed, it will be skipped on subsequent runs
- To force regeneration of all embeddings, add `--force` flag:
  ```bash
  bash data/gene_embeddings/gather_embeddings.sh INPUT OUTPUT --force
  ```

**What these scripts do:**
- `run_all_get_data.py` - Downloads raw data from sources, performs QC, creates train/test splits
- `add_ground_truth_degs.py` - Calculates DEGs from first half of tech duplicate splits for calibration
- `add_interpolated_baseline.py` - Creates interpolated duplicate baseline (key positive control)
- `gather_embeddings.sh` - Generates foundation model gene embeddings (GenePT, scGPT, Geneformer, ESM2). Requires conda environments to be created first
- `batch_transfer_embeddings.py` - Transfers gene embeddings from reference dataset to all other datasets

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

### 1. Perturbation Strength Analysis

Examines how the mean baseline is a better estimator than the technical duplicate baseline when genes are not significantly detected as DEGs. Also shows the effect of sample size on the technical duplicate baseline:

```bash
# From project root
uv run python analyses/perturbation_strength/perturbation_strenght.py
```

### 2. Calibration Analysis (`analyses/calibration/`)

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

### 3. Multi-Model Comparison Plots

After running model benchmarks, generate summary visualizations:

```bash
python scripts/plot_multimodel_summary.py outputs/benchmark_*/detailed_metrics.csv
```

### 4. Figure Generation (`analyses/plotting/`)

Generates main and supplementary figures including MSE comparisons, DRF heatmaps, and GSEA self-enrichment analysis:

```bash
cd analyses/plotting

# Copy calibration results
cp ../calibration/results/per_perturbation_results.csv .

# Compute MSE comparisons and create GSEA input files
uv run python compute_replogle_mse.py
uv run python create_gsea_files.py

# Generate figures (requires R with renv)
Rscript -e "renv::restore()"
Rscript -e "rmarkdown::render('cellsimbench_figs_R.rmd')"
```

See `analyses/plotting/README.md` for detailed documentation.

See the preprint for full details on metric definitions and calibration assessment.

## Citation

If you use this code or data, please cite:

```bibtex
@article{miller2025perturbation,
  title={Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines on Well-Calibrated Metrics},
  author={Miller, Henry E. and Mejia, Gabriel M. and Leblanc, Francis J. A., and Swain, Brendan, and Wang, Bo, and Camillo, Lucas Paulo de Lima},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.10.20.683304}
}
```

## Contact

For questions, contact: henry@shiftbioscience.com



