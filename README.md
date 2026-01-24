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

---

## Quickstart

Get up and running with pre-built Docker images and pre-processed datasets.

**Prerequisites:**
- Python ≥3.12
- Docker installed and running
- AWS CLI installed (for downloading datasets)
- OpenAI API key (required if running scLambda)
- *Optional:* A conda environment with R (see `analyses/plotting/README.md`). Only for reproducing R-based plots.

**Recommended Hardware (for full reproduction):**
- 5 GPUs with at least 24GB VRAM each (run with 8xA10 instance).
- 384GB CPU RAM
- 64 CPU cores
- 2TB storage

```bash
# 1. Clone and install
git clone https://github.com/shiftbioscience/Perturbation-Models-Outperform-Baselines
cd Perturbation-Models-Outperform-Baselines
uv sync  # or: pip install -e .
source .venv/bin/activate

# 2. Pull pre-built model Docker images
./scripts/pull_all_models.sh

# 3. Build PRESAGE manually (not distributed due to license restrictions)
./docker/presage/build.sh

# 4. Download pre-processed datasets
./scripts/pull_all_datasets.sh

# 5. Create .env file (required if running scLambda)
echo "OPENAI_API_KEY=<your_api_key>" > .env

# 6. Run a benchmark
cellsimbench train model=fmlp_esm2 dataset=norman19
cellsimbench benchmark model=fmlp_esm2 dataset=norman19
```

---

## Running Model Benchmarks

After setup, train and evaluate models on any of the 14 datasets. 

**NOTE:** All training and benchmarking runs will launch multiple jobs (5 for single-gene prediction datasets, and 2 for combo-gene prediction datasets). Jobs will be automatically scheduled using multiple GPUs when available. Monitor and increase compute resources if necessary.

```bash
# Train a model on a dataset
uv run cellsimbench train model=fmlp_esm2 dataset=norman19

# Run benchmark (prediction + evaluation)
uv run cellsimbench benchmark model=fmlp_esm2 dataset=norman19

# Enable NIR (Nearest In-distribution Reference) analysis (slow)
uv run cellsimbench benchmark model=fmlp_esm2 dataset=norman19 +run_nir_analysis=true

# Train and benchmark across multiple datasets
for dataset in norman19 wessels23; do
    uv run cellsimbench train model=fmlp_esm2 dataset=$dataset
    uv run cellsimbench benchmark model=fmlp_esm2 dataset=$dataset
done
```

**Available models:** `fmlp_esm2`, `fmlp_geneformer`, `fmlp_scgpt`, `fmlp_genept`, `gears`, `sclambda`, `scgpt`, `presage`

**Available datasets:** `adamson16`, `frangieh21`, `kaden25fibroblast`, `kaden25rpe1`, `nadig25hepg2`, `nadig25jurkat`, `norman19`, `replogle22k562`, `replogle22k562gwps`, `replogle22rpe1`, `sunshine23`, `tian21crispra`, `tian21crispri`, `wessels23`

---

## Reproducing the Analyses

### 1. Perturbation Strength Analysis

Examines how the mean baseline is a better estimator than the technical duplicate baseline when genes are not significantly detected as DEGs. Also shows the effect of sample size on the technical duplicate baseline:

```bash
uv run python analyses/perturbation_strength/perturbation_strength.py
```

### 2. Calibration Analysis

Computes the Dynamic Range Fraction (DRF) and other calibration metrics across all datasets and evaluation metrics.

```bash
# Run baseline calculations for all datasets
bash analyses/calibration/run_all_dataset_baselines.sh

# Generate calibration plots and statistics
uv run python analyses/calibration/calibration_analysis.py
```

**Outputs:**
- `analyses/calibration/baseline_outputs/*/` - Per-dataset baseline predictions and metrics
- `analyses/calibration/results/` - Calibration plots and summary statistics

### 3. Multi-Model Comparisons (`modelgroup`)

After running model benchmarks (`cellsimbench train ...` and `cellsimbench benchmark...`), you can generate nice summary visualizations for each. In the output directory, after running `cellsimbench benchmark...`, you will see a `detailed_metrics.csv` file. You can use this to generate the figures.

```bash
DATASET="norman19"
RUN_TIMESTAMP="2026-01-20_21-51-09"  # This is the timestamp of the benchmark run, automatically generated for the given run.
python scripts/plot_multimodel_summary.py outputs/fmlp_esm2_${DATASET}/${RUN_TIMESTAMP}/detailed_metrics.csv
```

If you want to generate benchmarking outputs and figures for **multiple models**, you can use the `modelgroup` input type. Instructions:

1. Create a new config file in `cellsimbench/configs/modelgroup/` with the desired models. For example, `cellsimbench/configs/modelgroup/simplebenchmark.yaml`:

```yaml
models:
  - sclambda
  - fmlp_esm2

description: "Compare some different perturbation response models" 
```


2. Run the benchmark (if you already ran the benchmarks for each model individually, this will be much faster):

```bash
cellsimbench benchmark modelgroup=simplebenchmark dataset=wessels23
```

Output:

```shell
...other output...
[2026-01-23 15:26:57,054][cellsimbench.core.plotting_engine][INFO] - All plots saved to outputs/benchmark_sclambda_fmlp_esm2_wessels23/2026-01-23_15-26-30/plots
[2026-01-23 15:26:57,054][cellsimbench.core.benchmark][INFO] - K-fold benchmark completed successfully
[2026-01-23 15:26:57,059][cellsimbench.cli][INFO] - Benchmark completed.
```

3. Generate the figures (timestamp is automatically generated by CellSimBench --- see the output above). The results will be in the `additional_results` directory within the same output directory as the benchmarking results:

```bash
RUN_TIMESTAMP="2026-01-23_15-26-30"
python scripts/plot_multimodel_summary.py outputs/benchmark_sclambda_fmlp_esm2_wessels23/${RUN_TIMESTAMP}/detailed_metrics.csv
```

```shell
...other output...
================================================================================
✅ All plots generated successfully!
Output directory: outputs/benchmark_sclambda_fmlp_esm2_wessels23/2026-01-23_15-26-30/additional_results
================================================================================
```

### 4. Additional Misc Analyses

Generates some additional main and supplementary figures including MSE comparisons, DRF heatmaps, and GSEA self-enrichment analysis:

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

See `analyses/plotting/README.md` for detailed documentation and the preprint for full details on metric definitions and calibration assessment.

---

## Building From Scratch

This section is for users who want to rebuild the datasets and Docker containers from source rather than using the pre-built versions.

### Data Preparation

Process all 14 Perturb-seq datasets from raw sources:

```bash
# Step 1: Download and preprocess all datasets
# This runs each dataset's get_data.py script in parallel
python data/run_all_get_data.py --workers 4

# Step 2: Calculate ground truth DEGs (first half of technical duplicates)
python data/add_ground_truth_degs.py --all --workers 4

# Step 3: Add interpolated duplicate baseline (positive control)
python data/add_interpolated_baseline.py --all --workers 4
```

**Note:** The full pipeline may take several hours depending on your system. Each script supports `--force` to recompute existing data.

**What these scripts do:**
- `run_all_get_data.py` - Downloads raw data from sources, performs QC, creates train/test splits
- `add_ground_truth_degs.py` - Calculates DEGs from first half of tech duplicate splits for calibration
- `add_interpolated_baseline.py` - Creates interpolated duplicate baseline (key positive control)

### Generating Gene Embeddings (Optional)

Required only if you plan to run the fMLP models (fmlp_esm2, fmlp_geneformer, fmlp_genept):

```bash
# Create the required conda environments
conda env create -f data/gene_embeddings/envs/src-geneformer.yaml
conda env create -f data/gene_embeddings/envs/src-esm2.yaml
conda env create -f data/gene_embeddings/envs/src-scgpt.yaml

# Generate embeddings for Norman19 (reference dataset)
bash data/gene_embeddings/gather_embeddings.sh \
    data/norman19/norman19_processed_complete.h5ad \
    data/norman19/norman19_processed_complete.embeddings.h5ad

# Transfer gene embeddings to all other datasets
python data/batch_transfer_embeddings.py --workers 4
```

**Requirements:**
- Conda must be installed and available in PATH
- The conda environments (`src-scgpt`, `src-geneformer`, `src-esm2`) must be created before running `gather_embeddings.sh`
- The geneformer package will be automatically installed from HuggingFace when the script runs

**Caching behavior:**
- The script caches intermediate results for each stage (GenePT, scGPT, Geneformer, ESM2)
- If a stage has already been completed, it will be skipped on subsequent runs
- To force regeneration of all embeddings, add `--force` flag:
  ```bash
  bash data/gene_embeddings/gather_embeddings.sh INPUT OUTPUT --force
  ```

### Building Model Docker Containers

Each model runs in an isolated Docker container. Build them from source:

```bash
# Build all models
bash docker/fmlp/build.sh
bash docker/gears/build.sh
bash docker/sclambda/build.sh
bash docker/scgpt/build.sh
bash docker/presage/build.sh

# Or build individually as needed
bash docker/<model_name>/build.sh
```

**Note:** Docker must be installed and running. Building all containers may take 30-60 minutes.

---

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
