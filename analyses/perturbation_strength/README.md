# Perturbation Strength Analysis

This folder contains analyses examining how the mean baseline is a better estimator than the technical duplicate baseline when genes are not significantly detected as DEGs. Also shows the effect of sample size on the technical duplicate baseline.

## Scripts

### `perturbation_strenght.py`
Generates supplementary figures comparing Technical Duplicate baseline vs Mean Baseline performance across perturbations with varying numbers of DEGs.

**Figures generated:**
- `sup_fig_error_analysis.png` — Analysis of how the significance of a gene being recognized as a DEG affects the error estimation of both the mean baseline and the technical duplicate baseline
  
- `sup_fig_sample_size_effect.png` — Cell titration experiment showing how Technical Duplicate estimation error decreases with sampling but not reaching the Mean Baseline for weak perturbations

- `sup_fig_theoretical_model.png` — Wide p-value distribution plot for a perturbation with ~30 DEGs. Only used for theoretical diagram.

## Usage

To run the script, use the following command:

```bash
# From project root
uv run python analyses/perturbation_strength/perturbation_strenght.py

# Or interactively from this directory (cell-by-cell execution supported)
```

Processed data should be present in the `data` folder. Valid datasets are (choose dataset name inside code if needed):

- `replogle22k562gwps` - Replogle K562 GWPS dataset (default)
- `replogle22rpe1` - Replogle RPE1 dataset
- `replogle22k562` - Replogle K562 dataset
