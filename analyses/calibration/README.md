
# Calibration analysis

The goal of this analysis is to calculate calibration scores across all the available datasets and metrics. And then visualize the results.

First:

```bash
uv run bash analyses/calibration/run_all_dataset_baselines.sh
```

Then, 

```bash
uv run python analyses/calibration/calibration_analysis.py
```

