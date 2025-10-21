#!/usr/bin/env python3
"""Interactive exploration of replogle22k562gwps dataset."""

# %% Imports

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt


# %% Define baseline computation function

def compute_baselines(adata):
    """Compute mean and control baselines for the whole dataset.
    
    This follows the exact same logic as the preprocessing pipeline:
    - Mean baseline: pseudobulk by (donor_id, condition), then average
    - Control baseline: average of all control cells
    
    Args:
        adata: AnnData object with 'donor_id' and 'condition' in obs
        
    Returns:
        dict with 'mean_baseline' and 'control_baseline' as pandas Series
    """
    baselines = {}
    
    # 1. Compute mean baseline (with pseudobulking)
    print("Computing mean baseline...")
    
    # Get all non-control cells
    non_ctrl_cells = adata[~adata.obs['condition'].str.contains('control')]
    
    # Get unique (donor_id, condition) combinations
    unique_donor_condition_combos = non_ctrl_cells.obs[['donor_id', 'condition']].drop_duplicates()
    print(f"  Found {len(unique_donor_condition_combos)} unique (donor, condition) combinations")
    
    # Compute mean for each (donor_id, condition) combination
    condition_means = pd.DataFrame(index=range(len(unique_donor_condition_combos)), columns=adata.var_names)
    
    for idx, (_, row) in enumerate(tqdm(unique_donor_condition_combos.iterrows(), 
                                        desc="  Pseudobulking by condition", 
                                        total=len(unique_donor_condition_combos))):
        donor_id = row['donor_id']
        condition = row['condition']
        
        # Get cells for this specific combination
        combo_cells = non_ctrl_cells[
            (non_ctrl_cells.obs['donor_id'] == donor_id) & 
            (non_ctrl_cells.obs['condition'] == condition)
        ]
        
        if len(combo_cells) > 0:
            # Compute mean expression (pseudobulk)
            combo_mean = combo_cells.X.mean(axis=0)
            if hasattr(combo_mean, 'A1'):  # Handle sparse matrices
                combo_mean = combo_mean.A1
            condition_means.iloc[idx] = combo_mean
    
    # Average across all pseudobulked conditions
    mean_baseline = condition_means.mean(axis=0)
    baselines['mean_baseline'] = pd.Series(mean_baseline, index=adata.var_names)
    print(f"  ✓ Mean baseline computed (shape: {mean_baseline.shape})")
    
    # 2. Compute control baseline
    print("\nComputing control baseline...")
    
    # Get all control cells
    ctrl_cells = adata[adata.obs['condition'] == 'control']
    print(f"  Found {len(ctrl_cells)} control cells")
    
    if len(ctrl_cells) > 0:
        ctrl_mean = ctrl_cells.X.mean(axis=0)
        if hasattr(ctrl_mean, 'A1'):  # Handle sparse matrices
            ctrl_mean = ctrl_mean.A1
        baselines['control_baseline'] = pd.Series(ctrl_mean, index=adata.var_names)
        print(f"  ✓ Control baseline computed (shape: {ctrl_mean.shape})")
    else:
        raise ValueError("No control cells found in dataset!")
    
    return baselines


def compute_pearson_delta(reference, prediction, groundtruth):
    """Compute Pearson delta correlation for each perturbation.
    
    For each perturbation, computes:
    pearson(prediction - reference, groundtruth - reference)
    
    This measures how well the prediction captures the change from baseline.
    
    Args:
        reference: pandas Series with one value per gene (e.g., control baseline)
        prediction: pandas Series with one value per gene (e.g., mean baseline) 
        groundtruth: pandas DataFrame with one row per perturbation, columns are genes
        
    Returns:
        pandas Series with Pearson delta value for each perturbation (index from groundtruth)
    """
    # Convert to float to ensure numeric operations
    reference = pd.Series(reference, dtype=float)
    prediction = pd.Series(prediction, dtype=float)
    groundtruth = pd.DataFrame(groundtruth).astype(float)
    
    # Ensure consistent gene ordering
    genes = reference.index
    if not all(genes == prediction.index):
        raise ValueError("Reference and prediction must have the same gene ordering")
    if not all(genes == groundtruth.columns):
        groundtruth = groundtruth[genes]  # Reorder columns to match
    
    # Compute prediction delta (change from reference)
    prediction_delta = prediction - reference
    
    # Initialize results
    pearson_deltas = pd.Series(index=groundtruth.index, dtype=float)
    
    # Compute Pearson delta for each perturbation
    for perturbation in groundtruth.index:
        # Get groundtruth for this perturbation
        gt_values = groundtruth.loc[perturbation]
        
        # Compute groundtruth delta (change from reference)
        gt_delta = gt_values - reference
        
        # Calculate Pearson correlation between deltas
        # Handle edge cases where variance is zero
        if gt_delta.std() == 0 or prediction_delta.std() == 0:
            pearson_deltas[perturbation] = 0.0
        else:
            # Convert to numpy arrays to ensure compatibility
            corr, _ = pearsonr(prediction_delta.values, gt_delta.values)
            pearson_deltas[perturbation] = corr
    
    return pearson_deltas


# %% Read data
# Define file path
data_path = Path("/home/gabriel/CellSimBench/data/replogle22k562gwps/replogle22k562gwps_processed_complete.h5ad")
print(f"Loading data from: {data_path}")
print("-" * 80)

# Load the AnnData object
adata = sc.read_h5ad(data_path)
print(f"✓ Data loaded successfully!")
print("=" * 80)

# %% Compute baselines for the whole dataset

# Compute both baselines
baselines = compute_baselines(adata)
 
# %% Compute Pearson delta for each perturbation 

pearson_deltas = compute_pearson_delta(
    reference=baselines['control_baseline'], 
    prediction=baselines['mean_baseline'], 
    groundtruth=adata.uns['technical_duplicate_first_half_baseline'])


# %% Visualize Pearson delta distribution

# Simple histogram
plt.figure(figsize=(10, 6))
sns.set_style("ticks")  # Remove grid
sns.histplot(pearson_deltas, bins=30, kde=True)

# Add median line
median_val = pearson_deltas.median()
plt.axvline(median_val, color='red', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')

plt.xlabel('Pearson Delta Correlation')
plt.ylabel('Count')
plt.title('Pearson Delta Dist Original Control Bias')
plt.legend()
sns.despine()
plt.show()

# Basic statistics
print(f"\nMean: {pearson_deltas.mean():.3f}")
print(f"Median: {pearson_deltas.median():.3f}")
print(f"Std: {pearson_deltas.std():.3f}")