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


# %% Define auxiliary functions

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

def compute_mse_per_perturbation(prediction, groundtruth):
    """Compute MSE between prediction and groundtruth for each perturbation.
    
    Args:
        prediction: pandas DataFrame with one row per perturbation, columns are genes
        groundtruth: pandas DataFrame with one row per perturbation, columns are genes
        
    Returns:
        pandas Series with MSE value for each perturbation (index from groundtruth)
    """
    # Convert to float to ensure numeric operations
    prediction = pd.DataFrame(prediction).astype(float)
    groundtruth = pd.DataFrame(groundtruth).astype(float)
    
    # Ensure consistent gene ordering
    genes = groundtruth.columns
    if not all(genes == prediction.columns):
        prediction = prediction[genes]  # Reorder columns to match
    
    # Ensure consistent perturbation ordering
    perturbations = groundtruth.index
    if not all(perturbations == prediction.index):
        prediction = prediction.loc[perturbations]  # Reorder rows to match
    
    # Initialize results
    mse_values = pd.Series(index=perturbations, dtype=float)
    
    # Compute MSE for each perturbation
    for perturbation in perturbations:
        # Get prediction and groundtruth for this perturbation
        pred_values = prediction.loc[perturbation]
        gt_values = groundtruth.loc[perturbation]
        
        # Calculate MSE
        mse = ((pred_values - gt_values) ** 2).mean()
        mse_values[perturbation] = mse
    
    return mse_values

def compute_pearson_delta_per_perturbation(reference, prediction, groundtruth):
    """Compute Pearson delta correlation for each perturbation.
    
    For each perturbation, computes:
    pearson(prediction - reference, groundtruth - reference)
    
    This measures how well the prediction captures the change from baseline.
    
    Args:
        reference: pandas Series with one value per gene (e.g., control baseline)
        prediction: pandas DataFrame with one row per perturbation, columns are genes
        groundtruth: pandas DataFrame with one row per perturbation, columns are genes
        
    Returns:
        pandas Series with Pearson delta value for each perturbation (index from groundtruth)
    """
    # Convert to float to ensure numeric operations
    reference = pd.Series(reference, dtype=float)
    prediction = pd.DataFrame(prediction).astype(float)
    groundtruth = pd.DataFrame(groundtruth).astype(float)
    
    # Ensure consistent gene ordering
    genes = reference.index
    if not all(genes == groundtruth.columns):
        groundtruth = groundtruth[genes]  # Reorder columns to match
    if not all(genes == prediction.columns):
        prediction = prediction[genes]  # Reorder columns to match
    
    # Ensure consistent perturbation ordering
    perturbations = groundtruth.index
    if not all(perturbations == prediction.index):
        prediction = prediction.loc[perturbations]  # Reorder rows to match
    
    # Initialize results
    pearson_deltas = pd.Series(index=perturbations, dtype=float)
    
    # Compute Pearson delta for each perturbation
    for perturbation in perturbations:
        # Get prediction and groundtruth for this perturbation
        pred_values = prediction.loc[perturbation]
        gt_values = groundtruth.loc[perturbation]
        
        # Compute prediction delta (change from reference)
        pred_delta = pred_values - reference
        
        # Compute groundtruth delta (change from reference)
        gt_delta = gt_values - reference
        
        # Calculate Pearson correlation between deltas
        # Handle edge cases where variance is zero
        if gt_delta.std() == 0 or pred_delta.std() == 0:
            pearson_deltas[perturbation] = 0.0
        else:
            # Convert to numpy arrays to ensure compatibility
            corr, _ = pearsonr(pred_delta.values, gt_delta.values)
            pearson_deltas[perturbation] = corr
    
    return pearson_deltas


# %% Read data
data_path = Path("./data/replogle22k562gwps/replogle22k562gwps_processed_complete.h5ad")
print(f"Loading data from: {data_path}")
print("-" * 80)

# Load the AnnData object
adata = sc.read_h5ad(data_path)
print(f"✓ Data loaded successfully!")
print("=" * 80)

# %% Compute baselines for the whole dataset

# Compute both baselines
baselines = compute_baselines(adata)

# %% Compute MSE per perturbation when predicting technical duplicate and mean baseline

# Get technical duplicate data
technical_duplicate_first_half = adata.uns['technical_duplicate_first_half_baseline']
technical_duplicate_second_half = adata.uns['technical_duplicate_second_half_baseline']

# Convert mean baseline to DataFrame format (repeat for each perturbation)
mean_baseline_df = pd.DataFrame(
    [baselines['mean_baseline']] * len(technical_duplicate_second_half),
    index=technical_duplicate_second_half.index,
    columns=technical_duplicate_second_half.columns
)

# Compute MSEs
mse_mean_baseline = compute_mse_per_perturbation(
    mean_baseline_df,
    technical_duplicate_second_half
)
mse_technical_duplicate = compute_mse_per_perturbation(
    technical_duplicate_first_half,
    technical_duplicate_second_half
)

# %% Compute pearson delta per perturbation when predicting technical duplicate and mean baseline

# Convert mean baseline to DataFrame format
mean_baseline_df_for_pearson = pd.DataFrame(
    [baselines['mean_baseline']] * len(technical_duplicate_second_half),
    index=technical_duplicate_second_half.index,
    columns=technical_duplicate_second_half.columns
)

pearson_delta_mean_baseline = compute_pearson_delta_per_perturbation(
    reference=baselines['control_baseline'],
    prediction=mean_baseline_df_for_pearson,
    groundtruth=technical_duplicate_second_half
)
pearson_delta_technical_duplicate = compute_pearson_delta_per_perturbation(
    reference=baselines['control_baseline'],
    prediction=technical_duplicate_first_half,
    groundtruth=technical_duplicate_second_half
)

# Create combined results DataFrame
results = pd.DataFrame({
    'MSE Mean baseline': mse_mean_baseline,
    'MSE Technical duplicate': mse_technical_duplicate,
    'Pearson Delta Mean baseline': pearson_delta_mean_baseline,
    'Pearson Delta Technical duplicate': pearson_delta_technical_duplicate
})

# %% Add num of degs to the results

results['Num DEGs'] = 0
for pert in results.index:
    if pert not in adata.uns['deg_gene_dict']:
        continue
    results.loc[pert, 'Num DEGs'] = len(adata.uns['deg_gene_dict'][pert])

results.sort_values(by='Num DEGs', ascending=False, inplace=True)
results['Rank'] = np.arange(len(results))

# Remove control perturbations from the results
results = results[~results.index.str.contains('control')]


# %% Create elegant 2x2 subplot layout

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Baseline Comparison: Mean vs Technical Duplicate', fontsize=16, y=0.95)

# Shared colormap data
c = np.log10(results['Num DEGs'] + 1)

def add_identity_annotations(ax, x, y):
    """Add identity line and percentage annotations.

    The annotation positions are robust to negative axis limits.
    """
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.7)

    above, below = np.sum(x > y), np.sum(x < y)
    frac_above, frac_below = above / len(x), below / len(x)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # Top-left: y > x (below identity line)
    ax.text(
        xlim[0] + 0.05 * (xlim[1] - xlim[0]),
        ylim[1] - 0.05 * (ylim[1] - ylim[0]),
        f'{frac_below:.1%}',
        fontsize=15,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        va='top', ha='left'
    )
    # Bottom-right: x > y (above identity line)
    ax.text(
        xlim[1] - 0.05 * (xlim[1] - xlim[0]),
        ylim[0] + 0.05 * (ylim[1] - ylim[0]),
        f'{frac_above:.1%}',
        fontsize=15,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        va='bottom', ha='right'
    )

# Left-top: MSE vs Rank
results_mse = results.sort_values('MSE Technical duplicate', ascending=False)
rank_mse = np.arange(len(results_mse))
c_mse = np.log10(results_mse['Num DEGs'] + 1)  # Correct colors for sorted data
sc = axes[0,0].scatter(rank_mse, np.log10(results_mse['MSE Mean baseline']), c=c_mse, cmap='viridis', s=30, alpha=0.8)
axes[0,0].scatter(rank_mse, np.log10(results_mse['MSE Technical duplicate']), color='black', s=30, alpha=0.8)
axes[0,0].set_xlabel('Rank', fontsize=14), axes[0,0].set_ylabel('log10(MSE)', fontsize=14), axes[0,0].set_title('MSE vs Rank', fontsize=16)

# Left-bottom: Pearson Delta vs Rank  
results_pearson = results.sort_values('Pearson Delta Technical duplicate', ascending=False)
rank_pearson = np.arange(len(results_pearson))
c_pearson = np.log10(results_pearson['Num DEGs'] + 1)  # Correct colors for sorted data
axes[1,0].scatter(rank_pearson, results_pearson['Pearson Delta Mean baseline'], c=c_pearson, cmap='viridis', s=30, alpha=0.8)
axes[1,0].scatter(rank_pearson, results_pearson['Pearson Delta Technical duplicate'], color='black', s=30, alpha=0.8)
axes[1,0].set_xlabel('Rank', fontsize=14), axes[1,0].set_ylabel('Pearson Delta', fontsize=14), axes[1,0].set_title('Pearson Delta vs Rank', fontsize=16)

# Right-top: MSE comparison with identity line
x_mse, y_mse = np.log10(results['MSE Mean baseline']), np.log10(results['MSE Technical duplicate'])
axes[0,1].scatter(x_mse, y_mse, c=c, cmap='viridis', s=30, alpha=0.8)
add_identity_annotations(axes[0,1], x_mse, y_mse)
axes[0,1].set_xlabel('log10(MSE Mean)', fontsize=14), axes[0,1].set_ylabel('log10(MSE Tech Dup)', fontsize=14), axes[0,1].set_title('MSE Comparison', fontsize=16)

# Right-bottom: Pearson Delta comparison with identity line
x_pearson, y_pearson = results['Pearson Delta Mean baseline'], results['Pearson Delta Technical duplicate']
axes[1,1].scatter(x_pearson, y_pearson, c=c, cmap='viridis', s=30, alpha=0.8)
add_identity_annotations(axes[1,1], x_pearson, y_pearson)
axes[1,1].set_xlabel('Pearson Delta Mean', fontsize=14), axes[1,1].set_ylabel('Pearson Delta Tech Dup', fontsize=14), axes[1,1].set_title('Pearson Delta Comparison', fontsize=16)

# Style and finalize
for ax in axes.flat:
    sns.despine(ax=ax)

# Set square aspect ratio only for identity plots (right column)
axes[0,1].set_aspect('equal', adjustable='box')  # MSE comparison
axes[1,1].set_aspect('equal', adjustable='box')  # Pearson Delta comparison

# Improved colorbar positioning - spans full plot height
plt.tight_layout(rect=[0, 0, 0.90, 0.93])
cbar_ax = fig.add_axes([0.91, 0.08, 0.025, 0.82])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label('log10(Num DEGs + 1)', fontsize=14)

plt.show()