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
from matplotlib.colors import LogNorm



# %% Read data
path_dict = {
    "replogle22k562gwps": Path("./data/replogle22k562gwps/replogle22k562gwps_processed_complete.h5ad"), # 256 cells
    "replogle22rpe1": Path("./data/replogle22rpe1/replogle22rpe1_processed_complete.h5ad"), # 128 cells
    "replogle22k562": Path("./data/replogle22k562/replogle22k562_processed_complete.h5ad") # 128 cells
}

dataset_name = "replogle22k562gwps"

data_path = path_dict[dataset_name]
print(f"Loading data from: {data_path}")
print("-" * 80)

# Load the AnnData object
adata = sc.read_h5ad(data_path)
print(f"✓ Data loaded successfully!")
print("=" * 80)

# %% Get perturbation with most DEGs

deg_gene_dict = adata.uns.get('deg_gene_dict_gt', {})

if not deg_gene_dict:
    print("WARNING: No deg_gene_dict found")
else:
    # Get DEG counts for all perturbations (excluding controls)
    deg_counts = pd.Series({
        pert_name: len(deg_genes) 
        for pert_name, deg_genes in deg_gene_dict.items()
        if 'control' not in pert_name.lower()
    })
    
    # Sort by DEG count and select 3 perturbations: most DEGs, ~30 DEGs, and ~5 DEGs
    deg_counts = deg_counts.sort_values(ascending=False)
    
    # Find perturbations
    left_pert = deg_counts.index[0]  # Most DEGs
    middle_pert = deg_counts.index[(deg_counts - 30).abs().argmin()]
    right_pert = deg_counts.index[(deg_counts - 5).abs().argmin()]
    
    selected_perturbations = [left_pert, middle_pert, right_pert]
    selected_labels = ['Most DEGs', '~30 DEGs', '~5 DEGs']
    
    print(f"\nSelected perturbations:")
    for label, pert in zip(selected_labels, selected_perturbations):
        n_degs = deg_counts[pert]
        print(f"  {label}: {pert} (n={n_degs} DEGs)")

# %% Plot ranking plots, identity plots, and ratio plots for 3 selected perturbations

# Create figure with 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(18, 17))

for i, (perturbation, label) in enumerate(zip(selected_perturbations, selected_labels)):
    # Row 1: ranking plot
    ax_top = axes[0, i]
    
    # Get adjusted p-values and gene names (stored as lists)
    pvals = adata.uns['pvals_adj_df_dict_gt'][perturbation]
    gene_names = adata.uns['names_df_dict_gt'][perturbation]
    
    # Convert lists to Series for easier manipulation
    pvals_series = pd.Series(pvals, index=gene_names)
    
    # Sort by p-value (descending, so least significant first)
    pvals_series_sorted = pvals_series.sort_values(ascending=False)
    
    # Create ranking
    rank = np.arange(len(pvals_series_sorted))
    one_minus_pval = 1 - pvals_series_sorted.values
    
    # Categorize genes by p-value thresholds
    is_deg = one_minus_pval > 0.95
    is_intermediate = (one_minus_pval > 0.05) & (one_minus_pval <= 0.95)
    is_ns = one_minus_pval <= 0.05
    
    # Plot ranking with color coding
    ax_top.scatter(rank[is_ns], one_minus_pval[is_ns], c='gray', s=10, alpha=1.0)
    ax_top.scatter(rank[is_intermediate], one_minus_pval[is_intermediate], c='cornflowerblue', s=10, alpha=1.0)
    ax_top.scatter(rank[is_deg], one_minus_pval[is_deg], c='darkblue', s=10, alpha=1)
    ax_top.axhline(y=0.95, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    ax_top.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    ax_top.set_xlabel('Gene Rank (by significance)', fontsize=18)
    if i == 0:
        ax_top.set_ylabel('1 - adjusted p-value', fontsize=18)
    ax_top.set_title(f'{perturbation.split("_")[1]}\n{is_deg.sum()} DEGs', fontsize=18, fontweight='bold')
    ax_top.set_xlim([0, len(pvals_series_sorted) + 100])
    ax_top.set_ylim([0, 1.05])
    ax_top.grid(True, alpha=0.3, axis='y')
    ax_top.set_axisbelow(True)
    sns.despine(ax=ax_top)
    
    # Row 2: ratio plot (Tech. Dup. Error / Mean Baseline Error vs 1 - adjusted p-value)
    ax_ratio = axes[1, i]
    
    # Get baseline data for ratio calculation
    groundtruth = adata.uns['technical_duplicate_first_half_baseline'].loc[perturbation]
    td = adata.uns['technical_duplicate_second_half_baseline'].loc[perturbation]
    mean_baseline = adata.uns['split_fold_0_mean_baseline'].iloc[0]
    
    # Align genes
    common_genes = sorted(set(groundtruth.index) & set(td.index) & set(mean_baseline.index) & set(gene_names))
    pvals_aligned = pvals_series[common_genes]
    
    # Compute error ratio (TD Error / Mean Baseline Error)
    errors_td_ratio = np.abs(groundtruth[common_genes] - td[common_genes])
    errors_mean_ratio = np.abs(groundtruth[common_genes] - mean_baseline[common_genes])
    error_ratio = errors_td_ratio / errors_mean_ratio
    
    # Get 1 - adjusted p-value for common genes
    one_minus_pval_ratio = 1 - pvals_aligned.values
    
    # Categorize by p-value for ratio plot
    one_minus_pval_aligned = 1 - pvals_aligned.values
    is_deg_ratio = one_minus_pval_aligned > 0.95
    is_intermediate_ratio = (one_minus_pval_aligned > 0.05) & (one_minus_pval_aligned <= 0.95)
    is_ns_ratio = one_minus_pval_aligned <= 0.05
    
    # Plot with color coding
    ax_ratio.scatter(one_minus_pval_ratio[is_ns_ratio], error_ratio[is_ns_ratio], c='gray', s=10, alpha=0.7)
    ax_ratio.scatter(one_minus_pval_ratio[is_intermediate_ratio], error_ratio[is_intermediate_ratio], c='cornflowerblue', s=10, alpha=0.7)
    ax_ratio.scatter(one_minus_pval_ratio[is_deg_ratio], error_ratio[is_deg_ratio], c='darkblue', s=10, alpha=0.7)
    
    # Add horizontal line at y=1 (equal performance)
    ax_ratio.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Add vertical lines for p-value thresholds
    ax_ratio.axvline(x=0.95, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    ax_ratio.axvline(x=0.05, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax_ratio.set_xlabel('1 - adjusted p-value', fontsize=18)
    if i == 0:
        ax_ratio.set_ylabel('Tech. Dup. Error / Mean Baseline Error', fontsize=18)
    ax_ratio.set_xlim([0, 1.05])
    ax_ratio.set_ylim([0, 3])
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.set_axisbelow(True)
    sns.despine(ax=ax_ratio)
    
    # Row 3: identity/density plot (Tech. Dup. error vs Mean baseline error)
    # Reuse the same data from row 2 to ensure alignment
    ax_bottom = axes[2, i]
    
    # Compute log10 errors using the same aligned data from row 2
    errors_td_log = np.log10(errors_td_ratio + 1e-10)
    errors_mean_log = np.log10(errors_mean_ratio + 1e-10)
    
    # Reuse the same categorization from row 2
    is_deg_aligned = is_deg_ratio
    is_intermediate_aligned = is_intermediate_ratio
    is_ns_aligned = is_ns_ratio
    
    # Plot density plot with color-coded contours for each category
    categories_plot = [
        (is_ns_aligned, 'gray', 6),
        (is_intermediate_aligned, 'cornflowerblue', 6),
        (is_deg_aligned, 'darkblue', 6)
    ]
    
    for cat_mask, color, levels in categories_plot:
        if cat_mask.sum() > 1:
            sns.kdeplot(x=errors_td_log[cat_mask], y=errors_mean_log[cat_mask], 
                       ax=ax_bottom, color=color, levels=levels, alpha=0.6, linewidths=1.5)
    
    # Add identity line
    lims = [min(errors_td_log.min(), errors_mean_log.min()), max(errors_td_log.max(), errors_mean_log.max())]
    ax_bottom.plot(lims, lims, 'k--', alpha=0.7, linewidth=1.5)
    
    # Set intelligent axis limits based on data percentiles to avoid outliers
    x_min, x_max = np.percentile(errors_td_log, [1, 99])
    y_min, y_max = np.percentile(errors_mean_log, [1, 99])
    margin = 0.25 * max(x_max - x_min, y_max - y_min)
    ax_bottom.set_xlim([x_min - margin, x_max + margin])
    ax_bottom.set_ylim([y_min - margin, y_max + margin])
    
    # Calculate proportions above diagonal for each category
    categories = [
        ('DEGs', is_deg_aligned, 'darkblue'),
        ('Int', is_intermediate_aligned, 'cornflowerblue'),
        ('NS', is_ns_aligned, 'gray')
    ]
    
    y_pos = 0.98
    for cat_name, cat_mask, color in categories:
        if cat_mask.sum() > 0:
            above_diag = (errors_mean_log[cat_mask] > errors_td_log[cat_mask]).sum()
            total = cat_mask.sum()
            prop_above = above_diag / total
            text = f'{cat_name}: {prop_above:.1%}'
        else:
            text = f'{cat_name}: NA'
        
        ax_bottom.text(0.02, y_pos, text, transform=ax_bottom.transAxes,
                      fontsize=16, va='top', ha='left', color=color, fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color))
        y_pos -= 0.08
    
    ax_bottom.set_xlabel('log10(Tech. Dup. Error)', fontsize=18)
    if i == 0:
        ax_bottom.set_ylabel('log10(Mean Baseline Error)', fontsize=18)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.set_axisbelow(True)
    sns.despine(ax=ax_bottom)

plt.tight_layout()
plt.show()

# %% Cell titration experiment for 3 selected perturbations

def cell_titration_simplified(adata, perturbation_name, n_points=20, random_seed=42):
    """
    Simplified cell titration experiment for Tech. Dup. and Mean Baseline only.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing all data
    perturbation_name : str
        Name of the perturbation to analyze
    n_points : int
        Number of points in the linear space (default: 20)
    random_seed : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with n_cells, td_mse, and mb_mse columns
    """
    np.random.seed(random_seed)
    
    # Find cells that match this perturbation in condition column
    pert_mask = adata.obs['condition'] == perturbation_name.split('_')[1]
    pert_cells = adata[pert_mask]
    
    if pert_cells.n_obs == 0:
        raise ValueError(f"No cells found for perturbation '{perturbation_name}'")
    
    # Split perturbation cells randomly into GT and TD groups
    n_pert_cells = pert_cells.n_obs
    gt_indices = np.random.choice(n_pert_cells, size=n_pert_cells//2, replace=False)
    td_indices = np.setdiff1d(np.arange(n_pert_cells), gt_indices)
    
    gt_cells = pert_cells[gt_indices]
    td_cells = pert_cells[td_indices]
    
    # Get ground truth (average of GT cells)
    gt_profile_raw = gt_cells.X.mean(axis=0)
    gt_profile = np.asarray(gt_profile_raw).flatten()
    
    # Get mean baseline (same as used throughout)
    mean_baseline = adata.uns['split_fold_0_mean_baseline'].iloc[0].values
    
    # Compute MSE between GT and mean baseline (constant for all points)
    mb_mse = np.mean((gt_profile - mean_baseline) ** 2)
    
    # Define linear space for cell numbers
    max_cells = td_cells.n_obs
    cell_numbers = np.linspace(1, max_cells, n_points, dtype=int)
    
    results = []
    
    for n_cells in cell_numbers:
        # Sample n_cells from TD group
        td_sample_indices = np.random.choice(td_cells.n_obs, size=n_cells, replace=False)
        td_sample = td_cells[td_sample_indices]
        td_profile_raw = td_sample.X.mean(axis=0)
        td_profile = np.asarray(td_profile_raw).flatten()
        td_mse = np.mean((gt_profile - td_profile) ** 2)
        
        results.append({
            'n_cells': n_cells,
            'td_mse': td_mse,
            'mb_mse': mb_mse
        })
    
    return pd.DataFrame(results)


# Run cell titration for the 3 selected perturbations
titration_results = {}

for pert in selected_perturbations:
    try:
        results_df = cell_titration_simplified(adata, pert, n_points=20, random_seed=42)
        titration_results[pert] = results_df
    except Exception as e:
        print(f"Error processing {pert}: {e}")
        titration_results[pert] = None

# Create 1x3 subplot for cell titration
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# First pass: collect all MSE values to determine global y-axis limits
all_mse_values = []
for perturbation in selected_perturbations:
    if titration_results[perturbation] is not None:
        results_df = titration_results[perturbation]
        all_mse_values.extend(results_df['td_mse'].values)
        all_mse_values.append(results_df['mb_mse'].iloc[0])

# Determine global y-axis limits
if all_mse_values:
    y_min = min(all_mse_values)
    y_max = max(all_mse_values)
    # Add some padding in log space
    y_lims = [y_min * 0.8, y_max * 1.2]
else:
    y_lims = None

for i, (perturbation, label) in enumerate(zip(selected_perturbations, selected_labels)):
    ax = axes[i]
    
    if titration_results[perturbation] is not None:
        results_df = titration_results[perturbation]
        
        # Plot Tech. Dup. MSE (blue line)
        ax.plot(results_df['n_cells'], results_df['td_mse'], 'o-', 
                color='darkblue', label='Tech. Dup.', linewidth=2, markersize=4)
        
        # Plot Mean Baseline MSE (black horizontal line)
        ax.axhline(y=results_df['mb_mse'].iloc[0], color='black', 
                   linestyle='--', linewidth=2, alpha=0.7, label='Mean Baseline')
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        
        # Set shared y-axis limits
        if y_lims is not None:
            ax.set_ylim(y_lims)
        
        # Labels and title
        ax.set_xlabel('Number of Cells', fontsize=18)
        if i == 0:
            ax.set_ylabel('MSE (log scale)', fontsize=18)
        
        # Get DEG count for title
        n_degs = deg_counts[perturbation]
        ax.set_title(f'{perturbation.split("_")[1]}\n{n_degs} DEGs', 
                     fontsize=18, fontweight='bold')
        
        # Grid and styling
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(fontsize=14, loc='upper right')
    else:
        ax.text(0.5, 0.5, f'Error\n{perturbation}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'{perturbation}\nError', fontsize=18)

plt.tight_layout()
plt.show()

# %% Wide p-value plot with shading for ~30 DEG perturbation

# Find perturbation with approximately 30 DEGs
target_deg_pert = deg_counts.index[(deg_counts - 30).abs().argmin()]
target_deg_count = deg_counts[target_deg_pert]
print(f"Selected perturbation: {target_deg_pert} with {target_deg_count} DEGs")

# Get adjusted p-values and gene names
pvals = adata.uns['pvals_adj_df_dict_gt'][target_deg_pert]
gene_names = adata.uns['names_df_dict_gt'][target_deg_pert]

# Convert to Series and sort
pvals_series = pd.Series(pvals, index=gene_names)
pvals_series_sorted = pvals_series.sort_values(ascending=False)

# Create ranking and 1 - p-value
rank = np.arange(len(pvals_series_sorted))
one_minus_pval = 1 - pvals_series_sorted.values

# Create wide figure
fig, ax = plt.subplots(figsize=(20, 4))

# Plot line
ax.plot(rank, one_minus_pval, color='darkblue', linewidth=2, alpha=0.9)

# Fill area under curve with blue
ax.fill_between(rank, 0, one_minus_pval, color='blue', alpha=0.3)

# Labels and title
ax.set_xlabel('Rank', fontsize=18)
ax.set_ylabel('1 - DEG pval (adjusted)', fontsize=18)
ax.set_title(f'DEGs: {target_deg_count}', fontsize=20, fontweight='bold')

# Set axis limits
ax.set_xlim([0, len(pvals_series_sorted)])
ax.set_ylim([0, 1.05])

# Styling
sns.despine(ax=ax)

# Make x-axis less crowded
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

plt.tight_layout()
plt.show()

# %% 

