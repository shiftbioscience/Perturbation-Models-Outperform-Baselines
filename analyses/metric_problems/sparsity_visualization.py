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



# %% Read data
path_dict = {
    "replogle22k562gwps": Path("./data/replogle22k562gwps/replogle22k562gwps_processed_complete.h5ad"),
    "replogle22rpe1": Path("./data/replogle22rpe1/replogle22rpe1_processed_complete.h5ad"),
    "replogle22k562": Path("./data/replogle22k562/replogle22k562_processed_complete.h5ad")
}

dataset_name = "replogle22k562"

data_path = path_dict[dataset_name]
print(f"Loading data from: {data_path}")
print("-" * 80)

# Load the AnnData object
adata = sc.read_h5ad(data_path)
print(f"✓ Data loaded successfully!")
print("=" * 80)


# %% Define auxiliary functions for the analysis

def compute_binary_combinations_df(sparsity_gt, sparsity_td, sparsity_mb):
    """
    Compute binary combinations for all perturbations.
    
    Creates a DataFrame where each row is a perturbation and each column represents
    the count of genes in each binary combination (gt, td, mb).
    
    Args:
        sparsity_gt: DataFrame with ground truth sparsity (True = not expressed)
        sparsity_td: DataFrame with technical duplicate sparsity (True = not expressed)  
        sparsity_mb: DataFrame with mean baseline sparsity (True = not expressed)
        
    Returns:
        DataFrame with perturbations as rows and 8 binary combination columns
    """
    # Define the 8 binary combinations in order: (gt, td, mb)
    combinations = [
        (True, True, True),   # 0,0,0 - all not expressed
        (True, True, False),  # 0,0,1 - gt/td not expressed, mb expressed
        (True, False, True),  # 0,1,0 - gt/mb not expressed, td expressed
        (True, False, False), # 0,1,1 - gt not expressed, td/mb expressed
        (False, True, True),  # 1,0,0 - td/mb not expressed, gt expressed
        (False, True, False), # 1,0,1 - td not expressed, gt/mb expressed
        (False, False, True), # 1,1,0 - mb not expressed, gt/td expressed
        (False, False, False) # 1,1,1 - all expressed
    ]
    
    # Create column names
    column_names = [
        'gt0_td0_mb0', 'gt0_td0_mb1', 'gt0_td1_mb0', 'gt0_td1_mb1',
        'gt1_td0_mb0', 'gt1_td0_mb1', 'gt1_td1_mb0', 'gt1_td1_mb1'
    ]
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=sparsity_gt.index, columns=column_names)
    
    # Compute counts for each perturbation
    for pert in tqdm(sparsity_gt.index, desc="Computing binary combinations"):
        gt_row = sparsity_gt.loc[pert]
        td_row = sparsity_td.loc[pert]
        mb_row = sparsity_mb.loc[pert]
        
        # Count genes for each binary combination
        for i, (gt_val, td_val, mb_val) in enumerate(combinations):
            count = ((gt_row == gt_val) & (td_row == td_val) & (mb_row == mb_val)).sum()
            results.loc[pert, column_names[i]] = count
    
    return results


def compute_deg_counts(adata, perturbation_index):
    """
    Compute the number of DEGs for each perturbation.
    
    Args:
        adata: AnnData object containing deg_gene_dict in uns
        perturbation_index: Index of perturbations to compute DEGs for
        
    Returns:
        Series with DEG counts for each perturbation
    """
    deg_counts = pd.Series(index=perturbation_index, dtype=int)
    deg_dict = adata.uns.get('deg_gene_dict', {})
    
    for pert in perturbation_index:
        if pert in deg_dict:
            deg_counts[pert] = len(deg_dict[pert])
        else:
            deg_counts[pert] = 0
    
    return deg_counts


def compute_mse_to_mean(gt_baseline, mean_baseline, perturbation_index):
    """
    Compute MSE between ground truth and mean baseline for each perturbation.
    
    Args:
        gt_baseline: DataFrame with ground truth baseline values
        mean_baseline: Series or DataFrame with mean baseline values
        perturbation_index: Index of perturbations to compute MSE for
        
    Returns:
        Series with MSE values for each perturbation
    """
    mse_values = pd.Series(index=perturbation_index, dtype=float)
    
    # If mean_baseline is a Series (single row), convert to DataFrame format
    if isinstance(mean_baseline, pd.Series):
        mean_baseline_df = pd.DataFrame([mean_baseline.values] * len(perturbation_index),
                                       index=perturbation_index, 
                                       columns=mean_baseline.index)
    else:
        mean_baseline_df = mean_baseline
    
    # Compute MSE for each perturbation
    for pert in perturbation_index:
        if pert in gt_baseline.index and pert in mean_baseline_df.index:
            # Get the values for this perturbation
            gt_values = gt_baseline.loc[pert]
            mean_values = mean_baseline_df.loc[pert]
            
            # Compute MSE
            mse = ((gt_values - mean_values) ** 2).mean()
            mse_values[pert] = mse
        else:
            mse_values[pert] = np.nan
    
    return mse_values


def compute_mse_to_td(gt_baseline, td_baseline, perturbation_index):
    """
    Compute MSE between ground truth (first half) and technical duplicate (second half) for each perturbation.
    
    Args:
        gt_baseline: DataFrame with ground truth baseline values (first half)
        td_baseline: DataFrame with technical duplicate baseline values (second half)
        perturbation_index: Index of perturbations to compute MSE for
        
    Returns:
        Series with MSE values for each perturbation
    """
    mse_values = pd.Series(index=perturbation_index, dtype=float)
    
    # Compute MSE for each perturbation
    for pert in perturbation_index:
        if pert in gt_baseline.index and pert in td_baseline.index:
            # Get the values for this perturbation
            gt_values = gt_baseline.loc[pert]
            td_values = td_baseline.loc[pert]
            
            # Compute MSE
            mse = ((gt_values - td_values) ** 2).mean()
            mse_values[pert] = mse
        else:
            mse_values[pert] = np.nan
    
    return mse_values


def plot_stacked_proportions(df, x_col='size', value_cols=None, title="Proportion", 
                           figsize=(10, 14), colors=None, alpha=0.8):
    """
    Create a stacked bar chart showing proportions for each category,
    with additional subplots showing MSE metrics and number of DEGs.
    
    Args:
        df: DataFrame with data to plot (should include 'num_degs', 'mse_to_mean', 'mse_to_td' columns)
        x_col: Column name to rank by for x-axis (e.g., 'size')
        value_cols: List of column names to stack (if None, uses all numeric columns except special columns)
        title: Title for the plot
        figsize: Tuple for figure size
        colors: List of colors for each category (if None, uses default palette)
        alpha: Transparency level for bars
        
    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # If value_cols not specified, use all numeric columns except special columns
    if value_cols is None:
        exclude_cols = [x_col, 'num_degs', 'mse_to_mean', 'mse_to_td']
        value_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    # Create ranking based on x_col values and break ties with arange
    df_ranked = df.copy()
    df_ranked = df_ranked.sort_values(by=x_col, ascending=False).reset_index(drop=True)
    df_ranked['rank'] = np.arange(1, len(df_ranked) + 1)
    
    # Set up the plot with 4 subplots (shared x-axis)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize, sharex=True, 
                                              gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Define colors if not provided - use divergent palette for better distinction
    if colors is None:
        # Use a custom divergent palette with distinct colors
        if len(value_cols) == 8:
            # Custom palette for 8 categories (binary combinations)
            colors = ['#8B0000', '#FF6347', '#FFA500', '#FFD700',  # Reds to yellows (mb=0)
                      '#4682B4', '#00CED1', '#20B2AA', '#00FA9A']  # Blues to greens (mb=1)
        else:
            # Use seaborn's tab10 or Set1 for other cases
            colors = sns.color_palette("tab10", len(value_cols)) if len(value_cols) <= 10 else sns.color_palette("Set1", len(value_cols))
    
    # Calculate proportions for each row
    df_prop = df_ranked.copy()
    for col in value_cols:
        df_prop[col] = df_ranked[col] / df_ranked[value_cols].sum(axis=1)
    
    # SUBPLOT 1: Create stacked bars using ranking as x-axis
    bottom = np.zeros(len(df_ranked))
    
    for i, col in enumerate(value_cols):
        ax1.bar(df_ranked['rank'], df_prop[col], bottom=bottom, 
                label=col, color=colors[i], alpha=alpha, width=1.0)
        bottom += df_prop[col]
    
    # Customize subplot 1
    ax1.set_ylabel('Proportion')
    ax1.set_title(title)
    ax1.set_ylim(0, 1)
    
    # Add legend
    ax1.legend(title='category', loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    
    # SUBPLOT 2: Plot MSE to mean vs rank (if mse_to_mean column exists)
    if 'mse_to_mean' in df_ranked.columns:
        ax2.bar(df_ranked['rank'], df_ranked['mse_to_mean'], 
                color='#B22222', alpha=0.7, width=1.0)
        ax2.set_ylabel('MSE to Mean Baseline')
        ax2.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)
    else:
        ax2.text(0.5, 0.5, 'No MSE to Mean data available', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # SUBPLOT 3: Plot MSE to TD vs rank (if mse_to_td column exists)
    if 'mse_to_td' in df_ranked.columns:
        ax3.bar(df_ranked['rank'], df_ranked['mse_to_td'], 
                color='#4169E1', alpha=0.7, width=1.0)
        ax3.set_ylabel('MSE to Technical Duplicate')
        ax3.grid(True, alpha=0.3)
        ax3.set_axisbelow(True)
    else:
        ax3.text(0.5, 0.5, 'No MSE to TD data available', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # SUBPLOT 4: Plot number of DEGs vs rank (if num_degs column exists)
    if 'num_degs' in df_ranked.columns:
        # Apply log10(x+1) transformation to DEG counts
        log_degs = np.log10(df_ranked['num_degs'] + 1)
        ax4.bar(df_ranked['rank'], log_degs, 
                color='#2E8B57', alpha=0.7, width=1.0)
        ax4.set_ylabel('log10(DEGs + 1)')
        ax4.set_xlabel(f'Rank by {x_col}')
        ax4.grid(True, alpha=0.3)
        ax4.set_axisbelow(True)
    else:
        ax4.text(0.5, 0.5, 'No DEG data available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_xlabel(f'Rank by {x_col}')
    
    # Set x-axis limits for all subplots (0 to number of perturbations)
    ax1.set_xlim(0, len(df_ranked) + 1)
    ax2.set_xlim(0, len(df_ranked) + 1)
    ax3.set_xlim(0, len(df_ranked) + 1)
    ax4.set_xlim(0, len(df_ranked) + 1)
    
    # Despine all subplots
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    sns.despine(ax=ax3)
    sns.despine(ax=ax4)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig




# %% Compute bool sparsities for a single fold

fold = 'fold_1'

sparsity_gt = adata.uns['technical_duplicate_first_half_baseline'] == 0
sparsity_td = adata.uns['technical_duplicate_second_half_baseline'] == 0
# Compute sparsity for the mean baseline (repeat for each condition)
sparsity_mb = adata.uns[f'split_{fold}_mean_baseline'] == 0
sparsity_mb = pd.concat([sparsity_mb]*len(sparsity_gt))
sparsity_mb.index = sparsity_gt.index

# %% Compute binary combinations for all perturbations
print("Computing binary combinations for all perturbations...")
binary_combinations_df = compute_binary_combinations_df(sparsity_gt, sparsity_td, sparsity_mb)

print(f"Binary combinations computed for {len(binary_combinations_df)} perturbations")
print(f"Columns: {list(binary_combinations_df.columns)}")
print(f"\\nFirst few rows:")
print(binary_combinations_df.head())

# %% Add DEG counts to the binary combinations DataFrame
deg_counts = compute_deg_counts(adata, binary_combinations_df.index)
binary_combinations_df['num_degs'] = deg_counts
print(f"\\nAdded DEG counts. New columns: {list(binary_combinations_df.columns)}")
print(f"DEG count statistics:")
print(f"  Min DEGs: {binary_combinations_df['num_degs'].min()}")
print(f"  Max DEGs: {binary_combinations_df['num_degs'].max()}")
print(f"  Mean DEGs: {binary_combinations_df['num_degs'].mean():.2f}")

# %% Add MSE between GT and mean baseline
gt_baseline = adata.uns['technical_duplicate_first_half_baseline']
mean_baseline = adata.uns[f'split_{fold}_mean_baseline']
mean_baseline = pd.concat([mean_baseline]*len(gt_baseline))
mean_baseline.index = gt_baseline.index
mse_to_mean = compute_mse_to_mean(gt_baseline, mean_baseline, binary_combinations_df.index)
binary_combinations_df['mse_to_mean'] = mse_to_mean
print(f"\\nAdded MSE to mean. New columns: {list(binary_combinations_df.columns)}")
print(f"MSE to mean statistics:")
print(f"  Min MSE: {binary_combinations_df['mse_to_mean'].min():.4f}")
print(f"  Max MSE: {binary_combinations_df['mse_to_mean'].max():.4f}")
print(f"  Mean MSE: {binary_combinations_df['mse_to_mean'].mean():.4f}")

# %% Add MSE between GT and technical duplicate
td_baseline = adata.uns['technical_duplicate_second_half_baseline']
mse_to_td = compute_mse_to_td(gt_baseline, td_baseline, binary_combinations_df.index)
binary_combinations_df['mse_to_td'] = mse_to_td
print(f"\\nAdded MSE to TD. New columns: {list(binary_combinations_df.columns)}")
print(f"MSE to TD statistics:")
print(f"  Min MSE: {binary_combinations_df['mse_to_td'].min():.4f}")
print(f"  Max MSE: {binary_combinations_df['mse_to_td'].max():.4f}")
print(f"  Mean MSE: {binary_combinations_df['mse_to_td'].mean():.4f}")

# %% Create stacked proportion plot

fig = plot_stacked_proportions(
    df=binary_combinations_df, 
    x_col='gt1_td1_mb1',  # or any column you want on x-axis
    value_cols=['gt0_td0_mb0', 'gt0_td1_mb0', 'gt1_td0_mb0', 'gt1_td1_mb0',
                 'gt0_td0_mb1', 'gt0_td1_mb1', 'gt1_td0_mb1',  'gt1_td1_mb1'],
    title=f"Sparsity Visualization - {dataset_name}",
    figsize=(9, 15)
)
plt.show()
