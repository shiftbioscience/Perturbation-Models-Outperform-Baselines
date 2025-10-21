# %% Imports

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Import wmse function for weighted MSE calculations
from cellsimbench.core.data_manager import wmse

# Set high DPI for better resolution in interactive mode -- [Coding Agent]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200


# %% Read data

path_dict = {
    "replogle22k562gwps": Path("/home/gabriel/CellSimBench/data/replogle22k562gwps/replogle22k562gwps_processed_complete.h5ad"),  # 256 cells
    "replogle22rpe1": Path("/home/gabriel/CellSimBench/data/replogle22rpe1/replogle22rpe1_processed_complete.h5ad"),  # 128 cells
    "replogle22k562": Path("/home/gabriel/CellSimBench/data/replogle22k562/replogle22k562_processed_complete.h5ad")  # 128 cells
}

dataset_name = "replogle22k562gwps"  # Change this parameter as needed

data_path = path_dict[dataset_name]
print(f"Loading data from: {data_path}")
print("-" * 80)

# Load the AnnData object
adata = sc.read_h5ad(data_path)
print(f"✓ Data loaded successfully!")
print("=" * 80)


# %% Get DEG counts for all perturbations

# Get DEG dictionary from uns
deg_gene_dict = adata.uns.get('deg_gene_dict', {})

if not deg_gene_dict:
    print(f"WARNING: No deg_gene_dict found for {dataset_name}")
    deg_counts = pd.Series(dtype=int)
else:
    # Create Series with perturbation names as index and DEG counts as values
    # Note: deg_gene_dict uses the processed perturbation names from uns
    deg_counts = pd.Series({
        pert_name: len(deg_genes) 
        for pert_name, deg_genes in deg_gene_dict.items()
        if 'control' not in pert_name.lower()  # Skip control perturbations
    })
    
    # Sort by DEG count (descending)
    deg_counts = deg_counts.sort_values(ascending=False)
    
    print(f"Found {len(deg_counts)} perturbations (excluding controls)")
    print(f"Perturbation with most DEGs: {deg_counts.index[0]}")
    print(f"Number of DEGs: {deg_counts.iloc[0]}")
    print(f"DEG count range: {deg_counts.min()} to {deg_counts.max()}")
    print("\nTop 10 perturbations by DEG count:")
    print(deg_counts.head(10))


# %% Plot histogram of DEG counts

if len(deg_counts) > 0:
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Apply log10(x+1) transformation
    log_deg_counts = np.log10(deg_counts + 1)
    
    # Create histogram
    ax.hist(log_deg_counts, bins=30, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Customize plot
    ax.set_xlabel('log10(Number of DEGs + 1)', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title(f'Distribution of DEG Counts - {dataset_name}', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Despine
    sns.despine(ax=ax)
    
    # Add comprehensive statistics text
    stats_text = (f'n = {len(deg_counts)}\n'
                 f'Mean: {deg_counts.mean():.1f}\n'
                 f'Median: {deg_counts.median():.1f}\n'
                 f'Std: {deg_counts.std():.1f}\n'
                 f'Min: {deg_counts.min()}\n'
                 f'Max: {deg_counts.max()}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    plt.show()
else:
    print("No perturbations found to plot.")

# %% Plot absolute scores vs rank for strongest perturbation and single DEG perturbation

if len(deg_counts) > 0:
    # Get the strongest perturbation (most DEGs)
    strongest_pert = deg_counts.index[0]  # First in sorted list (descending)
    
    # Find a perturbation with exactly 1 DEG
    single_deg_pert = None
    for pert_name in deg_counts.index:
        if deg_counts[pert_name] == 1:
            single_deg_pert = pert_name
            break
    
    if single_deg_pert is None:
        # If no perturbation has exactly 1 DEG, use the one with the fewest DEGs
        single_deg_pert = deg_counts.index[-1]  # Last in sorted list (ascending)
        print(f"No perturbation with exactly 1 DEG found, using {single_deg_pert} with {deg_counts[single_deg_pert]} DEGs")
    
    print(f"Strongest perturbation: {strongest_pert} ({deg_counts[strongest_pert]} DEGs)")
    print(f"Single/Low DEG perturbation: {single_deg_pert} ({deg_counts[single_deg_pert]} DEGs)")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    
    # Plot for strongest perturbation
    max_score_global = 0  # Track global maximum for consistent y-axis scaling
    
    if strongest_pert in adata.uns['scores_df_dict']:
        scores_strong = adata.uns['scores_df_dict'][strongest_pert]
        abs_scores_strong = np.abs(scores_strong)
        # Sort by absolute value (biggest first)
        sorted_indices_strong = np.argsort(abs_scores_strong)[::-1]
        sorted_abs_scores_strong = abs_scores_strong[sorted_indices_strong]
        ranks_strong = np.arange(1, len(sorted_abs_scores_strong) + 1)
        max_score_global = max(max_score_global, sorted_abs_scores_strong.max())
        
        # Plot line and fill area under curve
        ax1.plot(ranks_strong, sorted_abs_scores_strong, 'o-', color='darkred', 
                linewidth=2, markersize=3, alpha=0.8)
        ax1.fill_between(ranks_strong, sorted_abs_scores_strong, 0, 
                        color='darkred', alpha=0.3)
        ax1.set_xlabel('Rank', fontsize=12)
        ax1.set_ylabel('abs(t-score)', fontsize=12)
        ax1.set_title(f'Strongest Perturbation\n{strongest_pert}\nDEGs: {deg_counts[strongest_pert]}', 
                     fontsize=12, fontweight='bold')
        ax1.set_axisbelow(True)
        sns.despine(ax=ax1)
    else:
        ax1.text(0.5, 0.5, f'Scores not found\nfor {strongest_pert}', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'Strongest Perturbation\n{strongest_pert}\nDEGs: {deg_counts[strongest_pert]}', 
                     fontsize=12, fontweight='bold')
    
    # Plot for single/low DEG perturbation
    if single_deg_pert in adata.uns['scores_df_dict']:
        scores_single = adata.uns['scores_df_dict'][single_deg_pert]
        abs_scores_single = np.abs(scores_single)
        # Sort by absolute value (biggest first)
        sorted_indices_single = np.argsort(abs_scores_single)[::-1]
        sorted_abs_scores_single = abs_scores_single[sorted_indices_single]
        ranks_single = np.arange(1, len(sorted_abs_scores_single) + 1)
        max_score_global = max(max_score_global, sorted_abs_scores_single.max())
        
        # Plot line and fill area under curve
        ax2.plot(ranks_single, sorted_abs_scores_single, 'o-', color='darkblue', 
                linewidth=2, markersize=3, alpha=0.8)
        ax2.fill_between(ranks_single, sorted_abs_scores_single, 0, 
                        color='darkblue', alpha=0.3)
        ax2.set_xlabel('Rank', fontsize=12)
        ax2.set_ylabel('abs(t-score)', fontsize=12)
        ax2.set_title(f'Low DEG Perturbation\n{single_deg_pert}\nDEGs: {deg_counts[single_deg_pert]}', 
                     fontsize=12, fontweight='bold')
        ax2.set_axisbelow(True)
        sns.despine(ax=ax2)
    else:
        ax2.text(0.5, 0.5, f'Scores not found\nfor {single_deg_pert}', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'Low DEG Perturbation\n{single_deg_pert}\nDEGs: {deg_counts[single_deg_pert]}', 
                     fontsize=12, fontweight='bold')
    
    # Set consistent y-axis limits for both subplots (0 to max score with some padding)
    if max_score_global > 0:
        y_max = max_score_global * 1.05  # Add 5% padding
        ax1.set_ylim(0, y_max)
        ax2.set_ylim(0, y_max)
    
    # Set x-axis limits from 0 to number of genes for both subplots
    n_genes = len(adata.var_names)
    ax1.set_xlim(0, n_genes)
    ax2.set_xlim(0, n_genes)
    
    plt.tight_layout()
    plt.show()
else:
    print("No perturbations found for score ranking plots.")

# %% Plot 1-pval vs rank for perturbation with 10-20 DEGs

if len(deg_counts) > 0:
    # Find a perturbation with 10-20 DEGs
    target_pert = None
    for pert_name in deg_counts.index:
        if 10 <= deg_counts[pert_name] <= 20:
            target_pert = pert_name
            break
    
    if target_pert is None:
        # If no perturbation has 10-20 DEGs, use the closest one
        closest_pert = None
        min_diff = float('inf')
        for pert_name in deg_counts.index:
            diff = abs(deg_counts[pert_name] - 15)  # Target around 15 DEGs
            if diff < min_diff:
                min_diff = diff
                closest_pert = pert_name
        target_pert = closest_pert
        print(f"No perturbation with 10-20 DEGs found, using {target_pert} with {deg_counts[target_pert]} DEGs")
    
    print(f"Target perturbation: {target_pert} ({deg_counts[target_pert]} DEGs)")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    
    if target_pert in adata.uns['pvals_df_dict']:
        pvals = adata.uns['pvals_df_dict'][target_pert]
        one_minus_pvals = 1 - pvals
        # Sort by 1-pval value (biggest first - most significant first)
        sorted_indices = np.argsort(one_minus_pvals)[::-1]
        sorted_one_minus_pvals = one_minus_pvals[sorted_indices]
        ranks = np.arange(1, len(sorted_one_minus_pvals) + 1)
        
        # Plot line and fill area under curve
        ax.plot(ranks, sorted_one_minus_pvals, 'o-', color='darkgreen', 
                linewidth=2, markersize=3, alpha=0.8)
        ax.fill_between(ranks, sorted_one_minus_pvals, 0, 
                        color='darkgreen', alpha=0.3)
        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel('1 - pval', fontsize=12)
        ax.set_title(f'1-pval vs Rank\n{target_pert}\nDEGs: {deg_counts[target_pert]}', 
                     fontsize=14, fontweight='bold')
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
        
        # Set axis limits
        ax.set_ylim(0, 1)  # 1-pval ranges from 0 to 1
        n_genes = len(adata.var_names)
        ax.set_xlim(0, n_genes)
    else:
        ax.text(0.5, 0.5, f'P-values not found\nfor {target_pert}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'1-pval vs Rank\n{target_pert}\nDEGs: {deg_counts[target_pert]}', 
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
else:
    print("No perturbations found for p-value ranking plot.")


# %% Define helper function to compute DEG weights from scores

def compute_deg_weights_from_scores(scores, gene_names, all_gene_names):
    """
    Compute DEG weights from scores following data_manager logic.
    
    Parameters:
    -----------
    scores : array-like
        DEG scores for the perturbation
    gene_names : array-like
        Gene names corresponding to the scores
    all_gene_names : array-like
        All gene names in the dataset (e.g., adata.var_names)
    
    Returns:
    --------
    np.ndarray: Weights array aligned with all_gene_names
    """
    # Convert to absolute scores
    abs_scores = np.abs(scores)
    
    # Min-max normalization
    min_val = np.min(abs_scores)
    max_val = np.max(abs_scores)
    
    if max_val > min_val:
        normalized_weights = (abs_scores - min_val) / (max_val - min_val)
    else:
        # Handle case where all scores are the same
        normalized_weights = np.zeros_like(abs_scores)
    
    # Handle NaNs
    normalized_weights = np.nan_to_num(normalized_weights, nan=0.0)
    
    # Square the weights for stronger emphasis
    normalized_weights = np.square(normalized_weights)
    
    # Create DataFrame and handle duplicates by taking the maximum weight for each gene
    weights_df = pd.DataFrame({
        'gene': gene_names,
        'weight': normalized_weights
    })
    
    # Group by gene and take the maximum weight in case of duplicates
    weights_aggregated = weights_df.groupby('gene')['weight'].max()
    
    # Reindex to match all_gene_names
    weights = weights_aggregated.reindex(all_gene_names, fill_value=0.0)
    
    return weights.values

# %% Define cell titration function

def cell_titration_experiment(adata, perturbation_name, n_points=20, random_seed=42):
    """
    Perform cell titration experiment for a given perturbation.
    
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
        DataFrame with results containing all metrics for each cell count
    """
    np.random.seed(random_seed)
    
    # Find cells that match this perturbation in condition column
    pert_mask = adata.obs['condition'] == perturbation_name.split('_')[1]
    pert_cells = adata[pert_mask]
    
    if pert_cells.n_obs == 0:
        raise ValueError(f"No cells found for perturbation '{perturbation_name}'")
    
    print(f"Found {pert_cells.n_obs} cells for perturbation '{perturbation_name}'")
    
    # 2. Split perturbation cells randomly into GT and TD groups
    n_pert_cells = pert_cells.n_obs
    gt_indices = np.random.choice(n_pert_cells, size=n_pert_cells//2, replace=False)
    td_indices = np.setdiff1d(np.arange(n_pert_cells), gt_indices)
    
    gt_cells = pert_cells[gt_indices]
    td_cells = pert_cells[td_indices]
    
    print(f"Split into GT: {len(gt_indices)} cells, TD: {len(td_indices)} cells")
    
    # 3. Get ground truth (average of GT cells)
    gt_profile = gt_cells.X.mean(axis=0).A1 if hasattr(gt_cells.X, 'A1') else gt_cells.X.mean(axis=0)
    
    # 4. Get mean baseline (already stored in adata)
    mean_baseline = adata.uns['split_fold_0_mean_baseline'].iloc[0]
    
    # 5. Compute DEG weights for this perturbation -- [Coding Agent]
    try:
        scores = adata.uns['scores_df_dict'][perturbation_name]
        gene_names = adata.uns['names_df_dict'][perturbation_name]
        weights = compute_deg_weights_from_scores(scores, gene_names, adata.var_names)
        print(f"Computed DEG weights: min={weights.min():.4f}, max={weights.max():.4f}, non-zero={np.sum(weights > 0)}")
    except KeyError as e:
        print(f"Warning: Could not find scores for {perturbation_name}, using uniform weights")
        weights = np.ones(len(adata.var_names))
    
    # Get control cells and non-control non-perturbation cells (using condition column)
    control_mask = adata.obs['condition'].str.contains('control', case=False, na=False)
    non_control_non_pert_mask = ~control_mask & (adata.obs['condition'] != perturbation_name.split('_')[1])
    
    control_cells = adata[control_mask]
    smb_cells = adata[non_control_non_pert_mask]  # SMB: Sparse Mean Baseline (previously 'other')
    
    # Compute global control profile (CTL) for reference -- [Coding Agent]
    global_control_profile = control_cells.X.mean(axis=0).A1 if hasattr(control_cells.X, 'A1') else control_cells.X.mean(axis=0)
    
    print(f"SCTL cells: {control_cells.n_obs}, SMB cells: {smb_cells.n_obs}")
    
    # 5. Define linear space and perform titration
    max_cells = min(td_cells.n_obs, control_cells.n_obs, smb_cells.n_obs)
    cell_numbers = np.linspace(1, max_cells, n_points, dtype=int)
    
    # Compute reference delta using global control profile instead of MB -- [Coding Agent]
    gt_minus_ctl = gt_profile - global_control_profile
    
    # Compute delta for MB relative to CTL for pearson comparison -- [Coding Agent]
    mb_minus_ctl = mean_baseline.values - global_control_profile
    
    # Compute MSE between GT and global mean baseline (constant for all points)
    global_mb_mse = np.mean(np.square(gt_profile - mean_baseline.values))
    
    # Compute MSE between GT and global control (CTL) - constant for all points -- [Coding Agent]
    global_ctl_mse = np.mean(np.square(gt_profile - global_control_profile))
    
    # Compute WMSE between GT and global baselines (constant for all points) -- [Coding Agent]
    global_mb_wmse = wmse(gt_profile, mean_baseline.values, weights)
    global_ctl_wmse = wmse(gt_profile, global_control_profile, weights)
    
    # Compute Pearson delta for MB relative to CTL reference -- [Coding Agent]
    mb_pearson_delta, _ = pearsonr(np.array(gt_minus_ctl).squeeze(), np.array(mb_minus_ctl).squeeze())
    
    results = []
    
    print(f"Performing titration with {n_points} points, max cells: {max_cells}")
    
    for n_cells in tqdm(cell_numbers, desc="Cell titration"):
        # a. Sample n_cells from TD group
        td_sample_indices = np.random.choice(td_cells.n_obs, size=n_cells, replace=False)
        td_sample = td_cells[td_sample_indices]
        td_profile = td_sample.X.mean(axis=0).A1 if hasattr(td_sample.X, 'A1') else td_sample.X.mean(axis=0)
        td_mse = np.mean(np.square(gt_profile - td_profile))
        td_wmse = wmse(gt_profile, td_profile, weights)  # Added WMSE calculation -- [Coding Agent]
        td_delta = td_profile - global_control_profile  # Using CTL as reference -- [Coding Agent]
        td_pearson, _ = pearsonr(np.array(gt_minus_ctl).squeeze(), np.array(td_delta).squeeze())
        
        # b. Sample n_cells from SMB cells (non-control, non-perturbation)
        smb_sample_indices = np.random.choice(smb_cells.n_obs, size=n_cells, replace=False)
        smb_sample = smb_cells[smb_sample_indices]
        smb_profile = smb_sample.X.mean(axis=0).A1 if hasattr(smb_sample.X, 'A1') else smb_sample.X.mean(axis=0)
        smb_mse = np.mean(np.square(gt_profile - smb_profile))
        smb_wmse = wmse(gt_profile, smb_profile, weights)  # Added WMSE calculation -- [Coding Agent]
        smb_delta = smb_profile - global_control_profile  # Using CTL as reference -- [Coding Agent]
        smb_pearson, _ = pearsonr(np.array(gt_minus_ctl).squeeze(), np.array(smb_delta).squeeze())
        
        # c. Sample n_cells from SCTL cells (sparse control)
        sctl_sample_indices = np.random.choice(control_cells.n_obs, size=n_cells, replace=False)
        sctl_sample = control_cells[sctl_sample_indices]
        sctl_profile = sctl_sample.X.mean(axis=0).A1 if hasattr(sctl_sample.X, 'A1') else sctl_sample.X.mean(axis=0)
        sctl_mse = np.mean(np.square(gt_profile - sctl_profile))
        sctl_wmse = wmse(gt_profile, sctl_profile, weights)  # Added WMSE calculation -- [Coding Agent]
        sctl_delta = sctl_profile - global_control_profile  # Using CTL as reference -- [Coding Agent]
        sctl_pearson, _ = pearsonr(np.array(gt_minus_ctl).squeeze(), np.array(sctl_delta).squeeze())
        
        results.append({
            'n_cells': n_cells,
            'td_mse': td_mse,
            'smb_mse': smb_mse,  # Renamed from other_mse -- [Coding Agent]
            'sctl_mse': sctl_mse,  # Renamed from control_mse -- [Coding Agent]
            'td_wmse': td_wmse,  # Added WMSE calculations -- [Coding Agent]
            'smb_wmse': smb_wmse,  # Added WMSE calculations -- [Coding Agent]
            'sctl_wmse': sctl_wmse,  # Added WMSE calculations -- [Coding Agent]
            'td_pearson': td_pearson,
            'smb_pearson': smb_pearson,  # Renamed from other_pearson -- [Coding Agent]
            'sctl_pearson': sctl_pearson,  # Renamed from control_pearson -- [Coding Agent]
            'mb_mse': global_mb_mse,
            'ctl_mse': global_ctl_mse,  # Added CTL MSE -- [Coding Agent]
            'mb_wmse': global_mb_wmse,  # Added global WMSE -- [Coding Agent]
            'ctl_wmse': global_ctl_wmse,  # Added global WMSE -- [Coding Agent]
            'mb_pearson': mb_pearson_delta  # Added MB pearson delta -- [Coding Agent]
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).set_index('n_cells')
    
    print(f"\nGlobal Mean Baseline MSE: {global_mb_mse:.4f}, WMSE: {global_mb_wmse:.4f}")
    print(f"Global Control (CTL) MSE: {global_ctl_mse:.4f}, WMSE: {global_ctl_wmse:.4f}")
    print(f"MB Pearson Delta (relative to CTL): {mb_pearson_delta:.4f}")
    print(f"Results DataFrame shape: {results_df.shape}")
    
    return results_df



# %% Plot 20 perturbations with equal spacing by DEG count

# Select 20 perturbations with equal spacing by DEG count
n_perturbations = 20
if len(deg_counts) >= n_perturbations:
    # Create equal spacing indices
    indices = np.linspace(0, len(deg_counts) - 1, n_perturbations, dtype=int)
    selected_perturbations = deg_counts.index[indices]
else:
    selected_perturbations = deg_counts.index

print(f"Selected perturbations: {selected_perturbations.tolist()}")
print(f"DEG counts: {deg_counts[selected_perturbations].tolist()}")

# First, compute all results for all perturbations
print("Computing titration experiments for all perturbations...")
all_results = {}

for i, pert_name in enumerate(selected_perturbations):
    print(f"\nProcessing {i+1}/{len(selected_perturbations)}: {pert_name}")
    
    try:
        # Run titration experiment
        results_df = cell_titration_experiment(adata, perturbation_name=pert_name, n_points=20, random_seed=42)
        all_results[pert_name] = results_df
        
    except Exception as e:
        print(f"  Error processing {pert_name}: {e}")
        all_results[pert_name] = None

# Create subplots for MSE (all 20 perturbations)
n_cols = 5
n_rows = 4
fig_mse, axes_mse = plt.subplots(n_rows, n_cols, figsize=(25, 14))  # 16:9 PPT aspect ratio -- [Coding Agent]
axes_mse = axes_mse.flatten()

for i, pert_name in enumerate(selected_perturbations):
    if i >= len(axes_mse):
        break
        
    # Get number of DEGs
    deg_gene_dict = adata.uns.get('deg_gene_dict', {})
    n_degs = len(deg_gene_dict.get(pert_name, []))
    
    if all_results[pert_name] is not None:
        results_df = all_results[pert_name]
        
        # Plot MSE
        ax = axes_mse[i]
        ax.plot(results_df.index, results_df['td_mse'], 'o-', color='darkblue', 
                label='TD', linewidth=1.5, markersize=2)
        ax.plot(results_df.index, results_df['smb_mse'], 'o-', color='red', 
                label='SMB', linewidth=1.5, markersize=2)
        ax.plot(results_df.index, results_df['sctl_mse'], 'o-', color='green', 
                label='SCTL', linewidth=1.5, markersize=2)
        
        # Add horizontal dashed line for global mean baseline
        ax.axhline(y=results_df['mb_mse'].iloc[0], color='black', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='MB')
        
        # Add horizontal dashed line for global control (CTL) -- [Coding Agent]
        ax.axhline(y=results_df['ctl_mse'].iloc[0], color='green', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='CTL')
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        
        # Customize subplot
        ax.set_xlabel('Cells', fontsize=8)
        ax.set_ylabel('MSE (log scale)', fontsize=8)
        ax.set_title(f'{pert_name}\nDEGs: {n_degs}', fontsize=10, fontweight='bold')  # Increased fontsize -- [Coding Agent]
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Legend will be added at figure level, not subplot level -- [Coding Agent]
        
        sns.despine(ax=ax)
    else:
        # Clear the subplot if there was an error
        axes_mse[i].clear()
        axes_mse[i].text(0.5, 0.5, f'Error\n{pert_name}', ha='center', va='center', transform=axes_mse[i].transAxes)
        axes_mse[i].set_title(f'{pert_name}\nError', fontsize=8)

# Hide unused subplots for MSE
for i in range(len(selected_perturbations), len(axes_mse)):
    axes_mse[i].set_visible(False)

# Add single legend for entire figure on the right -- [Coding Agent]
handles, labels = axes_mse[0].get_legend_handles_labels()
fig_mse.legend(handles, labels, loc='center left', bbox_to_anchor=(0.99, 0.5), fontsize=16, framealpha=0.9)  # Further increased fontsize -- [Coding Agent]

plt.tight_layout(rect=[0, 0, 0.94, 1])  # Leave more space for legend on right -- [Coding Agent]
plt.show()

# Create subplots for Pearson Delta (all 20 perturbations)
fig_pearson, axes_pearson = plt.subplots(n_rows, n_cols, figsize=(24, 13.5))  # 16:9 PPT aspect ratio -- [Coding Agent]
axes_pearson = axes_pearson.flatten()

for i, pert_name in enumerate(selected_perturbations):
    if i >= len(axes_pearson):
        break
        
    # Get number of DEGs
    deg_gene_dict = adata.uns.get('deg_gene_dict', {})
    n_degs = len(deg_gene_dict.get(pert_name, []))
    
    if all_results[pert_name] is not None:
        results_df = all_results[pert_name]
        
        # Plot Pearson Delta
        ax = axes_pearson[i]
        ax.plot(results_df.index, results_df['td_pearson'], 'o-', color='darkblue', 
                label='TD', linewidth=1.5, markersize=2)
        ax.plot(results_df.index, results_df['smb_pearson'], 'o-', color='red', 
                label='SMB', linewidth=1.5, markersize=2)
        ax.plot(results_df.index, results_df['sctl_pearson'], 'o-', color='green', 
                label='SCTL', linewidth=1.5, markersize=2)
        
        # Add horizontal dashed line for MB pearson delta -- [Coding Agent]
        ax.axhline(y=results_df['mb_pearson'].iloc[0], color='black', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='MB')
        
        # Customize subplot
        ax.set_xlabel('Cells', fontsize=8)
        ax.set_ylabel('Pearson Delta (relative to CTL)', fontsize=8)
        ax.set_title(f'{pert_name}\nDEGs: {n_degs}', fontsize=10, fontweight='bold')  # Increased fontsize -- [Coding Agent]
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Legend will be added at figure level, not subplot level -- [Coding Agent]
        
        sns.despine(ax=ax)
    else:
        # Clear the subplot if there was an error
        axes_pearson[i].clear()
        axes_pearson[i].text(0.5, 0.5, f'Error\n{pert_name}', ha='center', va='center', transform=axes_pearson[i].transAxes)
        axes_pearson[i].set_title(f'{pert_name}\nError', fontsize=8)

# Hide unused subplots for Pearson
for i in range(len(selected_perturbations), len(axes_pearson)):
    axes_pearson[i].set_visible(False)

# Add single legend for entire figure on the right -- [Coding Agent]
handles, labels = axes_pearson[0].get_legend_handles_labels()
fig_pearson.legend(handles, labels, loc='center left', bbox_to_anchor=(0.99, 0.5), fontsize=16, framealpha=0.9)  # Further increased fontsize -- [Coding Agent]

plt.tight_layout(rect=[0, 0, 0.94, 1])  # Leave more space for legend on right -- [Coding Agent]
plt.show()

# Create subplots for WMSE (all 20 perturbations) -- [Coding Agent]
fig_wmse, axes_wmse = plt.subplots(n_rows, n_cols, figsize=(25, 14))  # 16:9 PPT aspect ratio -- [Coding Agent]
axes_wmse = axes_wmse.flatten()

for i, pert_name in enumerate(selected_perturbations):
    if i >= len(axes_wmse):
        break
        
    # Get number of DEGs
    deg_gene_dict = adata.uns.get('deg_gene_dict', {})
    n_degs = len(deg_gene_dict.get(pert_name, []))
    
    if all_results[pert_name] is not None:
        results_df = all_results[pert_name]
        
        # Plot WMSE
        ax = axes_wmse[i]
        ax.plot(results_df.index, results_df['td_wmse'], 'o-', color='darkblue', 
                label='TD', linewidth=1.5, markersize=2)
        ax.plot(results_df.index, results_df['smb_wmse'], 'o-', color='red', 
                label='SMB', linewidth=1.5, markersize=2)
        ax.plot(results_df.index, results_df['sctl_wmse'], 'o-', color='green', 
                label='SCTL', linewidth=1.5, markersize=2)
        
        # Add horizontal dashed line for global mean baseline WMSE
        ax.axhline(y=results_df['mb_wmse'].iloc[0], color='black', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='MB')
        
        # Add horizontal dashed line for global control (CTL) WMSE -- [Coding Agent]
        ax.axhline(y=results_df['ctl_wmse'].iloc[0], color='green', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='CTL')
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        
        # Customize subplot
        ax.set_xlabel('Cells', fontsize=8)
        ax.set_ylabel('WMSE (log scale)', fontsize=8)
        ax.set_title(f'{pert_name}\nDEGs: {n_degs}', fontsize=10, fontweight='bold')  # Increased fontsize -- [Coding Agent]
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Legend will be added at figure level, not subplot level -- [Coding Agent]
        
        sns.despine(ax=ax)
    else:
        # Clear the subplot if there was an error
        axes_wmse[i].clear()
        axes_wmse[i].text(0.5, 0.5, f'Error\n{pert_name}', ha='center', va='center', transform=axes_wmse[i].transAxes)
        axes_wmse[i].set_title(f'{pert_name}\nError', fontsize=8)

# Hide unused subplots for WMSE
for i in range(len(selected_perturbations), len(axes_wmse)):
    axes_wmse[i].set_visible(False)

# Add single legend for entire figure on the right -- [Coding Agent]
handles, labels = axes_wmse[0].get_legend_handles_labels()
fig_wmse.legend(handles, labels, loc='center left', bbox_to_anchor=(0.99, 0.5), fontsize=16, framealpha=0.9)  # Further increased fontsize -- [Coding Agent]

plt.tight_layout(rect=[0, 0, 0.94, 1])  # Leave more space for legend on right -- [Coding Agent]
plt.show()

# %% SMB-focused experiment with log scale

def smb_titration_experiment(adata, perturbation_name, n_points=30, random_seed=42):
    """
    SMB-focused titration experiment with log scale from 1 to 10k cells.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing all data
    perturbation_name : str
        Name of the perturbation to analyze
    n_points : int
        Number of points in the log space (default: 30)
    random_seed : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    DataFrame with results
    """
    np.random.seed(random_seed)
    
    # Find cells that match this perturbation in condition column
    pert_mask = adata.obs['condition'] == perturbation_name.split('_')[1]
    pert_cells = adata[pert_mask]
    
    if pert_cells.n_obs == 0:
        raise ValueError(f"No cells found for perturbation '{perturbation_name}'")
    
    print(f"Found {pert_cells.n_obs} cells for perturbation '{perturbation_name}'")
    
    # Split perturbation cells randomly into GT and TD groups
    n_pert_cells = pert_cells.n_obs
    gt_indices = np.random.choice(n_pert_cells, size=n_pert_cells//2, replace=False)
    td_indices = np.setdiff1d(np.arange(n_pert_cells), gt_indices)
    
    gt_cells = pert_cells[gt_indices]
    td_cells = pert_cells[td_indices]
    
    print(f"Split into GT: {len(gt_indices)} cells, TD: {len(td_indices)} cells")
    
    # Get ground truth (average of GT cells)
    gt_profile = gt_cells.X.mean(axis=0).A1 if hasattr(gt_cells.X, 'A1') else gt_cells.X.mean(axis=0)
    
    # Get mean baseline (MB)
    mean_baseline = adata.uns['split_fold_0_mean_baseline'].iloc[0]
    
    # Get control cells and SMB cells (non-control non-perturbation)
    control_mask = adata.obs['condition'].str.contains('control', case=False, na=False)
    non_control_non_pert_mask = ~control_mask & (adata.obs['condition'] != perturbation_name.split('_')[1])
    
    control_cells = adata[control_mask]
    smb_cells = adata[non_control_non_pert_mask]
    
    # Compute global control profile (CTL) for reference
    global_control_profile = control_cells.X.mean(axis=0).A1 if hasattr(control_cells.X, 'A1') else control_cells.X.mean(axis=0)
    
    print(f"Control cells: {control_cells.n_obs}, SMB cells: {smb_cells.n_obs}")
    
    # Define log space from 1 to 100,000 cells (or max available)
    max_cells = min(100000, smb_cells.n_obs)
    cell_numbers = np.unique(np.logspace(0, np.log10(max_cells), n_points, dtype=int))
    
    # Compute reference delta using global control profile
    gt_minus_ctl = gt_profile - global_control_profile
    
    # Compute MB metrics (constant for all points)
    mb_mse = np.mean(np.square(gt_profile - mean_baseline.values))
    mb_minus_ctl = mean_baseline.values - global_control_profile
    mb_pearson, _ = pearsonr(np.array(gt_minus_ctl).squeeze(), np.array(mb_minus_ctl).squeeze())
    
    results = []
    
    print(f"Performing SMB titration with {len(cell_numbers)} points, max cells: {max_cells}")
    
    for n_cells in tqdm(cell_numbers, desc="SMB titration"):
        # Sample n_cells from SMB cells
        if n_cells <= smb_cells.n_obs:
            smb_sample_indices = np.random.choice(smb_cells.n_obs, size=n_cells, replace=False)
        else:
            # If requesting more cells than available, use all with replacement
            smb_sample_indices = np.random.choice(smb_cells.n_obs, size=n_cells, replace=True)
            
        smb_sample = smb_cells[smb_sample_indices]
        smb_profile = smb_sample.X.mean(axis=0).A1 if hasattr(smb_sample.X, 'A1') else smb_sample.X.mean(axis=0)
        smb_mse = np.mean(np.square(gt_profile - smb_profile))
        smb_delta = smb_profile - global_control_profile
        smb_pearson, _ = pearsonr(np.array(gt_minus_ctl).squeeze(), np.array(smb_delta).squeeze())
        
        results.append({
            'n_cells': n_cells,
            'smb_mse': smb_mse,
            'mb_mse': mb_mse,
            'smb_pearson': smb_pearson,
            'mb_pearson': mb_pearson
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).set_index('n_cells')
    
    print(f"\nGlobal Mean Baseline (MB) MSE: {mb_mse:.4f}")
    print(f"MB Pearson Delta (relative to CTL): {mb_pearson:.4f}")
    print(f"Results DataFrame shape: {results_df.shape}")
    
    return results_df

# %% Run SMB experiment for 20 perturbations

# Select 20 perturbations with equal spacing by DEG count
n_perturbations = 20
if len(deg_counts) >= n_perturbations:
    indices = np.linspace(0, len(deg_counts) - 1, n_perturbations, dtype=int)
    selected_perturbations = deg_counts.index[indices]
else:
    selected_perturbations = deg_counts.index

print(f"Selected perturbations for SMB experiment: {selected_perturbations.tolist()}")
print(f"DEG counts: {deg_counts[selected_perturbations].tolist()}")

# Compute all SMB results
print("\nComputing SMB titration experiments...")
smb_results = {}

for i, pert_name in enumerate(selected_perturbations):
    print(f"\nProcessing {i+1}/{len(selected_perturbations)}: {pert_name}")
    
    try:
        results_df = smb_titration_experiment(adata, perturbation_name=pert_name, n_points=30, random_seed=42)
        smb_results[pert_name] = results_df
    except Exception as e:
        print(f"  Error processing {pert_name}: {e}")
        smb_results[pert_name] = None

# %% Create aggregated plots for SMB experiment

# Create subplots for MSE (SMB experiment)
n_cols = 5
n_rows = 4
fig_smb_mse, axes_smb_mse = plt.subplots(n_rows, n_cols, figsize=(25, 14))  # 16:9 PPT aspect ratio
axes_smb_mse = axes_smb_mse.flatten()

for i, pert_name in enumerate(selected_perturbations):
    if i >= len(axes_smb_mse):
        break
        
    # Get number of DEGs
    deg_gene_dict = adata.uns.get('deg_gene_dict', {})
    n_degs = len(deg_gene_dict.get(pert_name, []))
    
    if smb_results[pert_name] is not None:
        results_df = smb_results[pert_name]
        
        # Plot MSE
        ax = axes_smb_mse[i]
        ax.plot(results_df.index, results_df['smb_mse'], 'o-', color='red', 
                label='SMB', linewidth=1.5, markersize=2)
        
        # Add horizontal line for MB
        ax.axhline(y=results_df['mb_mse'].iloc[0], color='black', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='MB')
        
        # Set both axes to log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Customize subplot
        ax.set_xlabel('Number of Cells (log)', fontsize=8)
        ax.set_ylabel('MSE (log scale)', fontsize=8)
        ax.set_title(f'{pert_name}\nDEGs: {n_degs}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_axisbelow(True)
        
        # Legend will be added at figure level
        
        sns.despine(ax=ax)
    else:
        # Clear the subplot if there was an error
        axes_smb_mse[i].clear()
        axes_smb_mse[i].text(0.5, 0.5, f'Error\n{pert_name}', ha='center', va='center', 
                             transform=axes_smb_mse[i].transAxes)
        axes_smb_mse[i].set_title(f'{pert_name}\nError', fontsize=8)

# Hide unused subplots
for i in range(len(selected_perturbations), len(axes_smb_mse)):
    axes_smb_mse[i].set_visible(False)

# Add single legend for entire figure on the right
handles, labels = axes_smb_mse[0].get_legend_handles_labels()
fig_smb_mse.legend(handles, labels, loc='center left', bbox_to_anchor=(0.99, 0.5), 
                    fontsize=16, framealpha=0.9)

fig_smb_mse.suptitle('SMB Titration Experiment - MSE (1 to 100K cells)', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.94, 0.96])  # Leave space for legend and title
plt.show()

# Create subplots for Pearson Delta (SMB experiment)
fig_smb_pearson, axes_smb_pearson = plt.subplots(n_rows, n_cols, figsize=(25, 14))  # 16:9 PPT aspect ratio
axes_smb_pearson = axes_smb_pearson.flatten()

for i, pert_name in enumerate(selected_perturbations):
    if i >= len(axes_smb_pearson):
        break
        
    # Get number of DEGs
    deg_gene_dict = adata.uns.get('deg_gene_dict', {})
    n_degs = len(deg_gene_dict.get(pert_name, []))
    
    if smb_results[pert_name] is not None:
        results_df = smb_results[pert_name]
        
        # Plot Pearson Delta
        ax = axes_smb_pearson[i]
        ax.plot(results_df.index, results_df['smb_pearson'], 'o-', color='red', 
                label='SMB', linewidth=1.5, markersize=2)
        
        # Add horizontal line for MB
        ax.axhline(y=results_df['mb_pearson'].iloc[0], color='black', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='MB')
        
        # Set x-axis to log scale
        ax.set_xscale('log')
        
        # Customize subplot
        ax.set_xlabel('Number of Cells (log)', fontsize=8)
        ax.set_ylabel('Pearson Delta (relative to CTL)', fontsize=8)
        ax.set_title(f'{pert_name}\nDEGs: {n_degs}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_axisbelow(True)
        
        # Legend will be added at figure level
        
        sns.despine(ax=ax)
    else:
        # Clear the subplot if there was an error
        axes_smb_pearson[i].clear()
        axes_smb_pearson[i].text(0.5, 0.5, f'Error\n{pert_name}', ha='center', va='center', 
                                 transform=axes_smb_pearson[i].transAxes)
        axes_smb_pearson[i].set_title(f'{pert_name}\nError', fontsize=8)

# Hide unused subplots
for i in range(len(selected_perturbations), len(axes_smb_pearson)):
    axes_smb_pearson[i].set_visible(False)

# Add single legend for entire figure on the right
handles, labels = axes_smb_pearson[0].get_legend_handles_labels()
fig_smb_pearson.legend(handles, labels, loc='center left', bbox_to_anchor=(0.99, 0.5), 
                        fontsize=16, framealpha=0.9)

fig_smb_pearson.suptitle('SMB Titration Experiment - Pearson Delta (1 to 100K cells)', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.94, 0.96])  # Leave space for legend and title
plt.show()

# %%