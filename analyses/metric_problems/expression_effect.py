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


# %% Read data

path_dict = {
    "replogle22k562gwps": Path("./data/replogle22k562gwps/replogle22k562gwps_processed_complete.h5ad"),  # 256 cells
    "replogle22rpe1": Path("./data/replogle22rpe1/replogle22rpe1_processed_complete.h5ad"),  # 128 cells
    "replogle22k562": Path("./data/replogle22k562/replogle22k562_processed_complete.h5ad")  # 128 cells
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


# %% Define gene expression titration function

def gene_expression_titration_experiment(adata, perturbation_name, random_seed=42, figsize=(14, 10), plot=True):
    """
    Perform gene expression titration experiment for a given perturbation.
    
    Groups analyzed:
    - GT (Ground Truth): Half of perturbation cells used as reference
    - TD (Technical Duplicate): Other half of perturbation cells
    - MB (Mean Baseline): Global mean baseline
    - SMB (Sparse Mean Baseline): Sampled non-perturbation cells
    - CTL (Control): All control cells (used as reference for Pearson Delta)
    
    Metrics:
    - MSE: Mean squared error vs GT
    - Pearson Delta: Correlation of (prediction - CTL) vs (GT - CTL)
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing all data
    perturbation_name : str
        Name of the perturbation to analyze
    random_seed : int
        Random seed for reproducibility (default: 42)
    figsize : tuple
        Figure size for plotting (default: (14, 10))
    plot : bool
        Whether to create plots (default: True)
    
    Returns:
    --------
    tuple: (results_dict, fig) if plot=True, else results_dict
        Dictionary with results and optionally matplotlib figure
    """
    np.random.seed(random_seed)
    
    # Find cells that match this perturbation in condition column
    pert_mask = adata.obs['condition'] == perturbation_name.split('_')[1]
    pert_cells = adata[pert_mask]
    
    if pert_cells.n_obs == 0:
        raise ValueError(f"No cells found for perturbation '{perturbation_name}'")
    
    print(f"Found {pert_cells.n_obs} cells for perturbation '{perturbation_name}'")
    
    # Split perturbation cells randomly into GT (Ground Truth) and TD (Technical Duplicate) groups
    n_pert_cells = pert_cells.n_obs
    gt_indices = np.random.choice(n_pert_cells, size=n_pert_cells//2, replace=False)
    td_indices = np.setdiff1d(np.arange(n_pert_cells), gt_indices)
    
    gt_cells = pert_cells[gt_indices]
    td_cells = pert_cells[td_indices]
    
    print(f"Split into GT (Ground Truth): {len(gt_indices)} cells, TD (Technical Duplicate): {len(td_indices)} cells")
    
    # Get ground truth (average of GT cells)
    gt_mean = gt_cells.X.mean(axis=0)
    if hasattr(gt_mean, 'A1'):
        gt_profile = gt_mean.A1  # Convert sparse matrix to 1D array
    else:
        gt_profile = np.asarray(gt_mean).squeeze()  # Ensure it's a 1D numpy array
    
    # Get global mean baseline (mb)
    mean_baseline = adata.uns['split_fold_0_mean_baseline'].iloc[0].values
    
    # Get control cells for ctl group
    control_mask = adata.obs['condition'].str.contains('control', case=False, na=False)
    control_cells = adata[control_mask]
    
    # Get non-control non-perturbation cells for sparse mean baseline
    non_control_non_pert_mask = ~control_mask & (adata.obs['condition'] != perturbation_name.split('_')[1])
    other_cells = adata[non_control_non_pert_mask]
    
    print(f"Control cells: {control_cells.n_obs}, Other cells: {other_cells.n_obs}")
    
    # Create predictions for each group
    # TD: Technical Duplicate - average of TD cells
    td_mean = td_cells.X.mean(axis=0)
    td_profile = td_mean.A1 if hasattr(td_mean, 'A1') else np.asarray(td_mean).squeeze()
    
    # MB: global mean baseline (already computed)
    mb_profile = mean_baseline
    
    # SMB: sparse mean baseline - sample same number as TD from other cells
    n_td = td_cells.n_obs
    smb_indices = np.random.choice(other_cells.n_obs, size=n_td, replace=False)
    smb_cells = other_cells[smb_indices]
    smb_mean = smb_cells.X.mean(axis=0)
    smb_profile = smb_mean.A1 if hasattr(smb_mean, 'A1') else np.asarray(smb_mean).squeeze()
    
    # CTL: average of all control cells
    ctl_mean = control_cells.X.mean(axis=0)
    ctl_profile = ctl_mean.A1 if hasattr(ctl_mean, 'A1') else np.asarray(ctl_mean).squeeze()
    
    # Divide genes into 5 quantiles based on GT expression levels
    n_genes = len(gt_profile)
    gene_expression_levels = gt_profile
    
    # Calculate quantile thresholds
    quantile_thresholds = np.percentile(gene_expression_levels, [20, 40, 60, 80])
    
    # Assign genes to quantiles
    gene_quantiles = np.zeros(n_genes, dtype=int)
    gene_quantiles[gene_expression_levels <= quantile_thresholds[0]] = 0
    gene_quantiles[(gene_expression_levels > quantile_thresholds[0]) & (gene_expression_levels <= quantile_thresholds[1])] = 1
    gene_quantiles[(gene_expression_levels > quantile_thresholds[1]) & (gene_expression_levels <= quantile_thresholds[2])] = 2
    gene_quantiles[(gene_expression_levels > quantile_thresholds[2]) & (gene_expression_levels <= quantile_thresholds[3])] = 3
    gene_quantiles[gene_expression_levels > quantile_thresholds[3]] = 4
    
    # Compute metrics for each quantile and each group
    results = {
        'mse': {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': []},
        'pearson': {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': []}
    }
    
    groups = {
        'CTL': ctl_profile,
        'MB': mb_profile,
        'SMB': smb_profile,
        'TD': td_profile
    }
    
    for q in range(5):
        quantile_mask = gene_quantiles == q
        quantile_name = f'Q{q+1}'
        
        # Get GT profile for this quantile
        gt_q = gt_profile[quantile_mask]
        ctl_q = ctl_profile[quantile_mask]  # Use CTL as reference for Pearson delta
        gt_delta_q = gt_q - ctl_q  # GT delta now uses CTL as reference
        
        for group_name, group_profile in groups.items():
            # Get group profile for this quantile
            group_q = group_profile[quantile_mask]
            
            # Compute MSE
            mse = np.mean(np.square(gt_q - group_q))
            
            # Compute Pearson delta (using CTL as reference)
            group_delta_q = group_q - ctl_q  # Group delta now uses CTL as reference
            pearson_corr, _ = pearsonr(gt_delta_q, group_delta_q)
            
            results['mse'][quantile_name].append(mse)
            results['pearson'][quantile_name].append(pearson_corr)
    
    # Convert results to DataFrame for easier plotting
    mse_df = pd.DataFrame(results['mse'], index=['CTL', 'MB', 'SMB', 'TD'])
    pearson_df = pd.DataFrame(results['pearson'], index=['CTL', 'MB', 'SMB', 'TD'])
    
    results_dict = {
        'mse': mse_df,
        'pearson': pearson_df,
        'quantile_thresholds': quantile_thresholds,
        'gene_quantiles': gene_quantiles
    }
    
    print(f"\nQuantile thresholds: {quantile_thresholds}")
    print(f"Genes per quantile: {[np.sum(gene_quantiles == q) for q in range(5)]}")
    
    if plot:
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Define colors for each group
        colors = {'CTL': 'green', 'MB': 'black', 'SMB': 'red', 'TD': 'darkblue'}  # Updated to match cell_titration.py -- [Coding Agent]
        
        # Plot MSE bar plot
        x = np.arange(5)
        width = 0.2
        
        for i, group in enumerate(['CTL', 'MB', 'SMB', 'TD']):
            mse_values = [mse_df[f'Q{q+1}'][group] for q in range(5)]
            ax1.bar(x + i*width - 1.5*width, mse_values, width, label=group, color=colors[group])
        
        ax1.set_xlabel('Gene Expression Quantile', fontsize=12)
        ax1.set_ylabel('MSE vs Ground Truth', fontsize=12)
        ax1.set_title('MSE by Gene Expression Level', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest)'])
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_axisbelow(True)
        ax1.set_yscale('log')
        sns.despine(ax=ax1)
        
        # Plot Pearson Delta bar plot
        for i, group in enumerate(['CTL', 'MB', 'SMB', 'TD']):
            pearson_values = [pearson_df[f'Q{q+1}'][group] for q in range(5)]
            ax2.bar(x + i*width - 1.5*width, pearson_values, width, label=group, color=colors[group])
        
        ax2.set_xlabel('Gene Expression Quantile', fontsize=12)
        ax2.set_ylabel('Pearson Delta (CTL-referenced)', fontsize=12)
        ax2.set_title('Pearson Delta by Gene Expression Level (CTL as reference)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest)'])
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_axisbelow(True)
        sns.despine(ax=ax2)
        
        # Get number of DEGs for this perturbation
        deg_gene_dict = adata.uns.get('deg_gene_dict', {})
        n_degs = len(deg_gene_dict.get(perturbation_name, []))
        
        # Add main title
        fig.suptitle(f'Gene Expression Titration - {perturbation_name}\nDEGs: {n_degs}', 
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        return results_dict, fig
    else:
        return results_dict


# %% Run gene expression titration experiment

results_dict, fig = gene_expression_titration_experiment(adata, perturbation_name=deg_counts.index[0], figsize=(16, 8))


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
        results_dict = gene_expression_titration_experiment(adata, perturbation_name=pert_name, random_seed=42, plot=False)
        all_results[pert_name] = results_dict
        
    except Exception as e:
        print(f"  Error processing {pert_name}: {e}")
        all_results[pert_name] = None

# Create subplots for MSE (all 20 perturbations)
n_cols = 5
n_rows = 4
fig_mse, axes_mse = plt.subplots(n_rows, n_cols, figsize=(22, 16))
axes_mse = axes_mse.flatten()

# Store handles and labels for legend
legend_handles = []
legend_labels = []

for i, pert_name in enumerate(selected_perturbations):
    if i >= len(axes_mse):
        break
        
    # Get number of DEGs
    deg_gene_dict = adata.uns.get('deg_gene_dict', {})
    n_degs = len(deg_gene_dict.get(pert_name, []))
    
    if all_results[pert_name] is not None:
        results_dict = all_results[pert_name]
        mse_df = results_dict['mse']
        
        # Plot MSE bar plot
        ax = axes_mse[i]
        x = np.arange(5)
        width = 0.18
        
        colors = {'CTL': 'green', 'MB': 'black', 'SMB': 'red', 'TD': 'darkblue'}  # Updated to match cell_titration.py -- [Coding Agent]
        
        for j, group in enumerate(['CTL', 'MB', 'SMB', 'TD']):
            mse_values = [mse_df[f'Q{q+1}'][group] for q in range(5)]
            bar = ax.bar(x + j*width - 1.5*width, mse_values, width, color=colors[group])
            
            # Store handles for legend (only once)
            if i == 0:
                legend_handles.append(bar[0])
                legend_labels.append(group)
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        
        # Customize subplot
        ax.set_xlabel('Quantile', fontsize=6)
        ax.set_ylabel('MSE (log)', fontsize=6)
        ax.set_title(f'{pert_name}\nDEGs: {n_degs}', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        sns.despine(ax=ax)
    else:
        # Clear the subplot if there was an error
        axes_mse[i].clear()
        axes_mse[i].text(0.5, 0.5, f'Error\n{pert_name}', ha='center', va='center', transform=axes_mse[i].transAxes)
        axes_mse[i].set_title(f'{pert_name}\nError', fontsize=10)

# Hide unused subplots for MSE
for i in range(len(selected_perturbations), len(axes_mse)):
    axes_mse[i].set_visible(False)

fig_mse.suptitle('MSE by Gene Expression Quantile - 20 Perturbations', fontsize=16, fontweight='bold', y=0.98)

# Add single legend on the right side
fig_mse.legend(legend_handles, legend_labels, 
               loc='center left', bbox_to_anchor=(0.95, 0.5),
               fontsize=12, frameon=True, framealpha=0.9,
               title='Group', title_fontsize=14)

plt.tight_layout(rect=[0, 0, 0.93, 0.97])  # Adjust layout to make room for legend
plt.show()

# Create subplots for Pearson Delta (all 20 perturbations)
fig_pearson, axes_pearson = plt.subplots(n_rows, n_cols, figsize=(22, 16))
axes_pearson = axes_pearson.flatten()

# Store handles and labels for legend
legend_handles_p = []
legend_labels_p = []

for i, pert_name in enumerate(selected_perturbations):
    if i >= len(axes_pearson):
        break
        
    # Get number of DEGs
    deg_gene_dict = adata.uns.get('deg_gene_dict', {})
    n_degs = len(deg_gene_dict.get(pert_name, []))
    
    if all_results[pert_name] is not None:
        results_dict = all_results[pert_name]
        pearson_df = results_dict['pearson']
        
        # Plot Pearson Delta bar plot
        ax = axes_pearson[i]
        x = np.arange(5)
        width = 0.18
        
        colors = {'CTL': 'green', 'MB': 'black', 'SMB': 'red', 'TD': 'darkblue'}  # Updated to match cell_titration.py -- [Coding Agent]
        
        for j, group in enumerate(['CTL', 'MB', 'SMB', 'TD']):
            pearson_values = [pearson_df[f'Q{q+1}'][group] for q in range(5)]
            bar = ax.bar(x + j*width - 1.5*width, pearson_values, width, color=colors[group])
            
            # Store handles for legend (only once)
            if i == 0:
                legend_handles_p.append(bar[0])
                legend_labels_p.append(group)
        
        # Customize subplot
        ax.set_xlabel('Quantile', fontsize=6)
        ax.set_ylabel('Pearson Delta (CTL-ref)', fontsize=6)
        ax.set_title(f'{pert_name}\nDEGs: {n_degs}', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        ax.set_ylim(-0.5, 1.0)  # Set consistent y-axis limits for Pearson
        
        sns.despine(ax=ax)
    else:
        # Clear the subplot if there was an error
        axes_pearson[i].clear()
        axes_pearson[i].text(0.5, 0.5, f'Error\n{pert_name}', ha='center', va='center', transform=axes_pearson[i].transAxes)
        axes_pearson[i].set_title(f'{pert_name}\nError', fontsize=10)

# Hide unused subplots for Pearson
for i in range(len(selected_perturbations), len(axes_pearson)):
    axes_pearson[i].set_visible(False)

fig_pearson.suptitle('Pearson Delta (CTL-referenced) by Gene Expression Quantile - 20 Perturbations', fontsize=16, fontweight='bold', y=0.98)

# Add single legend on the right side
fig_pearson.legend(legend_handles_p, legend_labels_p, 
                   loc='center left', bbox_to_anchor=(0.95, 0.5),
                   fontsize=12, frameon=True, framealpha=0.9,
                   title='Group', title_fontsize=14)

plt.tight_layout(rect=[0, 0, 0.93, 0.97])  # Adjust layout to make room for legend
plt.show()



# %%
