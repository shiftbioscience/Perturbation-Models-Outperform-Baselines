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



# %% Define function to analyze scores vs prediction errors

def plot_scores_vs_errors(adata, perturbation_name, figsize=(20, 12), alpha=0.6):
    """
    Plot perturbation scores vs prediction errors for a single perturbation.
    
    Creates a 2x3 grid of subplots:
    Top row (Mean Baseline):
    1. Points colored by DEG status (red for DEGs, gray for others)
    2. Points colored by p-values directly (logarithmic scale, 1e-10 to 1)
    3. Points colored by raw scores (diverging colormap, -10 to +10)
    
    Bottom row (Sparse Mean Baseline):
    Same three coloring schemes but with sparse mean baseline on y-axis
    
    Args:
        adata: AnnData object containing all necessary data
        perturbation_name: Name/key of the perturbation to analyze
        figsize: Figure size for the plot (width, height)
        alpha: Transparency for scatter points
        
    Returns:
        matplotlib figure object
    """
    # Retrieve all required data with validation
    required_keys = {
        'scores_df_dict': 'scores',
        'names_df_dict': 'gene names',
        'pvals_df_dict': 'p-values'
    }
    
    # Validate and retrieve from dictionaries
    for key, desc in required_keys.items():
        if perturbation_name not in adata.uns[key]:
            raise ValueError(f"Perturbation '{perturbation_name}' not found in {key}")
    
    scores = adata.uns['scores_df_dict'][perturbation_name]
    gene_names = adata.uns['names_df_dict'][perturbation_name]
    pvals = adata.uns['pvals_df_dict'][perturbation_name]
    
    # Retrieve baseline data
    required_baselines = [
        ('technical_duplicate_first_half_baseline', 'groundtruth'),
        ('technical_duplicate_second_half_baseline', 'td'),
        ('sparse_mean_baseline_seed_0', 'sparse_mean_baseline')
    ]
    
    data_dict = {}
    for baseline_key, var_name in required_baselines:
        if perturbation_name not in adata.uns[baseline_key].index:
            raise ValueError(f"Perturbation '{perturbation_name}' not found in {baseline_key}")
        data_dict[var_name] = adata.uns[baseline_key].loc[perturbation_name]
    
    groundtruth = data_dict['groundtruth']
    td = data_dict['td']
    sparse_mean_baseline = data_dict['sparse_mean_baseline']
    mean_baseline = adata.uns['split_fold_0_mean_baseline'].iloc[0]
    
    # Convert scores and p-values to pandas Series with gene names as index
    scores_series = pd.Series(scores, index=gene_names) if isinstance(scores, np.ndarray) else scores
    pvals_series = pd.Series(pvals, index=gene_names) if isinstance(pvals, np.ndarray) else pvals
    
    # Find common genes across all datasets and sort
    all_indices = [groundtruth.index, td.index, mean_baseline.index, 
                   sparse_mean_baseline.index, scores_series.index]
    common_genes = sorted(set.intersection(*[set(idx) for idx in all_indices]))
    
    # Align all data to common genes
    data_to_align = {
        'groundtruth': groundtruth, 'td': td, 'mean_baseline': mean_baseline,
        'sparse_mean_baseline': sparse_mean_baseline, 'scores': scores_series, 
        'pvals_aligned': pvals_series
    }
    aligned_data = {k: v[common_genes] for k, v in data_to_align.items()}
    
    # Unpack aligned data
    groundtruth = aligned_data['groundtruth']
    td = aligned_data['td']
    mean_baseline = aligned_data['mean_baseline']
    sparse_mean_baseline = aligned_data['sparse_mean_baseline']
    scores = aligned_data['scores']
    pvals_aligned = aligned_data['pvals_aligned']
    
    # Compute absolute errors and convert to log10 scale
    compute_log_error = lambda pred: np.log10(np.abs(groundtruth - pred) + 1e-10)
    
    log_errors_td = compute_log_error(td)
    log_errors_mean = compute_log_error(mean_baseline)
    log_errors_sparse_mean = compute_log_error(sparse_mean_baseline)
    
    # Create 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    ax1, ax2, ax3 = axes[0]  # Top row (mean baseline)
    ax4, ax5, ax6 = axes[1]  # Bottom row (sparse mean baseline)
    
    # Get DEG mask for this perturbation
    deg_genes = set(adata.uns.get('deg_gene_dict', {}).get(perturbation_name, []))
    significant_mask = pd.Series([gene in deg_genes for gene in common_genes], index=common_genes)
    
    # Prepare data for visualization
    pvals_clipped = np.maximum(pvals_aligned, 1e-10)  # Clip p-values for log scale
    scores_clipped = np.clip(scores, -10, 10)  # Clip scores to range [-10, 10]
    
    # Helper function to create scatter plots
    def create_scatter(ax, x, y, plot_type, alpha_val=alpha):
        if plot_type == 'deg':
            # DEG coloring
            ax.scatter(x, y, c='gray', s=30, alpha=alpha_val, label='All genes')
            if significant_mask.any():
                ax.scatter(x[significant_mask], y[significant_mask], 
                          c='red', s=30, alpha=0.8, label=f'DEGs (n={significant_mask.sum()})')
            ax.legend(loc='upper right')
        elif plot_type == 'pval':
            # P-value coloring
            scatter = ax.scatter(x, y, c=pvals_clipped, s=30, alpha=1.0,
                                cmap='viridis', norm=LogNorm(vmin=1e-10, vmax=1))
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Adjusted p-value\n(saturated at 1e-10)', rotation=270, labelpad=20)
        elif plot_type == 'score':
            # Score coloring
            scatter = ax.scatter(x, y, c=scores_clipped, s=30, alpha=1.0,
                                cmap='RdBu_r', vmin=-10, vmax=10)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Raw Score\n(clipped to ±10)', rotation=270, labelpad=20)
    
    # Create all subplots
    plot_configs = [
        (ax1, log_errors_mean, 'deg'),
        (ax2, log_errors_mean, 'pval'),
        (ax3, log_errors_mean, 'score'),
        (ax4, log_errors_sparse_mean, 'deg'),
        (ax5, log_errors_sparse_mean, 'pval'),
        (ax6, log_errors_sparse_mean, 'score')
    ]
    
    for ax, y_errors, plot_type in plot_configs:
        create_scatter(ax, log_errors_td, y_errors, plot_type)
    
    # Add identity lines and annotations to all subplots
    all_axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    y_errors_list = [log_errors_mean] * 3 + [log_errors_sparse_mean] * 3
    
    for ax, y_errors in zip(all_axes, y_errors_list):
        # Add identity line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.7, linewidth=2)
        
        # Force matplotlib to update the plot and set proper axis limits
        ax.relim()
        ax.autoscale_view()
        
        # Add percentage annotations (above and below identity line)
        above_identity = np.sum(y_errors > log_errors_td)
        below_identity = np.sum(y_errors < log_errors_td)
        total_genes = len(y_errors)
        frac_above = above_identity / total_genes
        frac_below = below_identity / total_genes
        
        # Get current axis limits after plotting
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        
        # Top-left: mean > td (above identity line) - points where y > x
        ax.text(
            xlim[0] + 0.05 * (xlim[1] - xlim[0]),
            ylim[1] - 0.05 * (ylim[1] - ylim[0]),
            f'{frac_above:.1%}',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            va='top', ha='left'
        )
        # Bottom-right: mean < td (below identity line) - points where y < x  
        ax.text(
            xlim[1] - 0.05 * (xlim[1] - xlim[0]),
            ylim[0] + 0.05 * (ylim[1] - ylim[0]),
            f'{frac_below:.1%}',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            va='bottom', ha='right'
        )
        
        # Customize plot
        ax.set_xlabel('log10(Technical Duplicate Error)', fontsize=11)
        # Set y-axis label based on row
        if ax in [ax1, ax2, ax3]:
            ax.set_ylabel('log10(Mean Baseline Error)', fontsize=11)
        else:
            ax.set_ylabel('log10(Sparse Mean Baseline Error)', fontsize=11)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Set square aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Despine
        sns.despine(ax=ax)
    
    # Set subplot titles
    titles = [
        'Mean Baseline: DEG Status', 'Mean Baseline: P-value (log scale)', 'Mean Baseline: Raw Score',
        'Sparse Mean: DEG Status', 'Sparse Mean: P-value (log scale)', 'Sparse Mean: Raw Score'
    ]
    for ax, title in zip(all_axes, titles):
        ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Add main title
    fig.suptitle(f'Prediction Error Comparison (log10 scale)\n{perturbation_name}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


# %% Define function to analyze all perturbations and plot DEG ranking

def analyze_all_perturbations_deg_ranking(adata, dataset_name=None, figsize=(10, 8)):
    """
    Analyze technical duplicate performance vs baselines for all perturbations
    and plot results ranked by number of DEGs.
    
    Args:
        adata: AnnData object containing all necessary data
        dataset_name: Name of the dataset to include in plot title (optional)
        figsize: Figure size for the plot (width, height)
        
    Returns:
        tuple: (DataFrame with results, matplotlib figure object)
    """
    # Get list of available perturbations (excluding controls)
    available_perturbations = [p for p in adata.uns['scores_df_dict'].keys() 
                              if 'control' not in p.lower()]
    
    # Initialize result storage
    results = []
    
    print(f"Analyzing {len(available_perturbations)} perturbations...")
    
    for pert_name in tqdm(available_perturbations, desc="Processing perturbations"):
        try:
            # Retrieve required data for this perturbation
            scores = adata.uns['scores_df_dict'][pert_name]
            gene_names = adata.uns['names_df_dict'][pert_name]
            
            # Retrieve baseline data
            if pert_name not in adata.uns['technical_duplicate_first_half_baseline'].index:
                continue
                
            groundtruth = adata.uns['technical_duplicate_first_half_baseline'].loc[pert_name]
            td = adata.uns['technical_duplicate_second_half_baseline'].loc[pert_name]
            sparse_mean_baseline = adata.uns['sparse_mean_baseline_seed_0'].loc[pert_name]
            mean_baseline = adata.uns['split_fold_0_mean_baseline'].iloc[0]
            
            # Find common genes across all datasets and align
            all_indices = [groundtruth.index, td.index, mean_baseline.index, sparse_mean_baseline.index]
            common_genes = sorted(set.intersection(*[set(idx) for idx in all_indices]))
            
            # Align all data to common genes
            groundtruth_aligned = groundtruth[common_genes]
            td_aligned = td[common_genes]
            mean_baseline_aligned = mean_baseline[common_genes]
            sparse_mean_baseline_aligned = sparse_mean_baseline[common_genes]
            
            # Compute absolute errors
            errors_td = np.abs(groundtruth_aligned - td_aligned)
            errors_mean = np.abs(groundtruth_aligned - mean_baseline_aligned)
            errors_sparse_mean = np.abs(groundtruth_aligned - sparse_mean_baseline_aligned)
            
            # Compute proportions where TD beats each baseline (lower error is better)
            prop_td_beats_mean = (errors_td < errors_mean).mean()
            prop_td_beats_sparse_mean = (errors_td < errors_sparse_mean).mean()
            
            # Get number of DEGs
            deg_genes = adata.uns.get('deg_gene_dict', {}).get(pert_name, [])
            n_degs = len(deg_genes)
            
            results.append({
                'perturbation': pert_name,
                'prop_td_beats_mean': prop_td_beats_mean,
                'prop_td_beats_sparse_mean': prop_td_beats_sparse_mean,
                'n_degs': n_degs
            })
            
        except Exception as e:
            print(f"  Error processing {pert_name}: {e}")
            continue
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results).set_index('perturbation')
    
    # Sort by number of DEGs and create ranking (breaking ties with arange)
    results_df = results_df.sort_values('n_degs', ascending=False)
    results_df['deg_ranking'] = np.arange(1, len(results_df) + 1)
    
    print(f"\nProcessed {len(results_df)} perturbations successfully")
    print(f"DEG count range: {results_df['n_degs'].min()} to {results_df['n_degs'].max()}")
    print(f"Mean proportion TD beats mean baseline: {results_df['prop_td_beats_mean'].mean():.3f}")
    print(f"Mean proportion TD beats sparse mean baseline: {results_df['prop_td_beats_sparse_mean'].mean():.3f}")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot both proportions using scatter (points only)
    ax.scatter(results_df['deg_ranking'], results_df['prop_td_beats_mean'], 
               color='darkblue', alpha=0.7, s=10,
               label='TD beats Mean Baseline')
    ax.scatter(results_df['deg_ranking'], results_df['prop_td_beats_sparse_mean'], 
               color='red', alpha=0.7, s=10,
               label='TD beats Sparse Mean Baseline')
    
    # Add horizontal line at 0.5 for reference (black and more visible)
    ax.axhline(y=0.5, color='black', linestyle='-', alpha=0.8, linewidth=2,
              label='Equal Performance')
    
    # Customize plot
    ax.set_xlabel('DEG Ranking (1 = most DEGs)', fontsize=12)
    ax.set_ylabel('Proportion where TD beats baseline', fontsize=12)
    # Include dataset name in title if provided
    if dataset_name:
        title = f'Technical Duplicate Performance vs Baselines - {dataset_name}\nRanked by Number of DEGs'
    else:
        title = 'Technical Duplicate Performance vs Baselines\nRanked by Number of DEGs'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set axis limits
    ax.set_xlim([0, len(results_df)])
    ax.set_ylim([0, 1])
    
    # Add legend
    ax.legend(loc='best', framealpha=0.9, edgecolor='none')
    
    # Despine
    sns.despine(ax=ax)
    
    plt.tight_layout()
    
    return results_df, fig


# %% Test the function with a few random perturbations

# Get list of available perturbations (excluding controls)
available_perturbations = [p for p in adata.uns['scores_df_dict'].keys() 
                          if 'control' not in p.lower()]

# Randomly select 3 perturbations
import random
random.seed(42)  # For reproducibility
selected_perturbations = random.sample(available_perturbations, 30)

print(f"Selected perturbations for visualization: {selected_perturbations}")

# Create plots for each selected perturbation
for i, pert_name in enumerate(selected_perturbations, 1):
    print(f"\nCreating plot {i}/30 for: {pert_name}")
    
    try:
        fig = plot_scores_vs_errors(adata, pert_name)
        plt.show()
        
        # Print some statistics
        scores = adata.uns['scores_df_dict'][pert_name]
        if isinstance(scores, np.ndarray):
            print(f"  Score statistics:")
            print(f"    Min: {scores.min():.4f}, Max: {scores.max():.4f}, Mean: {scores.mean():.4f}")
            print(f"    Non-zero scores: {(scores != 0).sum()} / {len(scores)}")
        else:
            print(f"  Score statistics:")
            print(f"    Min: {scores.min():.4f}, Max: {scores.max():.4f}, Mean: {scores.mean():.4f}")
            print(f"    Non-zero scores: {(scores != 0).sum()} / {len(scores)}")
        
    except Exception as e:
        print(f"  Error processing {pert_name}: {e}")


# %% Run the new analysis function

# Run the analysis for all perturbations
results_df, fig = analyze_all_perturbations_deg_ranking(adata, dataset_name=dataset_name)

# Display the plot
plt.show()
