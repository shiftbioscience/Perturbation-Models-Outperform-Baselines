# %% Imports

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

np.random.seed(42)

plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200

# %% Load data

data_path = Path("/home/gabriel/CellSimBench/data/frangieh21/frangieh21_processed_complete.h5ad")
print(f"Loading data from: {data_path}")
adata = sc.read_h5ad(data_path)
print(f"Data loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")
print(f"Conditions: {adata.obs['condition'].nunique()}")

# %% Identify perturbations (excluding controls)

all_conditions = adata.obs['condition'].unique()
non_control_conditions = sorted([cond for cond in all_conditions if 'control' not in cond.lower() and 'ctrl' not in cond.lower()])

print(f"Total conditions: {len(all_conditions)}")
print(f"Non-control conditions: {len(non_control_conditions)}")

# Get cell counts per perturbation
pert_counts = adata.obs['condition'].value_counts()
valid_perts = [pert for pert in non_control_conditions if pert_counts[pert] >= 10]

print(f"Perturbations with >= 10 cells: {len(valid_perts)}")

# %% Split perturbations into two halves

print("Splitting perturbations into two halves...")

adata.obs['coherence_split'] = 'unassigned'

for pert in tqdm(valid_perts, desc="Splitting perturbations"):
    pert_cells = adata.obs[adata.obs['condition'] == pert].index
    n_cells = len(pert_cells)
    
    # Randomly shuffle cells
    shuffled_cells = np.random.permutation(pert_cells)
    
    # Split in half
    half_point = n_cells // 2
    half1_cells = shuffled_cells[:half_point]
    half2_cells = shuffled_cells[half_point:]
    
    # Assign to halves
    adata.obs.loc[half1_cells, 'coherence_split'] = 'half1'
    adata.obs.loc[half2_cells, 'coherence_split'] = 'half2'

print("Split complete!")
print(f"Half 1 cells: {(adata.obs['coherence_split'] == 'half1').sum()}")
print(f"Half 2 cells: {(adata.obs['coherence_split'] == 'half2').sum()}")

# %% Compute DEGs for each half

def compute_degs_for_half(adata, half_name, valid_perts):
    """Compute DEGs for one half of the data."""
    print(f"\nComputing DEGs for {half_name}...")
    
    # Filter to this half
    adata_half = adata[adata.obs['coherence_split'] == half_name].copy()
    
    # Filter to valid perturbations
    adata_half = adata_half[adata_half.obs['condition'].isin(valid_perts)].copy()
    
    print(f"Using {adata_half.shape[0]} cells from {len(valid_perts)} perturbations")
    
    # Compute DEGs using scanpy
    sc.tl.rank_genes_groups(adata_half, 'condition', method='t-test_overestim_var', reference='rest')
    
    # Extract results
    names_df = pd.DataFrame(adata_half.uns["rank_genes_groups"]["names"])
    pvals_df = pd.DataFrame(adata_half.uns["rank_genes_groups"]["pvals_adj"])
    scores_df = pd.DataFrame(adata_half.uns["rank_genes_groups"]["scores"])
    
    return names_df, pvals_df, scores_df

# Compute DEGs for both halves
names_half1, pvals_half1, scores_half1 = compute_degs_for_half(adata, 'half1', valid_perts)
names_half2, pvals_half2, scores_half2 = compute_degs_for_half(adata, 'half2', valid_perts)

print("\nDEG computation complete!")

# %% Coherence analysis per perturbation

results = []

for pert in tqdm(valid_perts, desc="Computing coherence metrics"):
    # Get scores for this perturbation from both halves
    if pert not in scores_half1.columns or pert not in scores_half2.columns:
        continue
    
    scores1 = scores_half1[pert].values
    scores2 = scores_half2[pert].values
    genes1 = names_half1[pert].values
    genes2 = names_half2[pert].values
    
    # Get absolute t-statistics
    abs_scores1 = np.abs(scores1)
    abs_scores2 = np.abs(scores2)
    
    # Get top 1000 genes by absolute t-statistic for each half
    top1000_idx1 = np.argsort(abs_scores1)[-1000:]
    top1000_idx2 = np.argsort(abs_scores2)[-1000:]
    
    top1000_genes1 = set(genes1[top1000_idx1])
    top1000_genes2 = set(genes2[top1000_idx2])
    
    # Compute union
    union_genes = top1000_genes1.union(top1000_genes2)
    
    # Compute intersection
    intersection_genes = top1000_genes1.intersection(top1000_genes2)
    
    # Compute IoU (Intersection over Union)
    iou = len(intersection_genes) / len(union_genes) if len(union_genes) > 0 else 0.0
    
    # For correlation, align genes properly using dictionaries and a consistent gene order
    # Create dictionaries mapping gene to t-statistic and p-values
    gene_to_score1 = {gene: score for gene, score in zip(genes1, scores1)}
    gene_to_score2 = {gene: score for gene, score in zip(genes2, scores2)}
    gene_to_abs_score1 = {gene: abs_score for gene, abs_score in zip(genes1, abs_scores1)}
    gene_to_abs_score2 = {gene: abs_score for gene, abs_score in zip(genes2, abs_scores2)}
    
    # Get p-values for this perturbation
    pvals1 = pvals_half1[pert].values
    pvals2 = pvals_half2[pert].values
    
    # Count DEGs (padj < 0.05) in each half
    n_degs_half1 = (pvals1 < 0.05).sum()
    n_degs_half2 = (pvals2 < 0.05).sum()
    avg_n_degs = (n_degs_half1 + n_degs_half2) / 2.0
    
    # Get common genes and sort them to ensure consistent ordering
    common_genes = sorted(set(genes1).intersection(set(genes2)))
    
    # Align scores using the same gene order (ALL genes, not filtered)
    scores1_aligned = np.array([gene_to_score1[gene] for gene in common_genes])
    scores2_aligned = np.array([gene_to_score2[gene] for gene in common_genes])
    abs_scores1_aligned = np.array([gene_to_abs_score1[gene] for gene in common_genes])
    abs_scores2_aligned = np.array([gene_to_abs_score2[gene] for gene in common_genes])
    
    # Compute Pearson and Spearman correlations on RAW t-statistics (ALL genes)
    pearson_corr, _ = pearsonr(scores1_aligned, scores2_aligned)
    spearman_corr, _ = spearmanr(scores1_aligned, scores2_aligned)
    
    # Compute Pearson and Spearman correlations on ABSOLUTE t-statistics (ALL genes)
    pearson_corr_abs, _ = pearsonr(abs_scores1_aligned, abs_scores2_aligned)
    spearman_corr_abs, _ = spearmanr(abs_scores1_aligned, abs_scores2_aligned)
    
    results.append({
        'perturbation': pert,
        'n_cells_half1': ((adata.obs['condition'] == pert) & (adata.obs['coherence_split'] == 'half1')).sum(),
        'n_cells_half2': ((adata.obs['condition'] == pert) & (adata.obs['coherence_split'] == 'half2')).sum(),
        'n_degs_half1': n_degs_half1,
        'n_degs_half2': n_degs_half2,
        'avg_n_degs': avg_n_degs,
        'union_size': len(union_genes),
        'intersection_size': len(intersection_genes),
        'iou': iou,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'pearson_corr_abs': pearson_corr_abs,
        'spearman_corr_abs': spearman_corr_abs
    })

# Create results DataFrame
results_df = pd.DataFrame(results)

print("\nCoherence analysis complete!")
print(f"Results shape: {results_df.shape}")
print("\nSample results:")
print(results_df.head(10))

# %% Summary statistics

print("\n" + "="*50)
print("Summary Statistics")
print("="*50)
print(f"Mean IoU: {results_df['iou'].mean():.4f}")
print(f"Median IoU: {results_df['iou'].median():.4f}")
print(f"\nCorrelations on RAW t-statistics (with sign):")
print(f"Mean Pearson correlation: {results_df['pearson_corr'].mean():.4f}")
print(f"Median Pearson correlation: {results_df['pearson_corr'].median():.4f}")
print(f"Mean Spearman correlation: {results_df['spearman_corr'].mean():.4f}")
print(f"Median Spearman correlation: {results_df['spearman_corr'].median():.4f}")
print(f"\nCorrelations on ABSOLUTE t-statistics (magnitude only):")
print(f"Mean Pearson correlation (abs): {results_df['pearson_corr_abs'].mean():.4f}")
print(f"Median Pearson correlation (abs): {results_df['pearson_corr_abs'].median():.4f}")
print(f"Mean Spearman correlation (abs): {results_df['spearman_corr_abs'].mean():.4f}")
print(f"Median Spearman correlation (abs): {results_df['spearman_corr_abs'].median():.4f}")

# %% Select 10 perturbations with varying strengths

# Define strength as avg_n_degs (average number of DEGs with padj < 0.05 between halves)
results_df_sorted = results_df.sort_values('avg_n_degs', ascending=False)

# Select 10 perturbations evenly spaced by rank
n_perts = len(results_df_sorted)
selected_indices = np.linspace(0, n_perts - 1, 10, dtype=int)

selected_perts = results_df_sorted.iloc[selected_indices]['perturbation'].values

print("\n" + "="*50)
print("Selected perturbations for scatter plots:")
print("="*50)
for i, pert in enumerate(selected_perts):
    pert_data = results_df_sorted[results_df_sorted['perturbation'] == pert].iloc[0]
    print(f"{i+1}. {pert}: avg_n_degs={pert_data['avg_n_degs']:.1f}, IoU={pert_data['iou']:.4f}, Pearson={pert_data['pearson_corr']:.4f}")

# %% Plot scatter plots of t-statistics

fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()

for i, pert in enumerate(selected_perts):
    ax = axes[i]
    
    # Get scores and gene names for this perturbation
    scores1 = scores_half1[pert].values
    scores2 = scores_half2[pert].values
    genes1 = names_half1[pert].values
    genes2 = names_half2[pert].values
    
    # Get common genes (should be all genes since both use same gene set)
    common_genes = sorted(set(genes1).intersection(set(genes2)))
    
    # Create dictionaries mapping gene to t-statistic
    gene_to_score1 = {gene: score for gene, score in zip(genes1, scores1)}
    gene_to_score2 = {gene: score for gene, score in zip(genes2, scores2)}
    
    # Get aligned scores for ALL genes
    scores1_plot = np.array([gene_to_score1[gene] for gene in common_genes])
    scores2_plot = np.array([gene_to_score2[gene] for gene in common_genes])
    
    # Plot the aligned scores
    ax.scatter(scores1_plot, scores2_plot, alpha=0.2, s=20, color='steelblue', edgecolors='none')
    
    # Add diagonal line
    min_val = min(scores1_plot.min(), scores2_plot.min())
    max_val = max(scores1_plot.max(), scores2_plot.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
    
    # Get metrics for this perturbation
    pert_metrics = results_df[results_df['perturbation'] == pert].iloc[0]
    
    # Set labels and title
    ax.set_xlabel('t-statistic (Half 1)', fontsize=10)
    ax.set_ylabel('t-statistic (Half 2)', fontsize=10)
    ax.set_title(f'{pert} (n={len(common_genes)} genes)\navg_DEGs={pert_metrics["avg_n_degs"]:.0f}, Pearson={pert_metrics["pearson_corr"]:.3f}', 
                 fontsize=9, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Despine
    sns.despine(ax=ax)

plt.tight_layout()
plt.show()

# %% Save results

output_path = Path("/home/gabriel/CellSimBench/analyses/metric_problems/deg_coherence_results.csv")
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# %%

