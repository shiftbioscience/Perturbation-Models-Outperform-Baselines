#!/usr/bin/env python3
"""
DEG Reproducibility Analysis Script

This script analyzes the reproducibility of differential expression analysis
by comparing DEG results calculated from two independent halves of the same data
(technical duplicate splits).

The script automatically caches DEG calculation results to speed up subsequent runs.
Cached results are stored in the output directory and will be reused unless:
  - The input file changes (detected via file modification time and size)
  - The --min-cells parameter changes
  - The --recalculate flag is used

Usage:
    # First run (calculates DEGs and caches them)
    python data/analyze_deg_reproducibility.py --input <path_to_processed_adata.h5ad>
    
    # Subsequent runs (reuses cached DEGs, much faster!)
    python data/analyze_deg_reproducibility.py --input <path_to_processed_adata.h5ad>
    
    # Force recalculation
    python data/analyze_deg_reproducibility.py --input <path_to_processed_adata.h5ad> --recalculate
    
    # Regenerate plots only (doesn't load AnnData, fastest)
    python data/analyze_deg_reproducibility.py --input <path_to_processed_adata.h5ad> --plots-only
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm
import warnings
import pickle
import hashlib
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


def get_input_hash(input_path, min_cells):
    """
    Generate a hash of the input file and parameters to detect changes.
    
    Args:
        input_path: Path to input h5ad file
        min_cells: Minimum cells parameter
        
    Returns:
        Hash string
    """
    # Use file modification time and size as proxy for content
    stat = os.stat(input_path)
    hash_input = f"{input_path}_{stat.st_mtime}_{stat.st_size}_{min_cells}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def save_deg_results(first_half_results, second_half_results, output_dir, input_hash, min_cells):
    """
    Save DEG calculation results to disk for caching.
    
    Args:
        first_half_results: DEG results for first half
        second_half_results: DEG results for second half
        output_dir: Directory to save cache files
        input_hash: Hash of input parameters
        min_cells: Minimum cells parameter
    """
    cache_dir = Path(output_dir) / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_hash': input_hash,
        'min_cells': min_cells,
        'n_perturbations_first': len(first_half_results),
        'n_perturbations_second': len(second_half_results)
    }
    
    metadata_path = cache_dir / 'deg_cache_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save DEG results
    first_half_path = cache_dir / 'first_half_degs.pkl'
    second_half_path = cache_dir / 'second_half_degs.pkl'
    
    with open(first_half_path, 'wb') as f:
        pickle.dump(first_half_results, f)
    
    with open(second_half_path, 'wb') as f:
        pickle.dump(second_half_results, f)
    
    print(f"\nCached DEG results saved to {cache_dir}")


def load_cached_deg_results(output_dir, input_hash, min_cells):
    """
    Load cached DEG results if they exist and are valid.
    
    Args:
        output_dir: Directory containing cache files
        input_hash: Hash of input parameters to validate cache
        min_cells: Minimum cells parameter
        
    Returns:
        Tuple of (first_half_results, second_half_results) or (None, None) if invalid
    """
    cache_dir = Path(output_dir) / 'cache'
    metadata_path = cache_dir / 'deg_cache_metadata.json'
    first_half_path = cache_dir / 'first_half_degs.pkl'
    second_half_path = cache_dir / 'second_half_degs.pkl'
    
    # Check if all files exist
    if not (metadata_path.exists() and first_half_path.exists() and second_half_path.exists()):
        return None, None
    
    # Load and validate metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if metadata['input_hash'] != input_hash:
            print("Cache invalid: input file has changed")
            return None, None
        
        if metadata['min_cells'] != min_cells:
            print(f"Cache invalid: min_cells changed ({metadata['min_cells']} -> {min_cells})")
            return None, None
        
        # Load cached results
        with open(first_half_path, 'rb') as f:
            first_half_results = pickle.load(f)
        
        with open(second_half_path, 'rb') as f:
            second_half_results = pickle.load(f)
        
        print(f"\nLoaded cached DEG results from {cache_dir}")
        print(f"  Cache created: {metadata['timestamp']}")
        print(f"  First half: {len(first_half_results)} perturbations")
        print(f"  Second half: {len(second_half_results)} perturbations")
        
        return first_half_results, second_half_results
    
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None, None


def calculate_degs_by_split(adata, split_value, min_cells=4, dataset_name=None):
    """
    Calculate differential expression genes for a specific technical duplicate split.
    
    Args:
        adata: AnnData object with tech_dup_split column
        split_value: Either 'first_half' or 'second_half'
        min_cells: Minimum number of cells required per perturbation
        dataset_name: Optional dataset name for condition prefixing
        
    Returns:
        Dictionary containing DEG results for each perturbation
    """
    print(f"\nCalculating DEGs for {split_value}...")
    
    # Filter to only use specified split
    adata_split = adata[adata.obs['tech_dup_split'] == split_value].copy()
    print(f"Using {adata_split.shape[0]} cells from {split_value}")
    
    # Get non-control conditions
    all_conditions = adata_split.obs['condition'].unique()
    non_control_conditions = [cond for cond in all_conditions 
                             if 'control' not in cond.lower() and 'ctrl' not in cond.lower()]
    
    # Filter perturbations with enough cells
    pert_counts = adata_split.obs['condition'].value_counts()
    valid_perts = pert_counts[(pert_counts >= min_cells) & 
                              (pert_counts.index.isin(non_control_conditions))].index
    adata_deg = adata_split[adata_split.obs['condition'].isin(valid_perts)].copy()
    
    print(f"Calculating DEGs for {len(valid_perts)} perturbations with ≥{min_cells} cells")
    
    if len(valid_perts) == 0:
        print(f"WARNING: No valid perturbations found in {split_value}")
        return {}
    
    # Calculate DEGs vs rest
    print(f"Computing DEGs vs rest for {split_value}...")
    sc.tl.rank_genes_groups(adata_deg, 'condition', method='t-test_overestim_var', 
                            reference='rest', key_added='rank_genes_groups')
    
    # Extract results
    names_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["names"])
    pvals_adj_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["pvals_adj"])
    pvals_unadj_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["pvals"])
    scores_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["scores"])
    
    # Store results per perturbation
    results = {}
    for pert in valid_perts:
        if pert not in names_df.columns:
            continue
            
        results[pert] = {
            'genes': names_df[pert].tolist(),
            'scores': scores_df[pert].tolist(),
            'pvals_adj': pvals_adj_df[pert].tolist(),
            'pvals_unadj': pvals_unadj_df[pert].tolist(),
            'sig_genes': names_df[pert][pvals_adj_df[pert] < 0.05].tolist(),
            'n_cells': int(pert_counts[pert])
        }
    
    print(f"Completed DEG calculation for {len(results)} perturbations in {split_value}")
    return results


def compare_deg_results(first_half_results, second_half_results, perturbation):
    """
    Compare DEG results between two halves for a single perturbation.
    
    Args:
        first_half_results: DEG results dictionary for first half
        second_half_results: DEG results dictionary for second half
        perturbation: Name of the perturbation to compare
        
    Returns:
        Dictionary containing comparison metrics
    """
    if perturbation not in first_half_results or perturbation not in second_half_results:
        return None
    
    first = first_half_results[perturbation]
    second = second_half_results[perturbation]
    
    metrics = {
        'perturbation': perturbation,
        'n_cells_first': first['n_cells'],
        'n_cells_second': second['n_cells'],
        'n_cells_total': first['n_cells'] + second['n_cells'],
        'n_degs_first': len(first['sig_genes']),
        'n_degs_second': len(second['sig_genes'])
    }
    
    # 1. Jaccard similarity of significant DEGs
    sig_first = set(first['sig_genes'])
    sig_second = set(second['sig_genes'])
    
    # Flag whether this perturbation has a measurable DEG signal
    metrics['has_degs_first'] = len(sig_first) > 0
    metrics['has_degs_second'] = len(sig_second) > 0
    metrics['has_degs_either'] = len(sig_first) > 0 or len(sig_second) > 0
    metrics['has_degs_both'] = len(sig_first) > 0 and len(sig_second) > 0
    
    if len(sig_first) == 0 and len(sig_second) == 0:
        # Both empty - cannot measure agreement, set to NaN
        metrics['jaccard_similarity'] = np.nan
    elif len(sig_first) == 0 or len(sig_second) == 0:
        # One empty, one not - complete disagreement
        metrics['jaccard_similarity'] = 0.0
    else:
        # Both have DEGs - calculate Jaccard
        intersection = len(sig_first & sig_second)
        union = len(sig_first | sig_second)
        metrics['jaccard_similarity'] = intersection / union if union > 0 else 0.0
    
    # 2. Overlap at top-N genes (as proportion)
    for top_n in [10, 50, 100, 200]:
        first_topn = set(first['genes'][:top_n])
        second_topn = set(second['genes'][:top_n])
        overlap = len(first_topn & second_topn)
        metrics[f'top{top_n}_overlap'] = overlap / top_n if top_n > 0 else 0.0
    
    # 3. Jaccard similarity for top-N genes by score
    for top_n in [5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200]:
        first_topn = set(first['genes'][:top_n])
        second_topn = set(second['genes'][:top_n])
        intersection = len(first_topn & second_topn)
        union = len(first_topn | second_topn)
        metrics[f'jaccard_top{top_n}'] = intersection / union if union > 0 else 0.0
    
    # 4. Create score dictionaries for correlation calculations
    first_scores = dict(zip(first['genes'], first['scores']))
    second_scores = dict(zip(second['genes'], second['scores']))
    
    # Get common genes
    common_genes = list(set(first['genes']) & set(second['genes']))
    
    if len(common_genes) > 1:
        first_scores_common = [first_scores[g] for g in common_genes]
        second_scores_common = [second_scores[g] for g in common_genes]
        
        # 5. Spearman correlation of scores (all common genes)
        try:
            corr, pval = spearmanr(first_scores_common, second_scores_common)
            metrics['score_spearman'] = corr if not np.isnan(corr) else 0.0
        except:
            metrics['score_spearman'] = 0.0
        
        # 6. Spearman correlation of top 500 genes
        top_500_genes = list(set(first['genes'][:500]) & set(second['genes'][:500]))
        if len(top_500_genes) > 1:
            first_top500 = [first_scores[g] for g in top_500_genes]
            second_top500 = [second_scores[g] for g in top_500_genes]
            try:
                corr_top500, _ = spearmanr(first_top500, second_top500)
                metrics['score_spearman_top500'] = corr_top500 if not np.isnan(corr_top500) else 0.0
            except:
                metrics['score_spearman_top500'] = 0.0
        else:
            metrics['score_spearman_top500'] = 0.0
        
        # 7. Spearman correlation for top-N genes by score
        for top_n in [5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200]:
            top_n_genes = list(set(first['genes'][:top_n]) & set(second['genes'][:top_n]))
            if len(top_n_genes) > 1:
                first_topn_scores = [first_scores[g] for g in top_n_genes]
                second_topn_scores = [second_scores[g] for g in top_n_genes]
                try:
                    corr_topn, _ = spearmanr(first_topn_scores, second_topn_scores)
                    metrics[f'spearman_top{top_n}'] = corr_topn if not np.isnan(corr_topn) else 0.0
                except:
                    metrics[f'spearman_top{top_n}'] = 0.0
            else:
                metrics[f'spearman_top{top_n}'] = 0.0
        
        # 8. Kendall's tau for rank agreement
        try:
            tau, _ = kendalltau(first_scores_common, second_scores_common)
            metrics['rank_kendall_tau'] = tau if not np.isnan(tau) else 0.0
        except:
            metrics['rank_kendall_tau'] = 0.0
        
        # 9. Effect direction agreement
        same_direction = sum(1 for i in range(len(first_scores_common)) 
                            if np.sign(first_scores_common[i]) == np.sign(second_scores_common[i]))
        metrics['effect_direction_agreement'] = same_direction / len(first_scores_common)
        
        # 10. P-value correlation (use -log10 for better visualization)
        first_pvals = dict(zip(first['genes'], first['pvals_unadj']))
        second_pvals = dict(zip(second['genes'], second['pvals_unadj']))
        
        first_logp = [-np.log10(max(first_pvals[g], 1e-300)) for g in common_genes]
        second_logp = [-np.log10(max(second_pvals[g], 1e-300)) for g in common_genes]
        
        try:
            pval_corr, _ = spearmanr(first_logp, second_logp)
            metrics['pval_correlation'] = pval_corr if not np.isnan(pval_corr) else 0.0
        except:
            metrics['pval_correlation'] = 0.0
    else:
        # Not enough common genes
        metrics['score_spearman'] = 0.0
        metrics['score_spearman_top500'] = 0.0
        metrics['rank_kendall_tau'] = 0.0
        metrics['effect_direction_agreement'] = 0.0
        metrics['pval_correlation'] = 0.0
    
    return metrics


def create_comparison_dataframe(first_half_results, second_half_results):
    """
    Create comprehensive comparison dataframe for all perturbations.
    
    Args:
        first_half_results: DEG results for first half
        second_half_results: DEG results for second half
        
    Returns:
        DataFrame with comparison metrics for all perturbations
    """
    print("\nComparing DEG results between halves...")
    
    # Get perturbations present in both halves
    common_perts = set(first_half_results.keys()) & set(second_half_results.keys())
    print(f"Found {len(common_perts)} perturbations present in both halves")
    
    # Calculate metrics for each perturbation
    comparison_data = []
    for pert in tqdm(sorted(common_perts), desc="Computing comparison metrics"):
        metrics = compare_deg_results(first_half_results, second_half_results, pert)
        if metrics is not None:
            comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(f"Successfully compared {len(comparison_df)} perturbations")
    
    return comparison_df


def visualize_reproducibility(comparison_df, output_dir):
    """
    Generate comprehensive visualizations of DEG reproducibility.
    
    Args:
        comparison_df: DataFrame with comparison metrics
        output_dir: Directory to save plots
    """
    print("\nGenerating visualizations...")
    
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to perturbations with DEGs in at least one half for most plots
    df_with_degs = comparison_df[comparison_df['has_degs_either']].copy()
    df_both_degs = comparison_df[comparison_df['has_degs_both']].copy()
    
    n_total = len(comparison_df)
    n_with_degs = len(df_with_degs)
    n_both_degs = len(df_both_degs)
    
    print(f"  Total perturbations: {n_total}")
    print(f"  With DEGs in at least one half: {n_with_degs} ({n_with_degs/n_total*100:.1f}%)")
    print(f"  With DEGs in both halves: {n_both_degs} ({n_both_degs/n_total*100:.1f}%)")
    
    # 1. Jaccard similarity vs cell count (only perturbations with DEGs in both halves)
    print("  Creating Jaccard similarity plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(df_both_degs) > 0:
        scatter = ax.scatter(df_both_degs['n_cells_total'], 
                            df_both_degs['jaccard_similarity'],
                            c=df_both_degs['n_degs_first'] + df_both_degs['n_degs_second'],
                            cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel('Total Number of Cells', fontsize=12)
        ax.set_ylabel('Jaccard Similarity (Significant DEGs)', fontsize=12)
        ax.set_title(f'DEG Reproducibility vs Cell Count\n({len(df_both_degs)} perturbations with DEGs in both halves)', 
                    fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Total DEGs (both halves)')
        
        # Add trend line if enough points
        if len(df_both_degs) > 3:
            data_clean = df_both_degs[['n_cells_total', 'jaccard_similarity']].dropna()
            if len(data_clean) > 3:
                z = np.polyfit(data_clean['n_cells_total'], data_clean['jaccard_similarity'], 2)
                p = np.poly1d(z)
                x_trend = np.linspace(data_clean['n_cells_total'].min(), 
                                     data_clean['n_cells_total'].max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
                ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No perturbations with DEGs in both halves', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xlabel('Total Number of Cells', fontsize=12)
        ax.set_ylabel('Jaccard Similarity (Significant DEGs)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'jaccard_vs_cells.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Score correlation vs cell count (use all perturbations)
    print("  Creating score correlation plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by whether they have DEGs
    colors = []
    for _, row in comparison_df.iterrows():
        if row['has_degs_both']:
            colors.append('green')
        elif row['has_degs_either']:
            colors.append('orange')
        else:
            colors.append('gray')
    
    scatter = ax.scatter(comparison_df['n_cells_total'], 
                        comparison_df['score_spearman'],
                        c=colors, alpha=0.6, s=50)
    ax.set_xlabel('Total Number of Cells', fontsize=12)
    ax.set_ylabel('Spearman Correlation (DEG Scores)', fontsize=12)
    ax.set_title(f'Score Correlation vs Cell Count (All {len(comparison_df)} perturbations)', 
                fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label=f'DEGs in both halves (n={n_both_degs})'),
        Patch(facecolor='orange', alpha=0.6, label=f'DEGs in one half (n={n_with_degs - n_both_degs})'),
        Patch(facecolor='gray', alpha=0.6, label=f'No DEGs (n={n_total - n_with_degs})')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'score_correlation_vs_cells.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top-N overlap comparison
    print("  Creating top-N overlap plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    top_n_cols = ['top10_overlap', 'top50_overlap', 'top100_overlap', 'top200_overlap']
    
    # Create cell count bins
    comparison_df['cell_bin'] = pd.cut(comparison_df['n_cells_total'], 
                                       bins=[0, 20, 50, 100, 1000],
                                       labels=['<20', '20-50', '50-100', '>100'])
    
    # Plot boxplots for each top-N metric
    positions = []
    labels = []
    data_to_plot = []
    
    for i, col in enumerate(top_n_cols):
        for j, bin_label in enumerate(['<20', '20-50', '50-100', '>100']):
            bin_data = comparison_df[comparison_df['cell_bin'] == bin_label][col].dropna()
            if len(bin_data) > 0:
                positions.append(i * 5 + j)
                data_to_plot.append(bin_data)
                if i == 0:
                    labels.append(bin_label)
    
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
    
    ax.set_xticks([1.5, 6.5, 11.5, 16.5])
    ax.set_xticklabels(['Top 10', 'Top 50', 'Top 100', 'Top 200'])
    ax.set_ylabel('Overlap Fraction', fontsize=12)
    ax.set_title('Top-N Gene Overlap by Cell Count', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightblue', alpha=0.7, label='<20 cells'),
                      Patch(facecolor='lightgreen', alpha=0.7, label='20-50 cells'),
                      Patch(facecolor='lightyellow', alpha=0.7, label='50-100 cells'),
                      Patch(facecolor='lightcoral', alpha=0.7, label='>100 cells')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'topN_overlap_vs_cells.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Distribution of reproducibility metrics
    print("  Creating metric distributions plot...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('jaccard_similarity', 'Jaccard Similarity'),
        ('score_spearman', 'Spearman Correlation'),
        ('rank_kendall_tau', 'Kendall Tau'),
        ('effect_direction_agreement', 'Direction Agreement'),
        ('pval_correlation', 'P-value Correlation'),
        ('top50_overlap', 'Top 50 Overlap')
    ]
    
    for idx, (col, title) in enumerate(metrics_to_plot):
        ax = axes[idx]
        data = comparison_df[col].dropna()
        ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(data.median(), color='red', linestyle='--', linewidth=2, 
                  label=f'Median: {data.median():.3f}')
        ax.axvline(data.mean(), color='orange', linestyle='--', linewidth=2,
                  label=f'Mean: {data.mean():.3f}')
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Distribution of {title}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'reproducibility_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Heatmap of perturbations
    print("  Creating perturbation heatmap...")
    # Select top perturbations by cell count for visualization
    n_show = min(50, len(comparison_df))
    top_perts = comparison_df.nlargest(n_show, 'n_cells_total')
    
    heatmap_data = top_perts[[
        'jaccard_similarity', 'score_spearman', 'rank_kendall_tau',
        'effect_direction_agreement', 'top10_overlap', 'top50_overlap'
    ]].T
    
    fig, ax = plt.subplots(figsize=(max(12, n_show * 0.3), 6))
    sns.heatmap(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, 
                cbar_kws={'label': 'Metric Value'},
                xticklabels=top_perts['perturbation'].tolist(),
                yticklabels=['Jaccard', 'Spearman', 'Kendall', 'Direction', 'Top10', 'Top50'],
                ax=ax)
    ax.set_title(f'Reproducibility Metrics for Top {n_show} Perturbations (by cell count)', 
                fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / 'perturbation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Cell count stratified analysis
    print("  Creating stratified analysis plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Define bins
    bins = [0, 20, 50, 100, comparison_df['n_cells_total'].max() + 1]
    labels = ['<20', '20-50', '50-100', '>100']
    comparison_df['cell_range'] = pd.cut(comparison_df['n_cells_total'], 
                                         bins=bins, labels=labels)
    
    metrics = [
        ('jaccard_similarity', 'Jaccard Similarity'),
        ('score_spearman', 'Spearman Correlation'),
        ('top50_overlap', 'Top 50 Overlap'),
        ('effect_direction_agreement', 'Direction Agreement')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Create violin plot - filter out empty bins
        data_by_bin = []
        valid_positions = []
        valid_labels = []
        
        for i, label in enumerate(labels):
            data = comparison_df[comparison_df['cell_range'] == label][metric].dropna()
            if len(data) > 0:
                data_by_bin.append(data)
                valid_positions.append(i)
                valid_labels.append(label)
        
        if len(data_by_bin) > 0:
            parts = ax.violinplot(data_by_bin, positions=valid_positions, 
                                 showmeans=True, showmedians=True)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_xlabel('Cell Count Range', fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(f'{title} by Cell Count', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add n counts for all bins
            for i, label in enumerate(labels):
                data = comparison_df[comparison_df['cell_range'] == label][metric].dropna()
                n = len(data)
                ax.text(i, -0.1, f'n={n}', ha='center', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_xlabel('Cell Count Range', fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(f'{title} by Cell Count', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'stratified_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Jaccard similarity for top-N genes (line plot showing trend)
    print("  Creating top-N Jaccard similarity plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Mean Jaccard by top-N cutoff
    top_n_values = [5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200]
    jaccard_cols = [f'jaccard_top{n}' for n in top_n_values]
    
    # Calculate mean for all perturbations
    mean_all = [comparison_df[col].mean() for col in jaccard_cols]
    # Calculate mean for perturbations with DEGs
    mean_with_degs = [df_with_degs[col].mean() for col in jaccard_cols]
    # Calculate mean for perturbations with DEGs in both
    mean_both_degs = [df_both_degs[col].mean() if len(df_both_degs) > 0 else 0 for col in jaccard_cols]
    
    ax1.plot(top_n_values, mean_all, 'o-', linewidth=2, markersize=8, 
            label=f'All perturbations (n={n_total})', color='gray')
    ax1.plot(top_n_values, mean_with_degs, 's-', linewidth=2, markersize=8,
            label=f'With DEGs in either half (n={n_with_degs})', color='orange')
    if n_both_degs > 0:
        ax1.plot(top_n_values, mean_both_degs, '^-', linewidth=2, markersize=8,
                label=f'With DEGs in both halves (n={n_both_degs})', color='green')
    
    ax1.set_xlabel('Top N Genes', fontsize=12)
    ax1.set_ylabel('Mean Jaccard Similarity', fontsize=12)
    ax1.set_title('Gene Ranking Agreement by Top-N Cutoff', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Box plots for each top-N cutoff (using perturbations with DEGs)
    if len(df_with_degs) > 0:
        data_for_box = [df_with_degs[col].dropna() for col in jaccard_cols]
        bp = ax2.boxplot(data_for_box, positions=range(len(top_n_values)),
                        widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax2.set_xticks(range(len(top_n_values)))
        ax2.set_xticklabels([str(n) for n in top_n_values])
        ax2.set_xlabel('Top N Genes', fontsize=12)
        ax2.set_ylabel('Jaccard Similarity', fontsize=12)
        ax2.set_title(f'Distribution Across Perturbations (n={len(df_with_degs)})',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, 'No perturbations with DEGs',
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel('Top N Genes', fontsize=12)
        ax2.set_ylabel('Jaccard Similarity', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'jaccard_topN_genes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Top-N Jaccard vs cell count
    print("  Creating top-N Jaccard vs cell count plot...")
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    axes = axes.flatten()
    
    top_n_values = [5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200]
    jaccard_cols = [f'jaccard_top{n}' for n in top_n_values]
    
    for idx, (col, top_n) in enumerate(zip(jaccard_cols, top_n_values)):
        if idx < len(axes):
            ax = axes[idx]
            
            # Color by whether they have DEGs
            colors = []
            for _, row in comparison_df.iterrows():
                if row['has_degs_both']:
                    colors.append('green')
                elif row['has_degs_either']:
                    colors.append('orange')
                else:
                    colors.append('gray')
            
            scatter = ax.scatter(comparison_df['n_cells_total'], 
                                comparison_df[col],
                                c=colors, alpha=0.5, s=30)
            
            ax.set_xlabel('Total Number of Cells', fontsize=10)
            ax.set_ylabel(f'Jaccard Similarity (Top {top_n})', fontsize=10)
            ax.set_title(f'Top {top_n} Genes', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            
            # Add correlation for perturbations with DEGs
            if len(df_with_degs) > 2:
                data_clean = df_with_degs[['n_cells_total', col]].dropna()
                if len(data_clean) > 2:
                    corr, pval = spearmanr(data_clean['n_cells_total'], data_clean[col])
                    ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.1e}', 
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Use the last two subplots for legend and info
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label=f'DEGs in both halves (n={n_both_degs})'),
        Patch(facecolor='orange', alpha=0.6, label=f'DEGs in one half (n={n_with_degs - n_both_degs})'),
        Patch(facecolor='gray', alpha=0.6, label=f'No DEGs (n={n_total - n_with_degs})')
    ]
    axes[-2].legend(handles=legend_elements, loc='center', fontsize=11, frameon=True)
    axes[-2].axis('off')
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'jaccard_topN_vs_cells.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Spearman correlation for top-N genes (line plot showing trend)
    print("  Creating top-N Spearman correlation plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Mean Spearman by top-N cutoff
    top_n_values = [5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200]
    spearman_cols = [f'spearman_top{n}' for n in top_n_values]
    
    # Calculate mean for all perturbations
    mean_all = [comparison_df[col].mean() for col in spearman_cols]
    # Calculate mean for perturbations with DEGs
    mean_with_degs = [df_with_degs[col].mean() for col in spearman_cols]
    # Calculate mean for perturbations with DEGs in both
    mean_both_degs = [df_both_degs[col].mean() if len(df_both_degs) > 0 else 0 for col in spearman_cols]
    
    ax1.plot(top_n_values, mean_all, 'o-', linewidth=2, markersize=8, 
            label=f'All perturbations (n={n_total})', color='gray')
    ax1.plot(top_n_values, mean_with_degs, 's-', linewidth=2, markersize=8,
            label=f'With DEGs in either half (n={n_with_degs})', color='orange')
    if n_both_degs > 0:
        ax1.plot(top_n_values, mean_both_degs, '^-', linewidth=2, markersize=8,
                label=f'With DEGs in both halves (n={n_both_degs})', color='green')
    
    ax1.set_xlabel('Top N Genes', fontsize=12)
    ax1.set_ylabel('Mean Spearman Correlation', fontsize=12)
    ax1.set_title('Score Correlation Agreement by Top-N Cutoff', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Box plots for each top-N cutoff (using perturbations with DEGs)
    if len(df_with_degs) > 0:
        data_for_box = [df_with_degs[col].dropna() for col in spearman_cols]
        bp = ax2.boxplot(data_for_box, positions=range(len(top_n_values)),
                        widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax2.set_xticks(range(len(top_n_values)))
        ax2.set_xticklabels([str(n) for n in top_n_values])
        ax2.set_xlabel('Top N Genes', fontsize=12)
        ax2.set_ylabel('Spearman Correlation', fontsize=12)
        ax2.set_title(f'Distribution Across Perturbations (n={len(df_with_degs)})',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(-0.1, 1)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    else:
        ax2.text(0.5, 0.5, 'No perturbations with DEGs',
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel('Top N Genes', fontsize=12)
        ax2.set_ylabel('Spearman Correlation', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'spearman_topN_genes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. Top-N Spearman vs cell count
    print("  Creating top-N Spearman vs cell count plot...")
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    axes = axes.flatten()
    
    top_n_values = [5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200]
    spearman_cols = [f'spearman_top{n}' for n in top_n_values]
    
    for idx, (col, top_n) in enumerate(zip(spearman_cols, top_n_values)):
        if idx < len(axes):
            ax = axes[idx]
            
            # Color by whether they have DEGs
            colors = []
            for _, row in comparison_df.iterrows():
                if row['has_degs_both']:
                    colors.append('green')
                elif row['has_degs_either']:
                    colors.append('orange')
                else:
                    colors.append('gray')
            
            scatter = ax.scatter(comparison_df['n_cells_total'], 
                                comparison_df[col],
                                c=colors, alpha=0.5, s=30)
            
            ax.set_xlabel('Total Number of Cells', fontsize=10)
            ax.set_ylabel(f'Spearman Corr (Top {top_n})', fontsize=10)
            ax.set_title(f'Top {top_n} Genes', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.05)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            
            # Add correlation for perturbations with DEGs
            if len(df_with_degs) > 2:
                data_clean = df_with_degs[['n_cells_total', col]].dropna()
                if len(data_clean) > 2:
                    corr, pval = spearmanr(data_clean['n_cells_total'], data_clean[col])
                    ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.1e}', 
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Use the last two subplots for legend and info
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label=f'DEGs in both halves (n={n_both_degs})'),
        Patch(facecolor='orange', alpha=0.6, label=f'DEGs in one half (n={n_with_degs - n_both_degs})'),
        Patch(facecolor='gray', alpha=0.6, label=f'No DEGs (n={n_total - n_with_degs})')
    ]
    axes[-2].legend(handles=legend_elements, loc='center', fontsize=11, frameon=True)
    axes[-2].axis('off')
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'spearman_topN_vs_cells.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All plots saved to {plots_dir}")


def generate_summary_report(comparison_df, output_dir):
    """
    Generate a text summary report of the reproducibility analysis.
    
    Args:
        comparison_df: DataFrame with comparison metrics
        output_dir: Directory to save report
    """
    print("\nGenerating summary report...")
    
    report_path = Path(output_dir) / 'deg_reproducibility_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEG REPRODUCIBILITY ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total perturbations analyzed: {len(comparison_df)}\n")
        f.write(f"Cell count range: {comparison_df['n_cells_total'].min():.0f} - "
                f"{comparison_df['n_cells_total'].max():.0f}\n")
        f.write(f"Mean cells per perturbation: {comparison_df['n_cells_total'].mean():.1f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("DEG SIGNAL DETECTION\n")
        f.write("-"*80 + "\n\n")
        
        n_has_both = comparison_df['has_degs_both'].sum()
        n_has_either = comparison_df['has_degs_either'].sum()
        n_has_none = (~comparison_df['has_degs_either']).sum()
        n_has_only_one = comparison_df['has_degs_either'].sum() - comparison_df['has_degs_both'].sum()
        
        f.write(f"Perturbations with DEGs in BOTH halves:   {n_has_both:4d} ({n_has_both/len(comparison_df)*100:5.1f}%)\n")
        f.write(f"Perturbations with DEGs in EITHER half:   {n_has_either:4d} ({n_has_either/len(comparison_df)*100:5.1f}%)\n")
        f.write(f"Perturbations with DEGs in only ONE half: {n_has_only_one:4d} ({n_has_only_one/len(comparison_df)*100:5.1f}%)\n")
        f.write(f"Perturbations with NO DEGs in either:     {n_has_none:4d} ({n_has_none/len(comparison_df)*100:5.1f}%)\n\n")
        
        f.write("Note: Reproducibility metrics are only meaningful for perturbations with\n")
        f.write("      DEGs in at least one half. Jaccard similarity is set to NaN when both\n")
        f.write("      halves have zero DEGs (no signal to measure).\n\n")
        
        f.write("-"*80 + "\n")
        f.write("OVERALL REPRODUCIBILITY METRICS (All Perturbations with DEGs)\n")
        f.write("-"*80 + "\n\n")
        
        # Filter to only perturbations with DEGs in at least one half
        df_with_degs = comparison_df[comparison_df['has_degs_either']].copy()
        f.write(f"Analyzing {len(df_with_degs)} perturbations with DEGs in at least one half\n\n")
        
        metrics = [
            ('jaccard_similarity', 'Jaccard Similarity (Significant DEGs)'),
            ('score_spearman', 'Spearman Correlation (Scores)'),
            ('rank_kendall_tau', 'Kendall Tau (Ranks)'),
            ('effect_direction_agreement', 'Effect Direction Agreement'),
            ('pval_correlation', 'P-value Correlation'),
            ('top10_overlap', 'Top 10 Gene Overlap (proportion)'),
            ('top50_overlap', 'Top 50 Gene Overlap (proportion)'),
            ('top100_overlap', 'Top 100 Gene Overlap (proportion)'),
            ('top200_overlap', 'Top 200 Gene Overlap (proportion)'),
            ('jaccard_top5', 'Jaccard Top 5 Genes'),
            ('jaccard_top10', 'Jaccard Top 10 Genes'),
            ('jaccard_top25', 'Jaccard Top 25 Genes'),
            ('jaccard_top50', 'Jaccard Top 50 Genes'),
            ('jaccard_top100', 'Jaccard Top 100 Genes'),
            ('jaccard_top200', 'Jaccard Top 200 Genes'),
            ('jaccard_top400', 'Jaccard Top 400 Genes'),
            ('jaccard_top800', 'Jaccard Top 800 Genes'),
            ('jaccard_top1600', 'Jaccard Top 1600 Genes'),
            ('jaccard_top3200', 'Jaccard Top 3200 Genes'),
            ('spearman_top5', 'Spearman Top 5 Genes'),
            ('spearman_top10', 'Spearman Top 10 Genes'),
            ('spearman_top25', 'Spearman Top 25 Genes'),
            ('spearman_top50', 'Spearman Top 50 Genes'),
            ('spearman_top100', 'Spearman Top 100 Genes'),
            ('spearman_top200', 'Spearman Top 200 Genes'),
            ('spearman_top400', 'Spearman Top 400 Genes'),
            ('spearman_top800', 'Spearman Top 800 Genes'),
            ('spearman_top1600', 'Spearman Top 1600 Genes'),
            ('spearman_top3200', 'Spearman Top 3200 Genes')
        ]
        
        for col, name in metrics:
            data = df_with_degs[col].dropna()
            if len(data) > 0:
                f.write(f"{name}:\n")
                f.write(f"  N:      {len(data)}\n")
                f.write(f"  Mean:   {data.mean():.4f}\n")
                f.write(f"  Median: {data.median():.4f}\n")
                f.write(f"  Std:    {data.std():.4f}\n")
                f.write(f"  Min:    {data.min():.4f}\n")
                f.write(f"  Max:    {data.max():.4f}\n\n")
            else:
                f.write(f"{name}: No data available\n\n")
        
        # Separate analysis for perturbations with DEGs in BOTH halves
        f.write("-"*80 + "\n")
        f.write("REPRODUCIBILITY METRICS (Only Perturbations with DEGs in BOTH Halves)\n")
        f.write("-"*80 + "\n\n")
        
        df_both_degs = comparison_df[comparison_df['has_degs_both']].copy()
        f.write(f"Analyzing {len(df_both_degs)} perturbations with DEGs in both halves\n\n")
        
        if len(df_both_degs) > 0:
            for col, name in metrics:
                data = df_both_degs[col].dropna()
                if len(data) > 0:
                    f.write(f"{name}:\n")
                    f.write(f"  N:      {len(data)}\n")
                    f.write(f"  Mean:   {data.mean():.4f}\n")
                    f.write(f"  Median: {data.median():.4f}\n")
                    f.write(f"  Std:    {data.std():.4f}\n")
                    f.write(f"  Min:    {data.min():.4f}\n")
                    f.write(f"  Max:    {data.max():.4f}\n\n")
        else:
            f.write("No perturbations with DEGs in both halves.\n\n")
        
        f.write("-"*80 + "\n")
        f.write("TOP-N GENE RANKING AGREEMENT\n")
        f.write("-"*80 + "\n\n")
        
        top_n_jaccard = ['jaccard_top5', 'jaccard_top10', 'jaccard_top25', 'jaccard_top50', 
                        'jaccard_top100', 'jaccard_top200', 'jaccard_top400', 'jaccard_top800',
                        'jaccard_top1600', 'jaccard_top3200']
        top_n_spearman = ['spearman_top5', 'spearman_top10', 'spearman_top25', 'spearman_top50', 
                         'spearman_top100', 'spearman_top200', 'spearman_top400', 'spearman_top800',
                         'spearman_top1600', 'spearman_top3200']
        top_n_labels = ['Top 5', 'Top 10', 'Top 25', 'Top 50', 'Top 100', 'Top 200', 'Top 400',
                       'Top 800', 'Top 1600', 'Top 3200']
        
        f.write("Jaccard similarity for top-N genes ranked by score:\n\n")
        for col, label in zip(top_n_jaccard, top_n_labels):
            data = df_with_degs[col].dropna()
            if len(data) > 0:
                f.write(f"{label:12s}: Mean={data.mean():.4f}, "
                       f"Median={data.median():.4f}, Std={data.std():.4f}\n")
        
        f.write("\n")
        
        f.write("Spearman correlation for top-N genes ranked by score:\n\n")
        for col, label in zip(top_n_spearman, top_n_labels):
            data = df_with_degs[col].dropna()
            if len(data) > 0:
                f.write(f"{label:12s}: Mean={data.mean():.4f}, "
                       f"Median={data.median():.4f}, Std={data.std():.4f}\n")
        
        f.write("\n")
        
        f.write("-"*80 + "\n")
        f.write("CORRELATION WITH CELL COUNT (Perturbations with DEGs)\n")
        f.write("-"*80 + "\n\n")
        
        for col, name in metrics:
            data = df_with_degs[[col, 'n_cells_total']].dropna()
            if len(data) > 2:
                corr, pval = spearmanr(data['n_cells_total'], data[col])
                f.write(f"{name} vs Cell Count:\n")
                f.write(f"  N:          {len(data)}\n")
                f.write(f"  Spearman r: {corr:.4f}\n")
                f.write(f"  P-value:    {pval:.4e}\n\n")
        
        f.write("\nTop-N Gene Jaccard Correlations with Cell Count:\n\n")
        for col, label in zip(top_n_jaccard, top_n_labels):
            data = df_with_degs[[col, 'n_cells_total']].dropna()
            if len(data) > 2:
                corr, pval = spearmanr(data['n_cells_total'], data[col])
                f.write(f"{label:12s}: r={corr:7.4f}, p={pval:.4e}\n")
        
        f.write("\nTop-N Gene Spearman Correlations with Cell Count:\n\n")
        for col, label in zip(top_n_spearman, top_n_labels):
            data = df_with_degs[[col, 'n_cells_total']].dropna()
            if len(data) > 2:
                corr, pval = spearmanr(data['n_cells_total'], data[col])
                f.write(f"{label:12s}: r={corr:7.4f}, p={pval:.4e}\n")
        
        f.write("-"*80 + "\n")
        f.write("STRATIFIED ANALYSIS BY CELL COUNT\n")
        f.write("-"*80 + "\n\n")
        
        bins = [0, 20, 50, 100, comparison_df['n_cells_total'].max() + 1]
        labels = ['<20 cells', '20-50 cells', '50-100 cells', '>100 cells']
        comparison_df['cell_range'] = pd.cut(comparison_df['n_cells_total'], 
                                             bins=bins, labels=labels)
        
        for label in labels:
            subset = comparison_df[comparison_df['cell_range'] == label]
            if len(subset) == 0:
                continue
            
            f.write(f"\n{label} (n={len(subset)}):\n")
            f.write(f"  Jaccard Similarity:    {subset['jaccard_similarity'].mean():.4f} ± "
                   f"{subset['jaccard_similarity'].std():.4f}\n")
            f.write(f"  Score Correlation:     {subset['score_spearman'].mean():.4f} ± "
                   f"{subset['score_spearman'].std():.4f}\n")
            f.write(f"  Top 50 Overlap:        {subset['top50_overlap'].mean():.4f} ± "
                   f"{subset['top50_overlap'].std():.4f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("TOP 10 MOST REPRODUCIBLE PERTURBATIONS (with DEGs in both halves)\n")
        f.write("-"*80 + "\n\n")
        
        if len(df_both_degs) > 0:
            top_10 = df_both_degs.nlargest(min(10, len(df_both_degs)), 'jaccard_similarity')
            for idx, row in top_10.iterrows():
                f.write(f"{row['perturbation']}:\n")
                f.write(f"  Cells: {row['n_cells_total']:.0f}, "
                       f"DEGs: {row['n_degs_first']:.0f}/{row['n_degs_second']:.0f}, "
                       f"Jaccard: {row['jaccard_similarity']:.4f}, "
                       f"Spearman: {row['score_spearman']:.4f}\n\n")
        else:
            f.write("No perturbations with DEGs in both halves.\n\n")
        
        f.write("-"*80 + "\n")
        f.write("TOP 10 LEAST REPRODUCIBLE PERTURBATIONS (with DEGs in both halves)\n")
        f.write("-"*80 + "\n\n")
        
        if len(df_both_degs) > 0:
            bottom_10 = df_both_degs.nsmallest(min(10, len(df_both_degs)), 'jaccard_similarity')
            for idx, row in bottom_10.iterrows():
                f.write(f"{row['perturbation']}:\n")
                f.write(f"  Cells: {row['n_cells_total']:.0f}, "
                       f"DEGs: {row['n_degs_first']:.0f}/{row['n_degs_second']:.0f}, "
                       f"Jaccard: {row['jaccard_similarity']:.4f}, "
                       f"Spearman: {row['score_spearman']:.4f}\n\n")
        else:
            f.write("No perturbations with DEGs in both halves.\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Summary report saved to {report_path}")


def main():
    """Main analysis workflow"""
    parser = argparse.ArgumentParser(
        description='Analyze DEG reproducibility between technical duplicate splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis with caching (default)
  python data/analyze_deg_reproducibility.py --input data/dataset_processed.h5ad
  
  # Force recalculation of DEGs
  python data/analyze_deg_reproducibility.py --input data/dataset_processed.h5ad --recalculate
  
  # Use cached DEGs but regenerate plots
  python data/analyze_deg_reproducibility.py --input data/dataset_processed.h5ad --plots-only
        """
    )
    parser.add_argument('--input', required=True, type=str,
                       help='Path to processed h5ad file with tech_dup_split column')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: same directory as input with _reproducibility suffix)')
    parser.add_argument('--min-cells', type=int, default=4,
                       help='Minimum cells required per perturbation per half (default: 4)')
    parser.add_argument('--recalculate', action='store_true',
                       help='Force recalculation of DEGs even if cache exists')
    parser.add_argument('--plots-only', action='store_true',
                       help='Skip DEG calculation and only regenerate plots from cached results')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    # Set output directory
    if args.output is None:
        output_dir = input_path.parent / f"{input_path.stem}_reproducibility"
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("DEG REPRODUCIBILITY ANALYSIS")
    print("="*80)
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Minimum cells per half: {args.min_cells}")
    print("="*80)
    
    # Generate input hash for cache validation
    input_hash = get_input_hash(input_path, args.min_cells)
    
    # Try to load cached results first (before loading AnnData!)
    first_half_results = None
    second_half_results = None
    
    if not args.recalculate:
        print("\nChecking for cached DEG results...")
        first_half_results, second_half_results = load_cached_deg_results(
            output_dir, input_hash, args.min_cells
        )
        
        if first_half_results is not None and second_half_results is not None:
            print("✓ Using cached DEG results (skipping AnnData loading)")
        elif args.plots_only:
            print("ERROR: No cached results found. Cannot run in --plots-only mode.")
            print("Please run without --plots-only first to calculate DEGs.")
            sys.exit(1)
    
    # Only load AnnData if we need to calculate DEGs
    if first_half_results is None or second_half_results is None:
        if args.recalculate:
            print("\nRecalculating DEGs (--recalculate flag set)...")
        else:
            print("\nNo valid cache found. Calculating DEGs...")
        
        # Load data
        print("Loading AnnData...")
        adata = sc.read_h5ad(input_path)
        print(f"Loaded AnnData: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        # Validate tech_dup_split column
        if 'tech_dup_split' not in adata.obs.columns:
            print("ERROR: 'tech_dup_split' column not found in adata.obs")
            print("Available columns:", adata.obs.columns.tolist())
            sys.exit(1)
        
        split_counts = adata.obs['tech_dup_split'].value_counts()
        print(f"\nTechnical duplicate split distribution:")
        for split, count in split_counts.items():
            print(f"  {split}: {count} cells")
        
        if 'first_half' not in split_counts or 'second_half' not in split_counts:
            print("ERROR: Both 'first_half' and 'second_half' must be present in tech_dup_split")
            sys.exit(1)
        
        # Get dataset name
        dataset_name = None
        if 'donor_id' in adata.obs.columns:
            dataset_name = adata.obs['donor_id'].iloc[0]
        
        # Calculate DEGs for both halves
        first_half_results = calculate_degs_by_split(adata, 'first_half', 
                                                      min_cells=args.min_cells,
                                                      dataset_name=dataset_name)
        second_half_results = calculate_degs_by_split(adata, 'second_half', 
                                                       min_cells=args.min_cells,
                                                       dataset_name=dataset_name)
        
        if len(first_half_results) == 0 or len(second_half_results) == 0:
            print("\nERROR: No valid perturbations found in one or both halves.")
            print("Try reducing --min-cells parameter.")
            sys.exit(1)
        
        # Save results to cache
        save_deg_results(first_half_results, second_half_results, 
                        output_dir, input_hash, args.min_cells)
    
    # Validate loaded results
    if len(first_half_results) == 0 or len(second_half_results) == 0:
        print("\nERROR: No valid perturbations found in cached results.")
        sys.exit(1)
    
    # Create comparison dataframe
    comparison_df = create_comparison_dataframe(first_half_results, second_half_results)
    
    if len(comparison_df) == 0:
        print("\nERROR: No perturbations could be compared between halves.")
        sys.exit(1)
    
    # Save comparison data
    output_csv = output_dir / 'deg_reproducibility_metrics.csv'
    comparison_df.to_csv(output_csv, index=False)
    print(f"\nComparison metrics saved to {output_csv}")
    
    # Generate visualizations
    visualize_reproducibility(comparison_df, output_dir)
    
    # Generate summary report
    generate_summary_report(comparison_df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Metrics CSV: deg_reproducibility_metrics.csv")
    print(f"  - Summary report: deg_reproducibility_summary.txt")
    print(f"  - Plots: plots/")
    print(f"      • jaccard_vs_cells.png - Jaccard similarity vs cell count")
    print(f"      • score_correlation_vs_cells.png - Score correlation vs cell count")
    print(f"      • topN_overlap_vs_cells.png - Top-N overlap comparison")
    print(f"      • reproducibility_distributions.png - Metric distributions")
    print(f"      • perturbation_heatmap.png - Heatmap of top perturbations")
    print(f"      • stratified_analysis.png - Metrics stratified by cell count")
    print(f"      • jaccard_topN_genes.png - Jaccard for top-N genes")
    print(f"      • jaccard_topN_vs_cells.png - Jaccard top-N vs cell count")
    print(f"      • spearman_topN_genes.png - Spearman for top-N genes")
    print(f"      • spearman_topN_vs_cells.png - Spearman top-N vs cell count")
    print(f"  - Cached DEGs: cache/")
    print(f"\nNote: Cached DEG results will be reused on subsequent runs.")
    print(f"      Use --recalculate to force recalculation or --plots-only to regenerate plots.")
    print("\n")


if __name__ == '__main__':
    main()

