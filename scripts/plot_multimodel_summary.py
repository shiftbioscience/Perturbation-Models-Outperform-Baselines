#!/usr/bin/env python3
"""
Generate beautiful summary plots for multi-model benchmarking results.

Creates:
1. Heatmap of model × metric with annotated mean values
2. Table with mean ± SEM for all models/metrics
3. Z-score heatmap showing relative performance vs. mean

Usage:
    python scripts/plot_multimodel_summary.py <path_to_detailed_metrics.csv>
    
Example:
    python scripts/plot_multimodel_summary.py outputs/benchmark_*/2025-*/detailed_metrics.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import argparse
from scipy.stats import ttest_rel, bootstrap
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


# Metric display names and whether higher is better
METRIC_INFO = {
    'mse': {'display': 'MSE', 'higher_better': False},
    'wmse': {'display': 'WMSE', 'higher_better': False},
    'pearson_deltactrl': {'display': 'Pearson(Δ Ctrl)', 'higher_better': True},
    'pearson_deltactrl_degs': {'display': 'Pearson(Δ Ctrl DEG)', 'higher_better': True},
    'pearson_deltapert': {'display': 'Pearson(Δ Pert)', 'higher_better': True},
    'pearson_deltapert_degs': {'display': 'Pearson(Δ Pert DEG)', 'higher_better': True},
    'r2_deltactrl': {'display': 'R²(Δ Ctrl)', 'higher_better': True},
    'r2_deltactrl_degs': {'display': 'R²(Δ Ctrl DEG)', 'higher_better': True},
    'r2_deltapert': {'display': 'R²(Δ Pert)', 'higher_better': True},
    'r2_deltapert_degs': {'display': 'R²(Δ Pert DEG)', 'higher_better': True},
    'weighted_r2_deltactrl': {'display': 'WR²(Δ Ctrl)', 'higher_better': True},
    'weighted_r2_deltapert': {'display': 'WR²(Δ Pert)', 'higher_better': True},
    'nir': {'display': 'NIR', 'higher_better': True},
}


def format_model_name(model_name: str) -> str:
    """Transform model names to nice display format."""
    # Define nice names for common models and baselines
    name_map = {
        # Baselines
        'dataset_mean': 'Dataset Mean',
        'control_mean': 'Control Mean',
        'technical_duplicate': 'Technical Duplicate',
        'sparse_mean': 'Sparse Mean',
        'interpolated_duplicate': 'Interpolated Duplicate',
        'additive': 'Additive',
        'linear': 'Linear',
        'ground_truth': 'Ground Truth',
        'baselines': 'Baselines',
        # Models
        'presage': 'PRESAGE',
        'sclambda': 'scLambda',
        'scgpt': 'scGPT',
        'fmlp_genept': 'fMLP-GenePT',
        'fmlp_esm2': 'fMLP-ESM2',
        'fmlp_geneformer': 'fMLP-Geneformer',
        'fmlp_scgpt': 'fMLP-scGPT',
        'gears': 'GEARS',
    }
    
    return name_map.get(model_name, model_name)


def load_and_process_data(csv_path: Path, exclude_baselines: bool = False) -> pd.DataFrame:
    """Load detailed metrics CSV and calculate summary statistics."""
    df = pd.read_csv(csv_path)
    
    # Always exclude sparse_mean
    df = df[df['model'] != 'sparse_mean']
    
    # Optionally exclude baseline models to focus on trained models
    if exclude_baselines:
        baseline_keywords = ['control_mean', 'technical_duplicate',
                           'interpolated_duplicate', 'additive', 'dataset_mean', 
                           'linear', 'baselines', 'ground_truth']
        df = df[~df['model'].isin(baseline_keywords)]
    
    # Calculate mean and SEM for each model-metric combination
    summary = df.groupby(['model', 'metric'])['value'].agg(['mean', 'sem', 'count']).reset_index()
    
    return summary


def create_mean_heatmap(summary_df: pd.DataFrame, output_path: Path, figsize=(16, 10)):
    """Create heatmap of model × metric with annotated mean values."""
    # Remove sparse_mean
    summary_df = summary_df[summary_df['model'] != 'sparse_mean']
    
    # Pivot to get model × metric matrix
    pivot_df = summary_df.pivot(index='model', columns='metric', values='mean')
    
    # Reorder columns to match METRIC_INFO order
    available_metrics = [m for m in METRIC_INFO.keys() if m in pivot_df.columns]
    pivot_df = pivot_df[available_metrics]
    
    # Sort rows (models) by mean performance across all metrics
    # For each model, calculate average z-score (accounting for direction)
    # Treat NaN as 0 for sorting purposes
    model_scores = []
    for model in pivot_df.index:
        zscores = []
        for metric in available_metrics:
            val = pivot_df.loc[model, metric]
            if pd.isna(val):
                val = 0  # Treat NaN as 0
            metric_mean = pivot_df[metric].fillna(0).mean()  # Treat NaN as 0 in mean calculation
            metric_std = pivot_df[metric].fillna(0).std()
            if metric_std > 0:
                zscore = (val - metric_mean) / metric_std
                # Flip for "lower is better" metrics
                if not METRIC_INFO[metric]['higher_better']:
                    zscore = -zscore
                zscores.append(zscore)
        model_scores.append(np.mean(zscores) if zscores else 0)
    
    # Sort models by average z-score (best to worst)
    model_order = [m for _, m in sorted(zip(model_scores, pivot_df.index), reverse=True)]
    pivot_df = pivot_df.loc[model_order]
    
    # Sort columns (metrics) by mean value (treat NaN as 0)
    metric_means = pivot_df.fillna(0).mean(axis=0)
    sorted_metrics = metric_means.sort_values(ascending=False).index.tolist()
    pivot_df = pivot_df[sorted_metrics]
    
    # Rename row indices (models) to display names
    pivot_df.index = [format_model_name(m) for m in pivot_df.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with annotations
    heatmap = sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Metric Value'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        annot_kws={'size': 11, 'weight': 'bold'}
    )
    
    # Set colorbar label size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Metric Value', fontsize=14, fontweight='bold')
    
    ax.set_title('Multi-Model Benchmark Results: Mean Metric Values', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Models', fontsize=16, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=13)
    
    # Style y-axis labels: italicize and color baselines differently
    # Use formatted baseline names since we renamed the indices
    formatted_baseline_names = ['Control Mean', 'Technical Duplicate', 'Sparse Mean',
                                'Interpolated Duplicate', 'Additive', 'Dataset Mean',
                                'Linear', 'Baselines', 'Ground Truth']
    
    yticklabels = ax.get_yticklabels()
    for label in yticklabels:
        if label.get_text() in formatted_baseline_names:
            label.set_fontstyle('italic')
            label.set_color('#666666')  # Gray color for baselines
            label.set_fontsize(13)
        else:
            label.set_fontweight('bold')
            label.set_color('black')
            label.set_fontsize(13)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved mean heatmap to: {output_path}")


def create_zscore_heatmap(summary_df: pd.DataFrame, output_path: Path, figsize=(16, 10)):
    """Create heatmap showing performance relative to dataset_mean baseline."""
    # Remove sparse_mean
    summary_df = summary_df[summary_df['model'] != 'sparse_mean']
    
    # Pivot to get model × metric matrix
    pivot_df = summary_df.pivot(index='model', columns='metric', values='mean')
    
    # Reorder columns to match METRIC_INFO order
    available_metrics = [m for m in METRIC_INFO.keys() if m in pivot_df.columns]
    pivot_df = pivot_df[available_metrics]
    
    # Determine which baseline to use for each metric
    # deltapert metrics: control_mean is the null baseline
    # other metrics: dataset_mean is the null baseline
    
    # Calculate DRF (Discriminatory Reliability Factor) vs. appropriate baseline
    # DRF = fraction of distance from baseline to perfect performance
    relative_df = pivot_df.copy()
    for col in relative_df.columns:
        # Choose baseline based on metric type
        if 'deltapert' in col:
            # For deltapert metrics, use control_mean as null
            if 'control_mean' in pivot_df.index:
                baseline_val = pivot_df.loc['control_mean', col]
            else:
                print(f"Warning: control_mean not found for {col}, using dataset_mean")
                baseline_val = pivot_df.loc['dataset_mean', col] if 'dataset_mean' in pivot_df.index else pivot_df[col].mean()
        else:
            # For other metrics, use dataset_mean as null
            if 'dataset_mean' in pivot_df.index:
                baseline_val = pivot_df.loc['dataset_mean', col]
            else:
                print(f"Warning: dataset_mean not found for {col}, using mean")
                baseline_val = pivot_df[col].mean()
        
        if METRIC_INFO[col]['higher_better']:
            # For "higher is better" metrics (perfect = 1.0)
            # DRF = (model - baseline) / (1.0 - baseline)
            perfect_val = 1.0
            denominator = perfect_val - baseline_val
            if abs(denominator) > 1e-6:
                relative_df[col] = (pivot_df[col] - baseline_val) / denominator
            else:
                # Baseline already at perfect, just show difference
                relative_df[col] = pivot_df[col] - baseline_val
        else:
            # For "lower is better" metrics (perfect = 0.0)
            # DRF = (baseline - model) / (baseline - 0.0)
            perfect_val = 0.0
            denominator = baseline_val - perfect_val
            if abs(denominator) > 1e-6:
                relative_df[col] = (baseline_val - pivot_df[col]) / denominator
            else:
                # Baseline already at perfect, just show difference
                relative_df[col] = baseline_val - pivot_df[col]
        
        # Clip to reasonable range: DRF typically in [-1, 2]
        # DRF = 0: same as baseline
        # DRF = 1: perfect performance
        # DRF < 0: worse than baseline
        # DRF > 1: better than "perfect" (rare, usually means overfitting)
        relative_df[col] = relative_df[col].clip(-1, 2)
    
    # Sort rows (models) by mean relative improvement (best to worst)
    # Treat NaN as 0 for sorting
    model_mean_improvement = relative_df.fillna(0).mean(axis=1).sort_values(ascending=False)
    relative_df = relative_df.loc[model_mean_improvement.index]
    
    # Sort columns (metrics) by mean improvement across all models
    # Treat NaN as 0 for sorting
    metric_mean_improvement = relative_df.fillna(0).mean(axis=0).sort_values(ascending=False)
    relative_df = relative_df[metric_mean_improvement.index]
    
    # Rename columns to display names (after sorting)
    column_mapping = {m: METRIC_INFO[m]['display'] for m in available_metrics if m in relative_df.columns}
    relative_df.columns = [column_mapping.get(c, c) for c in relative_df.columns]
    
    # Rename row indices (models) to display names
    relative_df.index = [format_model_name(m) for m in relative_df.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create diverging heatmap (positive = better than dataset_mean)
    heatmap = sns.heatmap(
        relative_df,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',  # Red = worse, Blue = better
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'DRF: Distance to Perfect\n0=Baseline | 1=Perfect'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        annot_kws={'size': 11, 'weight': 'bold'}
    )
    
    # Set colorbar label size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('DRF: Distance to Perfect\n0=Baseline | 1=Perfect', fontsize=14, fontweight='bold')
    
    ax.set_title('Multi-Model Benchmark: DRF vs. Null Baseline\n(dataset_mean for most metrics, control_mean for Δpert metrics)', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Models', fontsize=16, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=13)
    
    # Style y-axis labels: italicize and color baselines differently
    # Use formatted baseline names since we renamed the indices
    formatted_baseline_names = ['Control Mean', 'Technical Duplicate', 'Sparse Mean',
                                'Interpolated Duplicate', 'Additive', 'Dataset Mean',
                                'Linear', 'Baselines', 'Ground Truth']
    
    yticklabels = ax.get_yticklabels()
    for label in yticklabels:
        if label.get_text() in formatted_baseline_names:
            label.set_fontstyle('italic')
            label.set_color('#666666')  # Gray color for baselines
            label.set_fontsize(13)
        else:
            label.set_fontweight('bold')
            label.set_color('black')
            label.set_fontsize(13)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved relative performance heatmap to: {output_path}")


def create_summary_table(summary_df: pd.DataFrame, output_path: Path):
    """Create table with mean ± SEM for all models and metrics."""
    # Pivot to get model × metric matrix for mean
    mean_df = summary_df.pivot(index='model', columns='metric', values='mean')
    sem_df = summary_df.pivot(index='model', columns='metric', values='sem')
    
    # Reorder columns to match METRIC_INFO order
    available_metrics = [m for m in METRIC_INFO.keys() if m in mean_df.columns]
    mean_df = mean_df[available_metrics]
    sem_df = sem_df[available_metrics]
    
    # Create formatted table with mean ± SEM
    table_data = []
    for model in mean_df.index:
        row = {'Model': model}
        for metric in available_metrics:
            mean_val = mean_df.loc[model, metric]
            sem_val = sem_df.loc[model, metric]
            # Format as "mean ± sem"
            row[METRIC_INFO[metric]['display']] = f"{mean_val:.4f} ± {sem_val:.4f}"
        table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    
    # Save as CSV
    table_df.to_csv(output_path, index=False)
    print(f"✓ Saved summary table to: {output_path}")
    
    # Also create a visual table figure
    fig, ax = plt.subplots(figsize=(20, len(table_df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.15] + [0.12] * (len(table_df.columns) - 1)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_df) + 1):
        for j in range(len(table_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Multi-Model Benchmark: Summary Statistics (Mean ± SEM)', 
              fontsize=14, fontweight='bold', pad=20)
    
    table_fig_path = output_path.parent / output_path.name.replace('.csv', '_figure.png')
    plt.savefig(table_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved summary table figure to: {table_fig_path}")


def create_statistical_comparison_heatmap(csv_path: Path, output_path: Path, dataset_name: str, figsize=(20, 10)):
    """Create heatmap showing statistical test results vs. appropriate baseline."""
    # Load full data (not summary)
    df = pd.read_csv(csv_path)
    
    # Remove sparse_mean
    df = df[df['model'] != 'sparse_mean']
    
    # Get unique models and metrics
    models = df['model'].unique()
    metrics = [m for m in METRIC_INFO.keys() if m in df['metric'].unique()]
    
    # Initialize matrices for test statistics and p-values
    # Initialize with NaN instead of leaving uninitialized
    test_stats = pd.DataFrame(np.nan, index=models, columns=metrics, dtype=float)
    p_values = pd.DataFrame(np.nan, index=models, columns=metrics, dtype=float)
    
    # For each model and metric, perform paired t-test vs. appropriate baseline
    # Store raw p-values for correction
    raw_p_values = p_values.copy()
    
    for metric in metrics:
        # Determine baseline based on metric type
        if 'deltapert' in metric:
            baseline_model = 'control_mean'
        else:
            baseline_model = 'dataset_mean'
        
        # Get baseline data
        baseline_data = df[(df['model'] == baseline_model) & (df['metric'] == metric)]
        
        if len(baseline_data) == 0:
            print(f"Warning: {baseline_model} not found for {metric}, skipping")
            continue
        
        # Store raw p-values for this metric to apply correction
        metric_p_values = []
        metric_models = []
        
        # For each model, test against baseline
        for model in models:
            if model == baseline_model or model == 'ground_truth':
                # Skip baseline vs itself and ground truth
                test_stats.loc[model, metric] = 0.0
                raw_p_values.loc[model, metric] = 1.0
                continue
            
            model_data = df[(df['model'] == model) & (df['metric'] == metric)]
            
            if len(model_data) == 0:
                continue
            
            # Merge on perturbation to ensure pairing
            merged = pd.merge(
                baseline_data[['perturbation', 'value']],
                model_data[['perturbation', 'value']],
                on='perturbation',
                suffixes=('_baseline', '_model')
            )
            
            # Remove rows with NaN values (common for DEG metrics)
            merged = merged.dropna(subset=['value_baseline', 'value_model'])
            
            if len(merged) < 3:
                # Need at least 3 pairs for t-test
                continue
            
            # Perform one-sided paired t-test
            try:
                if METRIC_INFO[metric]['higher_better']:
                    # Test if model > baseline
                    t_stat, pval = ttest_rel(
                        merged['value_model'], 
                        merged['value_baseline'],
                        alternative='greater'
                    )
                else:
                    # Test if model < baseline (better for MSE/WMSE)
                    t_stat, pval = ttest_rel(
                        merged['value_model'],
                        merged['value_baseline'],
                        alternative='less'
                    )
                    # Flip sign so positive = better
                    t_stat = -t_stat
                
                test_stats.loc[model, metric] = t_stat  
                raw_p_values.loc[model, metric] = pval
                metric_p_values.append(pval)
                metric_models.append(model)
            except Exception as e:
                print(f"Warning: t-test failed for {model} on {metric}: {e}")
                continue
        
        # Apply Bonferroni correction within this metric
        n_tests = len(metric_p_values)
        if n_tests > 0:
            for model, raw_p in zip(metric_models, metric_p_values):
                corrected_p = min(raw_p * n_tests, 1.0)  # Cap at 1.0
                p_values.loc[model, metric] = corrected_p
    
    # Convert to numeric
    test_stats = test_stats.apply(pd.to_numeric, errors='coerce')
    p_values = p_values.apply(pd.to_numeric, errors='coerce')
    
    # Sort rows and columns by mean test statistic
    # Treat NaN as 0 for sorting
    row_means = test_stats.fillna(0).mean(axis=1).sort_values(ascending=False)
    col_means = test_stats.fillna(0).mean(axis=0).sort_values(ascending=False)
    
    test_stats = test_stats.loc[row_means.index, col_means.index]
    p_values = p_values.loc[row_means.index, col_means.index]
    
    # Create annotations with test statistic and significance stars
    annotations = test_stats.copy().astype(str)
    for i, model in enumerate(test_stats.index):
        for j, metric in enumerate(test_stats.columns):
            stat_val = test_stats.iloc[i, j]
            p_val = p_values.iloc[i, j]
            
            if pd.isna(stat_val) or pd.isna(p_val):
                # Leave blank for null baselines and missing data
                annotations.iloc[i, j] = ''
            else:
                # Add significance stars (4 levels)
                if p_val < 0.0001:
                    sig = '****'
                elif p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = ''
                
                # Format: "t-stat\nstars"
                if sig:
                    annotations.iloc[i, j] = f"{stat_val:.1f}\n{sig}"
                else:
                    annotations.iloc[i, j] = f"{stat_val:.1f}"
    
    # Rename columns to display names
    display_cols = [METRIC_INFO[m]['display'] for m in test_stats.columns]
    test_stats.columns = display_cols
    annotations.columns = display_cols
    
    # Rename row indices (models) to display names
    test_stats.index = [format_model_name(m) for m in test_stats.index]
    annotations.index = [format_model_name(m) for m in annotations.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with diverging colors (clipped at ±30)
    heatmap = sns.heatmap(
        test_stats,
        annot=annotations,
        fmt='',
        cmap='RdBu_r',
        center=0,
        vmin=-30,
        vmax=30,
        cbar_kws={'label': 't-statistic'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        annot_kws={'size': 10, 'weight': 'bold'}
    )
    
    # Set colorbar label size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('t-statistic vs baseline (↑)', 
                   fontsize=14, fontweight='bold')
    
    ax.set_title(f'Performance over negative control ({dataset_name})', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Models', fontsize=16, fontweight='bold')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=13)
    
    # Style y-axis labels: italicize and color baselines differently
    # Use formatted baseline names since we renamed the indices
    formatted_baseline_names = ['Control Mean', 'Technical Duplicate', 'Sparse Mean',
                                'Interpolated Duplicate', 'Additive', 'Dataset Mean',
                                'Linear', 'Baselines', 'Ground Truth']
    
    yticklabels = ax.get_yticklabels()
    for label in yticklabels:
        if label.get_text() in formatted_baseline_names:
            label.set_fontstyle('italic')
            label.set_color('#666666')  # Gray color for baselines
            label.set_fontsize(13)
        else:
            label.set_fontweight('bold')
            label.set_color('black')
            label.set_fontsize(13)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved statistical comparison heatmap to: {output_path}")


def create_stripplot_for_metric(args):
    """Create a strip plot for a single metric (for parallel processing)."""
    df, metric, output_dir, dataset_name, use_log_scale = args
    
    metric_df = df[df['metric'] == metric].copy()
    
    # Always exclude sparse_mean
    metric_df = metric_df[metric_df['model'] != 'sparse_mean']
    
    if metric_df.empty:
        return None
    
    # Determine null baseline for this metric
    if 'deltapert' in metric:
        null_baseline = 'control_mean'
    else:
        null_baseline = 'dataset_mean'
    
    # Calculate mean for each model and sort
    model_means = metric_df.groupby('model')['value'].mean()
    
    # Sort based on metric type
    if metric.lower() in ['mse', 'wmse']:
        model_means = model_means.sort_values(ascending=True)
    else:
        model_means = model_means.sort_values(ascending=False)
    
    # Reorder models
    metric_df['model'] = pd.Categorical(metric_df['model'], 
                                       categories=model_means.index, 
                                       ordered=True)
    
    # Perform paired t-tests vs. null baseline
    test_results = {}  # model -> (t_stat, p_val)
    null_data = metric_df[metric_df['model'] == null_baseline]
    
    if len(null_data) > 0:
        for model_name in model_means.index:
            if model_name == null_baseline:
                test_results[model_name] = (0.0, 1.0)
                continue
            
            model_data = metric_df[metric_df['model'] == model_name]
            
            # Merge on perturbation for pairing
            merged = pd.merge(
                null_data[['perturbation', 'value']],
                model_data[['perturbation', 'value']],
                on='perturbation',
                suffixes=('_null', '_model')
            )
            
            # Remove NaN values
            merged = merged.dropna(subset=['value_null', 'value_model'])
            
            if len(merged) >= 3:
                try:
                    # One-sided paired t-test
                    if METRIC_INFO.get(metric, {}).get('higher_better', True):
                        # Test if model > baseline
                        t_stat, p_val = ttest_rel(
                            merged['value_model'],
                            merged['value_null'],
                            alternative='greater'
                        )
                    else:
                        # Test if model < baseline (better for MSE/WMSE)
                        t_stat, p_val = ttest_rel(
                            merged['value_model'],
                            merged['value_null'],
                            alternative='less'
                        )
                    
                    test_results[model_name] = (t_stat, p_val)
                except:
                    test_results[model_name] = (np.nan, np.nan)
            else:
                test_results[model_name] = (np.nan, np.nan)
        
        # Apply Bonferroni correction
        n_tests = sum(1 for _, (_, p) in test_results.items() if not np.isnan(p) and p < 1.0)
        if n_tests > 0:
            test_results = {model: (t, p * n_tests if not np.isnan(p) else np.nan) 
                          for model, (t, p) in test_results.items()}
    
    # Identify baselines
    baseline_keywords = ['control_mean', 'technical_duplicate', 'sparse_mean', 
                        'interpolated_duplicate', 'additive', 'dataset_mean', 
                        'linear', 'baselines', 'ground_truth']
    
    def is_baseline(model_name):
        return model_name in baseline_keywords
    
    # Calculate figure size
    n_models = len(model_means)
    fig_width = max(10, min(20, n_models * 1.1))  # More spacing
    fig_height = 7
    
    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    ax.set_facecolor('white')
    
    # Color scheme
    baseline_color = '#2E86AB'
    model_color = '#424242'
    mean_color = '#A23B72'
    
    # Create color palette
    palette = []
    for model_name in model_means.index:
        if is_baseline(model_name):
            palette.append(baseline_color)
        else:
            palette.append(model_color)
    
    # Strip plot (higher alpha for log scale version)
    alpha_val = 0.4 if use_log_scale else 0.2
    sns.stripplot(data=metric_df, x='model', y='value', 
                 hue='model',
                 palette=palette,
                 size=3,
                 alpha=alpha_val,
                 jitter=True,
                 legend=False,
                 ax=ax)
    
    # Add mean lines (narrower)
    for i, (model_name, model_mean) in enumerate(model_means.items()):
        ax.hlines(model_mean, i - 0.25, i + 0.25, 
                 colors=mean_color, 
                 linewidth=2.5,
                 zorder=3)
        
        if n_models <= 8:
            ax.text(i, model_mean, f'{model_mean:.3f}', 
                   horizontalalignment='center',
                   verticalalignment='bottom',
                   fontsize=13,
                   color=mean_color,
                   fontweight='bold')
    
    # Customize plot
    metric_display = METRIC_INFO.get(metric, {}).get('display', metric)
    
    # Add directional arrow in axis label
    direction_arrow = '↓' if not METRIC_INFO.get(metric, {}).get('higher_better', True) else '↑'
    
    # Apply log scale for MSE/WMSE in log version
    if use_log_scale and metric.lower() in ['mse', 'wmse']:
        ax.set_yscale('log')
        ylabel = f'{metric_display} (log scale, {direction_arrow})'
    else:
        ylabel = f'{metric_display} ({direction_arrow})'
    
    # Simple title: just metric and dataset
    ax.set_title(f'{metric_display} in {dataset_name}', 
                fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel('Model', fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    
    # Style x-axis labels with nice formatting
    ax.set_xticks(range(len(model_means)))
    display_labels = [format_model_name(m) for m in model_means.index]
    xticklabels = ax.set_xticklabels(display_labels, rotation=45, ha='right')
    
    for i, label in enumerate(xticklabels):
        model_name = model_means.index[i]
        if is_baseline(model_name):
            label.set_fontstyle('italic')
            label.set_color('#666666')
            label.set_fontsize(14)
        else:
            label.set_fontweight('bold')
            label.set_color('black')
            label.set_fontsize(14)
    
    # Grid and styling
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Cap y-axis at -1 for R² metrics and annotate per-model counts
    r2_metrics = ['r2_deltactrl', 'r2_deltactrl_degs', 'r2_deltapert', 'r2_deltapert_degs',
                  'weighted_r2_deltactrl', 'weighted_r2_deltapert']
    if metric in r2_metrics:
        any_clipped = False
        for i, model_name in enumerate(model_means.index):
            model_df_subset = metric_df[metric_df['model'] == model_name]
            n_below = (model_df_subset['value'] < -1).sum()
            if n_below > 0:
                any_clipped = True
                # Add clean text annotation on plot at y=-0.97, offset to the right
                ax.text(i + 0.12, -0.95, f'↓ {n_below}',
                       ha='left', va='center',
                       fontsize=10,
                       color='#555555')  # Dark grey
        
        if any_clipped:
            current_ylim = ax.get_ylim()
            ax.set_ylim(-1, current_ylim[1])
    
    # Add reference line at 0 for certain metrics
    if any(keyword in metric.lower() for keyword in ['pearson', 'r2', 'delta']):
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    # Add statistical annotations if we have test results
    if test_results:
        # For R² and Pearson metrics, use fixed position at y=1.1
        # For other metrics, use dynamic positioning
        r2_pearson_metrics = ['r2_deltactrl', 'r2_deltactrl_degs', 'r2_deltapert', 'r2_deltapert_degs',
                             'weighted_r2_deltactrl', 'weighted_r2_deltapert',
                             'pearson_deltactrl', 'pearson_deltactrl_degs', 
                             'pearson_deltapert', 'pearson_deltapert_degs']
        
        if metric in r2_pearson_metrics:
            # Fixed position at y=1.1
            annotation_y = 1.1
            y_min, y_max = ax.get_ylim()
            if y_max < 1.2:
                ax.set_ylim(y_min, 1.2)  # Ensure room for annotation
        else:
            # Dynamic positioning for other metrics
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            new_y_max = y_max + y_range * 0.2
            ax.set_ylim(y_min, new_y_max)
            annotation_y = y_max + y_range * 0.05
        
        # Add annotations for each model
        
        for i, model_name in enumerate(model_means.index):
            # Skip showing stats for the null baseline itself
            if model_name == null_baseline:
                continue
            
            if model_name in test_results:
                t_stat, p_val = test_results[model_name]
                
                if not np.isnan(t_stat) and not np.isnan(p_val):
                    # Format p-value
                    if p_val < 0.01:
                        p_text = f"p={p_val:.1e}"
                    elif p_val < 1.0:
                        p_text = f"p={p_val:.3f}"
                    else:
                        p_text = f"p=1"
                    
                    # Create annotation text with t-statistic
                    annot_text = f"{p_text}\nt={t_stat:.1f}"
                    
                    # Add text annotation (no border)
                    ax.text(i, annotation_y, annot_text,
                           ha='center', va='bottom',
                           fontsize=10)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=model_color, 
              markersize=10, alpha=0.6, label='Models'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=baseline_color, 
              markersize=10, alpha=0.6, label='Baselines'),
        Line2D([0], [0], color=mean_color, linewidth=3, label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
             frameon=True, fancybox=True, shadow=False, framealpha=0.9, fontsize=12)
    
    plt.tight_layout()
    
    # Save with appropriate filename
    if use_log_scale:
        filename = f"stripplot_log_{metric}.png"
    else:
        filename = f"stripplot_{metric}.png"
    
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    return filename


def create_forest_plot_for_metric(args):
    """Create a forest plot showing performance difference from baseline with 95% CI."""
    df, metric, output_dir, dataset_name = args
    
    metric_df = df[df['metric'] == metric].copy()
    
    # Always exclude sparse_mean
    metric_df = metric_df[metric_df['model'] != 'sparse_mean']
    
    if metric_df.empty:
        return None
    
    # Determine null baseline for this metric
    if 'deltapert' in metric:
        null_baseline = 'control_mean'
    else:
        null_baseline = 'dataset_mean'
    
    # Get baseline data
    baseline_data = metric_df[metric_df['model'] == null_baseline]
    
    if len(baseline_data) == 0:
        return None
    
    # Calculate differences from baseline for each model
    model_differences = {}
    models = metric_df['model'].unique()
    
    for model_name in models:
        if model_name == null_baseline:
            continue
        
        model_data = metric_df[metric_df['model'] == model_name]
        
        # Merge on perturbation
        merged = pd.merge(
            baseline_data[['perturbation', 'value']],
            model_data[['perturbation', 'value']],
            on='perturbation',
            suffixes=('_baseline', '_model')
        )
        
        # Remove NaN values
        merged = merged.dropna(subset=['value_baseline', 'value_model'])
        
        if len(merged) >= 3:
            # Calculate difference for each perturbation
            differences = merged['value_model'] - merged['value_baseline']
            
            # Calculate mean and 95% CI using bootstrap
            mean_diff = differences.mean()
            
            # Bootstrap CI (non-parametric)
            result = bootstrap(
                (differences.values,),
                statistic=np.mean,
                n_resamples=10000,
                confidence_level=0.95,
                method='percentile',
                random_state=42
            )
            
            model_differences[model_name] = {
                'mean': mean_diff,
                'ci_lower': result.confidence_interval.low,
                'ci_upper': result.confidence_interval.high,
                'n': len(differences)
            }
    
    # Calculate t-tests (same as stripplot)
    test_results = {}
    
    for model_name in models:
        if model_name == null_baseline:
            continue
        
        model_data = metric_df[metric_df['model'] == model_name]
        
        merged = pd.merge(
            baseline_data[['perturbation', 'value']],
            model_data[['perturbation', 'value']],
            on='perturbation',
            suffixes=('_baseline', '_model')
        )
        
        merged = merged.dropna(subset=['value_baseline', 'value_model'])
        
        if len(merged) >= 3:
            try:
                if METRIC_INFO.get(metric, {}).get('higher_better', True):
                    t_stat, p_val = ttest_rel(
                        merged['value_model'],
                        merged['value_baseline'],
                        alternative='greater'
                    )
                else:
                    t_stat, p_val = ttest_rel(
                        merged['value_model'],
                        merged['value_baseline'],
                        alternative='less'
                    )
                
                test_results[model_name] = (t_stat, p_val)
            except:
                test_results[model_name] = (np.nan, np.nan)
    
    if not model_differences:
        return None
    
    # Sort models by mean difference
    # For "higher is better": positive diff is better (sort descending)
    # For "lower is better": negative diff is better (sort ascending)
    if METRIC_INFO.get(metric, {}).get('higher_better', True):
        sorted_models = sorted(model_differences.items(), 
                              key=lambda x: x[1]['mean'], 
                              reverse=True)
    else:
        # For MSE/WMSE: more negative difference = better
        sorted_models = sorted(model_differences.items(), 
                              key=lambda x: x[1]['mean'], 
                              reverse=False)
    
    # Create forest plot (narrower)
    fig, ax = plt.subplots(figsize=(8, max(7, len(sorted_models) * 0.3)))
    
    # Identify baselines
    baseline_keywords = ['control_mean', 'technical_duplicate', 'sparse_mean', 
                        'interpolated_duplicate', 'additive', 'dataset_mean', 
                        'linear', 'baselines', 'ground_truth']
    
    # Plot each model
    y_positions = []
    for i, (model_name, stats) in enumerate(sorted_models):
        y_pos = len(sorted_models) - i - 1
        y_positions.append(y_pos)
        
        # Determine color
        is_baseline = model_name in baseline_keywords
        color = '#2E86AB' if is_baseline else '#424242'
        
        # Plot point estimate
        ax.plot(stats['mean'], y_pos, 'o', color=color, markersize=7.5, zorder=3)
        
        # Plot 95% CI as error bars
        ax.plot([stats['ci_lower'], stats['ci_upper']], [y_pos, y_pos],
               color=color, linewidth=2, zorder=2)
        
        # Add caps to error bars
        cap_height = 0.15
        ax.plot([stats['ci_lower'], stats['ci_lower']], 
               [y_pos - cap_height, y_pos + cap_height],
               color=color, linewidth=2, zorder=2)
        ax.plot([stats['ci_upper'], stats['ci_upper']], 
               [y_pos - cap_height, y_pos + cap_height],
               color=color, linewidth=2, zorder=2)
    
    # Add vertical line at 0 (no difference from baseline)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    
    # Set y-axis labels with styled text and nice formatting
    ax.set_yticks(y_positions)
    model_labels = [format_model_name(model_name) for model_name, _ in sorted_models]
    ax.set_yticklabels(model_labels)
    
    # Style y-axis labels
    yticklabels = ax.get_yticklabels()
    original_model_names = [model_name for model_name, _ in sorted_models]
    for i, label in enumerate(yticklabels):
        # Use original model name for baseline check
        original_name = original_model_names[i]
        if original_name in baseline_keywords:
            label.set_fontstyle('italic')
            label.set_color('#666666')
            label.set_fontsize(13)
        else:
            label.set_fontweight('bold')
            label.set_color('black')
            label.set_fontsize(13)
    
    # Add statistical annotations (same as stripplot)
    if test_results:
        # Find max CI upper bound to position annotations
        max_ci_upper = max(stats['ci_upper'] for _, stats in sorted_models)
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        
        # Position annotations slightly beyond max CI (5% of range)
        annot_x = max_ci_upper + x_range * 0.05
        
        for i, (model_name, _) in enumerate(sorted_models):
            y_pos = len(sorted_models) - i - 1
            
            if model_name in test_results:
                t_stat, p_val = test_results[model_name]
                
                if not np.isnan(t_stat) and not np.isnan(p_val):
                    # Format p-value
                    if p_val < 0.01:
                        p_text = f"p={p_val:.1e}"
                    elif p_val < 1.0:
                        p_text = f"p={p_val:.3f}"
                    else:
                        p_text = f"p=1"
                    
                    annot_text = f"{p_text}\nt={t_stat:.1f}"
                    
                    # Position to the right of all CIs, left-aligned
                    ax.text(annot_x, y_pos, annot_text,
                           ha='left', va='center',
                           fontsize=10)
    
    # Labels and title
    metric_display = METRIC_INFO.get(metric, {}).get('display', metric)
    direction_arrow = '↓' if not METRIC_INFO.get(metric, {}).get('higher_better', True) else '↑'
    baseline_display = format_model_name(null_baseline).lower()
    
    ax.set_xlabel(f'Δ metric (prediction - {baseline_display}) ({direction_arrow})', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Model', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_display} in {dataset_name}', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    filename = f"forest_{metric}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def create_all_stripplots(csv_path: Path, output_dir: Path, dataset_name: str):
    """Create strip plots for all metrics in parallel."""
    df = pd.read_csv(csv_path)
    
    # Get unique metrics
    metrics = [m for m in METRIC_INFO.keys() if m in df['metric'].unique()]
    
    print(f"Creating {len(metrics)} strip plots in parallel...")
    
    # Prepare arguments for parallel processing
    args_list = [(df, metric, output_dir, dataset_name, False) for metric in metrics]
    
    # Use all available CPUs
    n_workers = min(multiprocessing.cpu_count(), len(metrics))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        filenames = list(executor.map(create_stripplot_for_metric, args_list))
    
    # Filter out None results
    filenames = [f for f in filenames if f is not None]
    
    print(f"✓ Created {len(filenames)} strip plots")
    
    # Also create log-scale versions
    print(f"Creating {len(metrics)} log-scale strip plots in parallel...")
    args_list_log = [(df, metric, output_dir, dataset_name, True) for metric in metrics]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        filenames_log = list(executor.map(create_stripplot_for_metric, args_list_log))
    
    filenames_log = [f for f in filenames_log if f is not None]
    print(f"✓ Created {len(filenames_log)} log-scale strip plots")


def create_all_forest_plots(csv_path: Path, output_dir: Path, dataset_name: str):
    """Create forest plots for all metrics in parallel."""
    df = pd.read_csv(csv_path)
    
    # Get unique metrics
    metrics = [m for m in METRIC_INFO.keys() if m in df['metric'].unique()]
    
    print(f"Creating {len(metrics)} forest plots in parallel...")
    
    # Prepare arguments for parallel processing
    args_list = [(df, metric, output_dir, dataset_name) for metric in metrics]
    
    # Use all available CPUs
    n_workers = min(multiprocessing.cpu_count(), len(metrics))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        filenames = list(executor.map(create_forest_plot_for_metric, args_list))
    
    # Filter out None results
    filenames = [f for f in filenames if f is not None]
    
    print(f"✓ Created {len(filenames)} forest plots")


def create_auxiliary_plots(csv_path: Path, output_dir: Path, dataset_name: str):
    """Create focused auxiliary plots for paper figures."""
    df = pd.read_csv(csv_path)
    
    # Create aux_plots subdirectory
    aux_dir = output_dir / 'aux_plots'
    aux_dir.mkdir(exist_ok=True, parents=True)
    
    print("Creating auxiliary plots for paper...")
    
    # Simple 2-model plots
    create_aux_plot(df, 'mse', 'dataset_mean', 'technical_duplicate', 
                   aux_dir, dataset_name)
    create_aux_plot(df, 'pearson_deltactrl', 'dataset_mean', 'technical_duplicate',
                   aux_dir, dataset_name)
    
    # Expanded 3-model plots
    create_aux_plot_expanded(df, 'mse', 
                            ['dataset_mean', 'interpolated_duplicate', 'technical_duplicate'],
                            aux_dir, dataset_name)
    create_aux_plot_expanded(df, 'pearson_deltactrl',
                            ['dataset_mean', 'interpolated_duplicate', 'technical_duplicate'],
                            aux_dir, dataset_name)
    
    # Scatter plot comparing duplicates
    create_duplicate_comparison_scatter(df, aux_dir, dataset_name)
    create_baseline_comparison_scatter(df, aux_dir, dataset_name)
    
    print(f"✓ Created auxiliary plots in {aux_dir}")


def create_aux_plot(df: pd.DataFrame, metric: str, baseline_model: str, 
                   comparison_model: str, output_dir: Path, dataset_name: str):
    """Create a simplified comparison plot for paper figure."""
    
    metric_df = df[df['metric'] == metric].copy()
    
    if metric_df.empty:
        return
    
    # Filter to only the two models of interest
    models_to_plot = [baseline_model, comparison_model]
    plot_df = metric_df[metric_df['model'].isin(models_to_plot)]
    
    if plot_df.empty:
        return
    
    # Calculate means for ordering
    model_means = plot_df.groupby('model')['value'].mean()
    
    # Order: baseline first, then comparison
    ordered_models = [baseline_model, comparison_model]
    plot_df['model'] = pd.Categorical(plot_df['model'], 
                                      categories=ordered_models, 
                                      ordered=True)
    
    # Perform paired t-test
    baseline_data = metric_df[metric_df['model'] == baseline_model]
    comparison_data = metric_df[metric_df['model'] == comparison_model]
    
    merged = pd.merge(
        baseline_data[['perturbation', 'value']],
        comparison_data[['perturbation', 'value']],
        on='perturbation',
        suffixes=('_baseline', '_comparison')
    )
    
    merged = merged.dropna(subset=['value_baseline', 'value_comparison'])
    
    t_stat = np.nan
    p_val = np.nan
    if len(merged) >= 3:
        try:
            # One-sided t-test: is comparison model better than baseline?
            if METRIC_INFO.get(metric, {}).get('higher_better', True):
                # Test if comparison > baseline
                t_stat, p_val = ttest_rel(
                    merged['value_comparison'],
                    merged['value_baseline'],
                    alternative='greater'
                )
            else:
                # Test if comparison < baseline (better for MSE/WMSE)
                t_stat, p_val = ttest_rel(
                    merged['value_comparison'],
                    merged['value_baseline'],
                    alternative='less'
                )
        except:
            pass
    
    # Create figure (narrower)
    fig, ax = plt.subplots(figsize=(5, 7), facecolor='white')
    ax.set_facecolor('white')
    
    # Use same blue color for both models
    plot_color = '#2E86AB'
    colors = [plot_color, plot_color]
    
    # Strip plot
    sns.stripplot(data=plot_df, x='model', y='value',
                 hue='model',
                 palette=colors,
                 size=4,
                 alpha=0.5,
                 jitter=True,
                 legend=False,
                 ax=ax)
    
    # Add mean lines (black, narrower)
    mean_color = 'black'
    for i, model in enumerate(ordered_models):
        if model in model_means.index:
            mean_val = model_means[model]
            ax.hlines(mean_val, i - 0.25, i + 0.25,
                     colors=mean_color,
                     linewidth=3,
                     zorder=3)
            
            # Add mean value text
            ax.text(i, mean_val, f'{mean_val:.4f}',
                   ha='center', va='bottom',
                   fontsize=14,
                   color=mean_color,
                   fontweight='bold')
    
    # Add t-test annotation for comparison model
    if not np.isnan(t_stat) and not np.isnan(p_val):
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        # Extend y-axis
        new_y_max = y_max + y_range * 0.15
        ax.set_ylim(y_min, new_y_max)
        
        # Format p-value
        if p_val < 0.001:
            p_text = f"p={p_val:.2e}"
        else:
                    p_text = f"p={p_val:.4f}"
        
        annot_text = f"{p_text}\nt={t_stat:.2f}"
        
        # Add annotation above comparison model (position 1) - NO background, NO border
        ax.text(1, y_max + y_range * 0.03, annot_text,
               ha='center', va='bottom',
               fontsize=10)
    
    # Styling
    metric_display = METRIC_INFO.get(metric, {}).get('display', metric)
    direction_arrow = '↓' if not METRIC_INFO.get(metric, {}).get('higher_better', True) else '↑'
    ax.set_title(f'{metric_display} in {dataset_name}',
                fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel('', fontsize=16)
    ax.set_ylabel(f'{metric_display} ({direction_arrow})', fontsize=16, fontweight='bold')
    
    # Style x-axis labels with nice formatting
    display_labels = [format_model_name(baseline_model), format_model_name(comparison_model)]
    ax.set_xticklabels(display_labels, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Cap y-axis at -1 for R² metrics and annotate per-model counts
    r2_metrics = ['r2_deltactrl', 'r2_deltactrl_degs', 'r2_deltapert', 'r2_deltapert_degs',
                  'weighted_r2_deltactrl', 'weighted_r2_deltapert']
    if metric in r2_metrics:
        any_clipped = False
        for i, model_name in enumerate(ordered_models):
            model_df_subset = plot_df[plot_df['model'] == model_name]
            n_below = (model_df_subset['value'] < -1).sum()
            if n_below > 0:
                any_clipped = True
                # Add clean text annotation on plot at y=-0.97, offset to the right
                ax.text(i + 0.3, -0.97, f'↓ n={n_below}',
                       ha='left', va='center',
                       fontsize=9,
                       color='#555555')  # Dark grey
        
        if any_clipped:
            current_ylim = ax.get_ylim()
            ax.set_ylim(-1, current_ylim[1])
    
    # Set log scale for MSE and WMSE
    if metric.lower() in ['mse', 'wmse']:
        ax.set_yscale('log')
    
    # Add reference line at 0 for certain metrics
    if any(keyword in metric.lower() for keyword in ['pearson', 'r2', 'delta']):
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    plt.tight_layout()
    
    # Save
    filename = f"aux_{metric}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✓ Saved {filename}")


def create_aux_plot_expanded(df: pd.DataFrame, metric: str, models_to_plot: list,
                             output_dir: Path, dataset_name: str):
    """Create expanded auxiliary plot with multiple baselines."""
    
    metric_df = df[df['metric'] == metric].copy()
    
    # Always exclude sparse_mean
    metric_df = metric_df[metric_df['model'] != 'sparse_mean']
    
    if metric_df.empty:
        return
    
    # Filter to specified models
    plot_df = metric_df[metric_df['model'].isin(models_to_plot)]
    
    if plot_df.empty:
        return
    
    # Calculate means for ordering
    model_means = plot_df.groupby('model')['value'].mean()
    
    # Order by mean (best to worst)
    if metric.lower() in ['mse', 'wmse']:
        ordered_models = model_means.sort_values(ascending=True).index.tolist()
    else:
        ordered_models = model_means.sort_values(ascending=False).index.tolist()
    
    plot_df['model'] = pd.Categorical(plot_df['model'], 
                                      categories=ordered_models, 
                                      ordered=True)
    
    # Perform t-tests vs. dataset_mean for all models
    test_results = {}
    dataset_mean_data = metric_df[metric_df['model'] == 'dataset_mean']
    
    if len(dataset_mean_data) > 0:
        for model_name in ordered_models:
            if model_name == 'dataset_mean':
                test_results[model_name] = (0.0, 1.0)
                continue
            
            model_data = metric_df[metric_df['model'] == model_name]
            
            # Merge on perturbation
            merged = pd.merge(
                dataset_mean_data[['perturbation', 'value']],
                model_data[['perturbation', 'value']],
                on='perturbation',
                suffixes=('_baseline', '_model')
            )
            
            merged = merged.dropna(subset=['value_baseline', 'value_model'])
            
            if len(merged) >= 3:
                try:
                    if METRIC_INFO.get(metric, {}).get('higher_better', True):
                        t_stat, p_val = ttest_rel(
                            merged['value_model'],
                            merged['value_baseline'],
                            alternative='greater'
                        )
                    else:
                        t_stat, p_val = ttest_rel(
                            merged['value_model'],
                            merged['value_baseline'],
                            alternative='less'
                        )
                    
                    test_results[model_name] = (t_stat, p_val)
                except:
                    test_results[model_name] = (np.nan, np.nan)
            else:
                test_results[model_name] = (np.nan, np.nan)
    
    # Create figure (narrower for 3 models)
    fig, ax = plt.subplots(figsize=(7.3, 5), facecolor='white')
    ax.set_facecolor('white')
    
    # Use blue for all baselines
    plot_color = '#2E86AB'
    colors = [plot_color] * len(ordered_models)
    
    # Strip plot
    sns.stripplot(data=plot_df, x='model', y='value',
                 hue='model',
                 palette=colors,
                 size=4,
                 alpha=0.5,
                 jitter=True,
                 legend=False,
                 ax=ax)
    
    # Add mean lines (black)
    mean_color = 'black'
    for i, model in enumerate(ordered_models):
        if model in model_means.index:
            mean_val = model_means[model]
            ax.hlines(mean_val, i - 0.25, i + 0.25,
                     colors=mean_color,
                     linewidth=3,
                     zorder=3)
            
            # Add mean value text
            ax.text(i, mean_val, f'{mean_val:.4f}',
                   ha='center', va='bottom',
                   fontsize=14,
                   color=mean_color,
                   fontweight='bold')
    
    # Styling
    metric_display = METRIC_INFO.get(metric, {}).get('display', metric)
    direction_arrow = '↓' if not METRIC_INFO.get(metric, {}).get('higher_better', True) else '↑'
    ax.set_title(f'{metric_display} in {dataset_name}',
                fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel('', fontsize=16)
    ax.set_ylabel(f'{metric_display} ({direction_arrow})', fontsize=16, fontweight='bold')
    
    # Style x-axis labels with nice formatting (tilted to avoid overlap)
    display_labels = [format_model_name(m) for m in ordered_models]
    ax.set_xticklabels(display_labels, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add reference line at 0 for certain metrics
    if any(keyword in metric.lower() for keyword in ['pearson', 'r2', 'delta']):
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    # Add t-test annotations
    if test_results:
        # For R² and Pearson metrics, use fixed position
        r2_pearson_metrics = ['r2_deltactrl', 'r2_deltactrl_degs', 'r2_deltapert', 'r2_deltapert_degs',
                             'weighted_r2_deltactrl', 'weighted_r2_deltapert',
                             'pearson_deltactrl', 'pearson_deltactrl_degs',
                             'pearson_deltapert', 'pearson_deltapert_degs']
        
        if metric in r2_pearson_metrics:
            annotation_y = 1.1
            y_min, y_max = ax.get_ylim()
            if y_max < 1.2:
                ax.set_ylim(y_min, 1.2)
        else:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            new_y_max = y_max + y_range * 0.15
            ax.set_ylim(y_min, new_y_max)
            annotation_y = y_max + y_range * 0.03
        
        for i, model_name in enumerate(ordered_models):
            if model_name == 'dataset_mean':
                continue
            
            if model_name in test_results:
                t_stat, p_val = test_results[model_name]
                
                if not np.isnan(t_stat) and not np.isnan(p_val):
                    if p_val < 0.01:
                        p_text = f"p={p_val:.1e}"
                    elif p_val < 1.0:
                        p_text = f"p={p_val:.3f}"
                    else:
                        p_text = f"p=1"
                    
                    annot_text = f"{p_text}\nt={t_stat:.1f}"
                    
                    ax.text(i, annotation_y, annot_text,
                           ha='center', va='bottom',
                           fontsize=12)
    
    plt.tight_layout()
    
    # Save
    filename = f"aux_expanded_{metric}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✓ Saved {filename}")


def create_duplicate_comparison_scatter(df: pd.DataFrame, output_dir: Path, dataset_name: str):
    """Create scatter plot comparing technical vs interpolated duplicate MSE per perturbation."""
    
    # Filter to MSE metric only
    mse_df = df[df['metric'] == 'mse'].copy()
    
    if mse_df.empty:
        return
    
    # Get data for each baseline
    tech_dup = mse_df[mse_df['model'] == 'technical_duplicate'][['perturbation', 'value']]
    interp_dup = mse_df[mse_df['model'] == 'interpolated_duplicate'][['perturbation', 'value']]
    dataset_mean = mse_df[mse_df['model'] == 'dataset_mean'][['perturbation', 'value']]
    
    if tech_dup.empty or interp_dup.empty or dataset_mean.empty:
        print("  Warning: Missing baseline data for duplicate comparison scatter")
        return
    
    # Merge on perturbation
    merged = pd.merge(tech_dup, interp_dup, on='perturbation', suffixes=('_tech', '_interp'))
    merged = pd.merge(merged, dataset_mean, on='perturbation')
    merged.columns = ['perturbation', 'tech_dup_mse', 'interp_dup_mse', 'mean_mse']
    
    # Remove any NaN values
    merged = merged.dropna()
    
    if len(merged) == 0:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    
    # Log-transform mean_mse for coloring
    merged['log_mean_mse'] = np.log10(merged['mean_mse'])
    
    # Create scatter plot
    scatter = ax.scatter(
        merged['tech_dup_mse'],
        merged['interp_dup_mse'],
        c=merged['log_mean_mse'],
        cmap='viridis',
        s=50,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀(Dataset Mean MSE)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add identity line (x=y)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.7, linewidth=2, zorder=1, label='Identity (x=y)')
    
    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Labels and title
    ax.set_xlabel('Technical Duplicate MSE', fontsize=14, fontweight='bold')
    ax.set_ylabel('Interpolated Duplicate MSE', fontsize=14, fontweight='bold')
    ax.set_title('Interpolated Duplicate Vs Technical Duplicate Error',
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend for identity line
    ax.legend(loc='upper left', fontsize=11)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    filename = "aux_duplicate_comparison_scatter.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved {filename}")


def create_baseline_comparison_scatter(df: pd.DataFrame, output_dir: Path, dataset_name: str):
    """Create scatter plot comparing interpolated duplicate vs dataset mean, colored by tech dup."""
    
    # Filter to MSE metric only
    mse_df = df[df['metric'] == 'mse'].copy()
    
    if mse_df.empty:
        return
    
    # Get data for each baseline
    tech_dup = mse_df[mse_df['model'] == 'technical_duplicate'][['perturbation', 'value']]
    interp_dup = mse_df[mse_df['model'] == 'interpolated_duplicate'][['perturbation', 'value']]
    dataset_mean = mse_df[mse_df['model'] == 'dataset_mean'][['perturbation', 'value']]
    
    if tech_dup.empty or interp_dup.empty or dataset_mean.empty:
        print("  Warning: Missing baseline data for baseline comparison scatter")
        return
    
    # Merge on perturbation
    merged = pd.merge(interp_dup, dataset_mean, on='perturbation', suffixes=('_interp', '_mean'))
    merged = pd.merge(merged, tech_dup, on='perturbation')
    merged.columns = ['perturbation', 'interp_dup_mse', 'mean_mse', 'tech_dup_mse']
    
    # Remove any NaN values
    merged = merged.dropna()
    
    if len(merged) == 0:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    
    # Log-transform tech_dup_mse for coloring
    merged['log_tech_dup_mse'] = np.log10(merged['tech_dup_mse'])
    
    # Create scatter plot
    scatter = ax.scatter(
        merged['interp_dup_mse'],
        merged['mean_mse'],
        c=merged['log_tech_dup_mse'],
        cmap='viridis',
        s=50,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀(Technical Duplicate MSE)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add identity line (x=y)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.7, linewidth=2, zorder=1, label='Identity (x=y)')
    
    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Labels and title
    ax.set_xlabel('Interpolated Duplicate MSE', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dataset Mean MSE', fontsize=14, fontweight='bold')
    ax.set_title('Interpolated Duplicate Vs Dataset Mean Error',
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend for identity line
    ax.legend(loc='upper left', fontsize=11)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    filename = "aux_baseline_comparison_scatter.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved {filename}")


def create_latex_tables(csv_path: Path, mean_all_path: Path, mean_nodeg_path: Path,
                        ttest_all_path: Path, ttest_nodeg_path: Path, dataset_name: str):
    """Create LaTeX tables: two versions each for means and t-tests (with/without DEG metrics)."""
    df = pd.read_csv(csv_path)
    
    # Remove sparse_mean
    df = df[df['model'] != 'sparse_mean']
    
    # Get models and metrics
    all_models = [m for m in df['model'].unique() if m != 'ground_truth']
    all_metrics = [m for m in METRIC_INFO.keys() if m in df['metric'].unique()]
    
    # Classify models as baselines or trained models
    baseline_keywords = ['control_mean', 'technical_duplicate', 
                        'interpolated_duplicate', 'additive', 'dataset_mean', 
                        'linear', 'baselines']
    
    baselines = [m for m in all_models if m in baseline_keywords]
    trained_models = [m for m in all_models if m not in baseline_keywords]
    
    # Sort within each group
    baselines.sort()
    trained_models.sort()
    
    # Calculate mean and SEM
    summary = df.groupby(['model', 'metric'])['value'].agg(['mean', 'sem']).reset_index()
    
    # Calculate t-statistics for each model vs appropriate baseline
    t_stats = {}
    p_values = {}
    
    for metric in all_metrics:
        # Determine baseline
        if 'deltapert' in metric:
            baseline_model = 'control_mean'
        else:
            baseline_model = 'dataset_mean'
        
        baseline_data = df[(df['model'] == baseline_model) & (df['metric'] == metric)]
        
        if len(baseline_data) == 0:
            continue
        
        for model in all_models:
            if model == baseline_model:
                continue
            
            model_data = df[(df['model'] == model) & (df['metric'] == metric)]
            
            if len(model_data) == 0:
                continue
            
            # Merge on perturbation
            merged = pd.merge(
                baseline_data[['perturbation', 'value']],
                model_data[['perturbation', 'value']],
                on='perturbation',
                suffixes=('_baseline', '_model')
            )
            
            merged = merged.dropna()
            
            if len(merged) >= 3:
                try:
                    if METRIC_INFO[metric]['higher_better']:
                        t_stat, p_val = ttest_rel(
                            merged['value_model'],
                            merged['value_baseline'],
                            alternative='greater'
                        )
                    else:
                        t_stat, p_val = ttest_rel(
                            merged['value_model'],
                            merged['value_baseline'],
                            alternative='less'
                        )
                    
                    t_stats[(model, metric)] = t_stat
                    p_values[(model, metric)] = p_val
                except:
                    pass
    
    # Positive controls (don't bold if they win)
    positive_controls = ['technical_duplicate', 'interpolated_duplicate']
    
    # Create all four table versions
    # Version 1: Means with all metrics
    metrics_all = all_metrics
    create_single_latex_table(mean_all_path, summary, trained_models, baselines, metrics_all,
                             positive_controls, dataset_name, "mean", "all")
    
    # Version 2: Means without DEG metrics
    metrics_nodeg = [m for m in all_metrics if '_degs' not in m]
    create_single_latex_table(mean_nodeg_path, summary, trained_models, baselines, metrics_nodeg,
                             positive_controls, dataset_name, "mean", "nodeg")
    
    # Version 3: T-tests with all metrics  
    create_single_latex_table(ttest_all_path, summary, trained_models, baselines, metrics_all,
                             positive_controls, dataset_name, "ttest", "all", t_stats, p_values)
    
    # Version 4: T-tests without DEG metrics
    create_single_latex_table(ttest_nodeg_path, summary, trained_models, baselines, metrics_nodeg,
                             positive_controls, dataset_name, "ttest", "nodeg", t_stats, p_values)
    
    print(f"✓ Saved LaTeX tables to: {mean_all_path.parent}")


def create_single_latex_table(output_path, summary, trained_models, baselines, metrics,
                              positive_controls, dataset_name, table_type, version,
                              t_stats=None, p_values=None):
    """Helper to create a single LaTeX table."""
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    
    table_name = "Mean performance ± SEM" if table_type == "mean" else "Statistical comparison vs baseline"
    deg_suffix = " (all metrics)" if version == "all" else " (excluding DEG metrics)"
    lines.append(f"\\caption{{{table_name} for {dataset_name}{deg_suffix}.}}")
    lines.append(f"\\label{{tab:multimodel_{table_type}_{version}_{dataset_name}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    
    # Build header with Type column
    metric_headers = [METRIC_INFO[m]['display'] for m in metrics]
    header_line = "\\textbf{Type} & \\textbf{Model} & " + " & ".join([f"\\textbf{{{h}}}" for h in metric_headers]) + " \\\\"
    lines.append("\\begin{tabular}{ll" + "c" * len(metrics) + "}")
    lines.append("\\toprule")
    lines.append(header_line)
    lines.append("\\midrule")
    
    if table_type == "mean":
        # Mean ± SEM table
        for i, model in enumerate(trained_models):
            model_display = format_model_name(model)
            type_cell = "\\multirow{" + str(len(trained_models)) + "}{*}{Model}" if i == 0 else ""
            row_vals = []
            
            for metric in metrics:
                model_summary = summary[(summary['model'] == model) & (summary['metric'] == metric)]
                
                if len(model_summary) > 0:
                    mean_val = model_summary['mean'].iloc[0]
                    sem_val = model_summary['sem'].iloc[0]
                    
                    # Check if best among non-positive-control models
                    metric_summary = summary[summary['metric'] == metric]
                    # Filter to exclude positive controls
                    non_control_summary = metric_summary[~metric_summary['model'].isin(positive_controls)]
                    
                    if len(non_control_summary) > 0:
                        if METRIC_INFO[metric]['higher_better']:
                            best_val = non_control_summary['mean'].max()
                        else:
                            best_val = non_control_summary['mean'].min()
                        
                        is_best = abs(mean_val - best_val) < 1e-6
                        should_bold = is_best and model not in positive_controls
                    else:
                        should_bold = False
                    
                    if should_bold:
                        row_vals.append(f"\\textbf{{{mean_val:.3f} ± {sem_val:.3f}}}")
                    else:
                        row_vals.append(f"{mean_val:.3f} ± {sem_val:.3f}")
                else:
                    row_vals.append("-")
            
            lines.append(f"{type_cell} & {model_display} & " + " & ".join(row_vals) + " \\\\")
        
        lines.append("\\midrule")
        
        # Add baselines section
        for i, model in enumerate(baselines):
            model_display = format_model_name(model)
            type_cell = "\\multirow{" + str(len(baselines)) + "}{*}{Baseline}" if i == 0 else ""
            row_vals = []
            
            for metric in metrics:
                model_summary = summary[(summary['model'] == model) & (summary['metric'] == metric)]
                
                if len(model_summary) > 0:
                    mean_val = model_summary['mean'].iloc[0]
                    sem_val = model_summary['sem'].iloc[0]
                    
                    # Check if best among non-positive-control models (same logic as trained models)
                    metric_summary_all = summary[summary['metric'] == metric]
                    non_control_summary = metric_summary_all[~metric_summary_all['model'].isin(positive_controls)]
                    
                    if len(non_control_summary) > 0:
                        if METRIC_INFO[metric]['higher_better']:
                            best_val = non_control_summary['mean'].max()
                        else:
                            best_val = non_control_summary['mean'].min()
                        
                        is_best = abs(mean_val - best_val) < 1e-6
                        should_bold = is_best and model not in positive_controls
                    else:
                        should_bold = False
                    
                    if should_bold:
                        row_vals.append(f"\\textbf{{{mean_val:.3f} ± {sem_val:.3f}}}")
                    else:
                        row_vals.append(f"{mean_val:.3f} ± {sem_val:.3f}")
                else:
                    row_vals.append("-")
            
            lines.append(f"{type_cell} & {model_display} & " + " & ".join(row_vals) + " \\\\")
    
    else:  # t-test table
        if t_stats is None or p_values is None:
            return
        
        # Add trained models section
        for i, model in enumerate(trained_models):
            model_display = format_model_name(model)
            type_cell = "\\multirow{" + str(len(trained_models)) + "}{*}{Model}" if i == 0 else ""
            row_vals = []
            
            for metric in metrics:
                if (model, metric) in t_stats:
                    t_val = t_stats[(model, metric)]
                    p_val = p_values[(model, metric)]
                    
                    # Stars
                    if p_val < 0.0001:
                        stars = '****'
                    elif p_val < 0.001:
                        stars = '***'
                    elif p_val < 0.01:
                        stars = '**'
                    elif p_val < 0.05:
                        stars = '*'
                    else:
                        stars = ''
                    
                    # Check if best among non-positive-control models
                    metric_pvals = {k: v for k, v in p_values.items() 
                                   if k[1] == metric and k[0] not in positive_controls}
                    if len(metric_pvals) > 0:
                        best_pval = min(metric_pvals.values())
                        is_best_pval = abs(p_val - best_pval) < 1e-10
                    else:
                        is_best_pval = False
                    
                    should_bold = is_best_pval and stars and model not in positive_controls
                    
                    if should_bold:
                        row_vals.append(f"\\textbf{{{t_val:.1f}({stars})}}")
                    else:
                        row_vals.append(f"{t_val:.1f}({stars})" if stars else f"{t_val:.1f}")
                else:
                    row_vals.append("-")
            
            lines.append(f"{type_cell} & {model_display} & " + " & ".join(row_vals) + " \\\\")
        
        lines.append("\\midrule")
        
        # Add baselines section
        for i, model in enumerate(baselines):
            model_display = format_model_name(model)
            type_cell = "\\multirow{" + str(len(baselines)) + "}{*}{Baseline}" if i == 0 else ""
            row_vals = []
            
            for metric in metrics:
                if (model, metric) in t_stats:
                    t_val = t_stats[(model, metric)]
                    p_val = p_values[(model, metric)]
                    
                    # Stars
                    if p_val < 0.0001:
                        stars = '****'
                    elif p_val < 0.001:
                        stars = '***'
                    elif p_val < 0.01:
                        stars = '**'
                    elif p_val < 0.05:
                        stars = '*'
                    else:
                        stars = ''
                    
                    row_vals.append(f"{t_val:.1f}({stars})" if stars else f"{t_val:.1f}")
                else:
                    row_vals.append("-")
            
            lines.append(f"{type_cell} & {model_display} & " + " & ".join(row_vals) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    
    # Save
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description='Generate beautiful summary plots for multi-model benchmarking results'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to detailed_metrics.csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as input CSV)'
    )
    parser.add_argument(
        '--exclude-baselines',
        action='store_true',
        help='Exclude baseline models from plots (default: include them as controls)'
    )
    
    args = parser.parse_args()
    
    # Load CSV
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"\nProcessing: {csv_path}")
    print("=" * 80)
    
    # Determine output directory - create additional_results subfolder
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = csv_path.parent
    
    output_dir = base_dir / 'additional_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load and process data
    print("Loading data...")
    exclude_baselines = args.exclude_baselines
    summary_df = load_and_process_data(csv_path, exclude_baselines=exclude_baselines)
    
    if exclude_baselines:
        print("(Excluding baseline models, showing only trained models)")
    else:
        print("(Including baseline models as controls)")
    
    n_models = summary_df['model'].nunique()
    n_metrics = summary_df['metric'].nunique()
    print(f"Found {n_models} models × {n_metrics} metrics")
    
    # Extract dataset name from CSV path
    # Expected format: .../benchmark_{models}_{dataset}/...
    try:
        parent_dir_name = csv_path.parent.parent.name
        dataset_name = parent_dir_name.split('_')[-1]  # Get last part
    except:
        dataset_name = "dataset"
    
    print(f"Dataset: {dataset_name}")
    
    # Generate plots
    print("\nGenerating plots...")
    print("-" * 80)
    
    # 1. Mean heatmap
    mean_heatmap_path = output_dir / 'multimodel_summary_heatmap.png'
    create_mean_heatmap(summary_df, mean_heatmap_path)
    
    # 2. Relative performance heatmap (vs. dataset_mean baseline)
    relative_heatmap_path = output_dir / 'multimodel_relative_performance.png'
    create_zscore_heatmap(summary_df, relative_heatmap_path)
    
    # 3. Summary table
    table_path = output_dir / 'multimodel_summary_table.csv'
    create_summary_table(summary_df, table_path)
    
    # 4. Statistical comparison heatmap
    stats_heatmap_path = output_dir / 'multimodel_statistical_comparison.png'
    create_statistical_comparison_heatmap(csv_path, stats_heatmap_path, dataset_name)
    
    # 5. Strip plots for all metrics (parallelized)
    create_all_stripplots(csv_path, output_dir, dataset_name)
    
    # 6. Forest plots for all metrics (parallelized)
    create_all_forest_plots(csv_path, output_dir, dataset_name)
    
    # 7. Auxiliary plots for paper
    create_auxiliary_plots(csv_path, output_dir, dataset_name)
    
    # 8. LaTeX tables (four versions: means/ttests × with_deg/without_deg)
    latex_mean_all_path = output_dir / 'multimodel_latex_table_means_all.tex'
    latex_mean_nodeg_path = output_dir / 'multimodel_latex_table_means_nodeg.tex'
    latex_ttest_all_path = output_dir / 'multimodel_latex_table_ttests_all.tex'
    latex_ttest_nodeg_path = output_dir / 'multimodel_latex_table_ttests_nodeg.tex'
    create_latex_tables(csv_path, latex_mean_all_path, latex_mean_nodeg_path,
                       latex_ttest_all_path, latex_ttest_nodeg_path, dataset_name)
    
    print("\n" + "=" * 80)
    print("✅ All plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()


