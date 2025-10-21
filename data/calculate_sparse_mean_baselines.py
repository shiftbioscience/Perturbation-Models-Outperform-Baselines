#!/usr/bin/env python3
"""
Script to calculate sparsity-matched mean baselines for all datasets.

This creates a fair comparison with technical duplicate baselines by 
matching the sampling sparsity. For each test perturbation with N cells,
we subsample N/2 cells from training perturbations to match the sparsity
of the technical duplicate (which uses N/2 cells for its prediction).

Creates three versions with different random seeds (0, 1, 2) to assess reproducibility.
"""

import os
import yaml
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import warnings
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

def load_dataset_configs(config_dir):
    """Load all dataset configurations from YAML files"""
    configs = {}
    yaml_files = list(Path(config_dir).glob("*.yaml"))
    
    for yaml_file in yaml_files:
        # if not 'gwps' in str(yaml_file):
        #     continue
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            if config and 'name' in config and 'data_path' in config:
                configs[config['name']] = config
    
    return configs

def calculate_universal_sparse_mean_baseline(adata, dataset_name, seed=0):
    """
    Calculate a single sparsity-matched mean baseline for all perturbations across all folds.
    
    Args:
        adata: AnnData object with the dataset
        dataset_name: Name of the dataset for prefixing perturbations
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with sparse mean baseline for all test perturbations
    """
    
    # Set the seed for this entire calculation
    np.random.seed(seed)
    
    # Detect all split columns
    split_columns = [col for col in adata.obs.columns if col.startswith('split_fold_')]
    
    if len(split_columns) == 0:
        print(f"     Warning: No split_fold columns found for {dataset_name}")
        return pd.DataFrame()
    
    # Initialize universal result DataFrame
    sparse_mean_baseline = pd.DataFrame(columns=adata.var_names)
    
    # Process each fold
    for split_col in split_columns:
        # Get test conditions for THIS fold
        test_mask = adata.obs[split_col] == 'test'
        train_mask = adata.obs[split_col] == 'train'
            
        test_conditions = adata.obs.loc[test_mask, 'condition'].unique()
        
        # Filter out control conditions
        test_conditions = [c for c in test_conditions 
                          if 'control' not in str(c).lower() and 'ctrl' not in str(c).lower()]

        
        # Get training data for THIS fold (non-control)
        train_adata = adata[train_mask]
        train_non_ctrl_mask = ~train_adata.obs['condition'].astype(str).str.lower().str.contains('control|ctrl')
        train_non_ctrl = train_adata[train_non_ctrl_mask]
        
        # Calculate sparse baseline for each test condition
        for test_condition in tqdm(test_conditions, 
                                   desc=f"     Processing {split_col} (seed {seed})", 
                                   leave=False):

            # Get number of cells for this test perturbation
            test_cells = adata.obs[
                (adata.obs['condition'] == test_condition) & test_mask
            ]
            n_cells = len(test_cells)
                
            n_subsample = n_cells // 2  # Match technical duplicate split            
            
            # Calculate sparse mean from all training cells (pooled)            
            # Randomly subsample n_subsample cells from all training cells
            indices = np.random.choice(len(train_non_ctrl), 
                                        size=n_subsample, 
                                        replace=False)
            subsample = train_non_ctrl[indices]
            
            # Calculate mean expression from subsample
            subsample_mean = subsample.X.mean(axis=0)
            if hasattr(subsample_mean, 'A1'):
                subsample_mean = subsample_mean.A1
            elif hasattr(subsample_mean, 'A'):
                subsample_mean = subsample_mean.A[0]
                
            sparse_mean_baseline.loc[f'{dataset_name}_{test_condition}'] = subsample_mean

    
    return sparse_mean_baseline.astype(float)

def process_single_dataset(args):
    """Process a single dataset - worker function for multiprocessing"""
    dataset_name, config, force_recompute, SEEDS = args
    
    try:
        data_path = config['data_path']
        output_path = data_path.replace('.h5ad', f'.sparse_mean_baselines.h5ad')
        
        # Check if we should process this dataset
        # Process if output doesn't exist OR force_recompute is True
        if not os.path.exists(data_path):
            return (dataset_name, False, "Data file not found")
        
        # Check if already processed
        # Load the dataset to check if baselines already exist
        if not force_recompute:
            adata = sc.read_h5ad(data_path)
            has_all_baselines = all(
                f'sparse_mean_baseline_seed_{seed}' in adata.uns 
                for seed in SEEDS
            )
            if has_all_baselines:
                return (dataset_name, True, "Already has baselines (skipped)")
            # If not all baselines exist, we'll process it
        
        # Load the dataset (reload if needed)
        if force_recompute:
            adata = sc.read_h5ad(data_path)
        
        print(f"\n📊 Processing {dataset_name}...")
        print(f"   Path: {data_path}")
        print(f"   Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        # Detect split columns
        split_columns = [col for col in adata.obs.columns 
                        if col.startswith('split_fold_')]
        
        print(f"   Found {len(split_columns)} fold(s)")
        
        if len(split_columns) == 0:
            return (dataset_name, False, "No split_fold columns found")
        
        # Calculate sparse mean baseline with three different seeds
        for seed in SEEDS:
            print(f"   Calculating sparse mean baseline with seed {seed}...")
            sparse_baseline = calculate_universal_sparse_mean_baseline(
                adata, dataset_name, seed=seed
            )
            
            # Save with seed-specific key
            key = f'sparse_mean_baseline_seed_{seed}'
            adata.uns[key] = sparse_baseline
            print(f"     ✓ Created {key} for {len(sparse_baseline)} conditions")
        
        # Save to output path first, then rename
        adata.write_h5ad(output_path)
        print(f"   Saved to {output_path}")
        
        # Rename output_path to data_path (overwrite)
        os.rename(output_path, data_path)
        print(f"   Renamed to {data_path}")
        
        return (dataset_name, True, None)
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"   ⚠️  Error processing {dataset_name}: {error_msg}")
        return (dataset_name, False, error_msg)

def process_all_datasets(force_recompute=False, num_workers=None):
    """Main function to process all datasets with multiple seeds"""
    
    # Define seeds for reproducibility analysis
    SEEDS = [0, 1, 2]
    
    # Load all dataset configs
    config_dir = "cellsimbench/configs/dataset"
    configs = load_dataset_configs(config_dir)

    print(f"Found {len(configs)} dataset configurations")
    print("=" * 60)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count() - 1, 1024)  # Default to CPU count - 1, max 8
    
    print(f"Using {num_workers} parallel workers")
    print("=" * 60)
    
    # Prepare arguments for parallel processing
    dataset_args = [
        (dataset_name, config, force_recompute, SEEDS)
        for dataset_name, config in configs.items()
    ]
    
    # Process datasets in parallel
    successful = []
    failed = []
    
    with mp.Pool(processes=num_workers) as pool:
        # Use imap_unordered for better progress tracking
        results = list(tqdm(
            pool.imap_unordered(process_single_dataset, dataset_args),
            total=len(dataset_args),
            desc="Processing datasets"
        ))
    
    # Collect results
    for dataset_name, success, error_msg in results:
        if success:
            if error_msg and "skipped" in error_msg.lower():
                # Dataset was skipped (already processed)
                pass  # Don't count as successful since we didn't process it
            else:
                successful.append(dataset_name)
        else:
            failed.append((dataset_name, error_msg or "Unknown error"))
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully processed: {len(successful)} datasets")
    if successful:
        for name in successful:
            print(f"   - {name}")
    
    if failed:
        print(f"\n⚠️  Failed/skipped: {len(failed)} datasets")
        for name, reason in failed:
            print(f"   - {name}: {reason}")
    
    print("\n✨ Processing complete!")

if __name__ == "__main__":
    import argparse
    
    # Set multiprocessing start method for better compatibility
    # 'spawn' is safer for scientific computing libraries like numpy/scanpy
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    parser = argparse.ArgumentParser(
        description="Calculate sparsity-matched mean baselines for all datasets"
    )
    parser.add_argument('--force_recompute', action='store_true',
                        help='Force recomputation even if baselines already exist')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1, max 1024)')
    args = parser.parse_args()
    
    process_all_datasets(force_recompute=args.force_recompute, 
                         num_workers=args.num_workers)
