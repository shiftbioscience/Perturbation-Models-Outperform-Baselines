#!/usr/bin/env python
"""
Add interpolated duplicate baseline to existing processed adata files.

This script loads already-processed adata files and adds the new interpolated
duplicate baseline without regenerating all the data from scratch.

Usage:
    python add_interpolated_baseline.py <path_to_adata.h5ad>  # Process single file
    python add_interpolated_baseline.py --all                  # Process all datasets
    python add_interpolated_baseline.py --all --workers 8      # Process with 8 parallel workers
    python add_interpolated_baseline.py --all --force          # Force recompute all datasets
    python add_interpolated_baseline.py <path> --force         # Force recompute single file
"""

import scanpy as sc
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import argparse


def add_interpolated_baseline_to_adata(adata_path, force=False):
    """Add interpolated duplicate baseline to an existing processed adata file.
    
    Args:
        adata_path: Path to the processed adata file
        force: If True, recompute even if interpolated baseline already exists
    """
    print(f"\nProcessing: {adata_path}")
    
    # Load the adata
    adata = sc.read_h5ad(adata_path)
    
    # Get dataset name from the parent directory
    dataset_name = Path(adata_path).parent.name
    print(f"Dataset: {dataset_name}")
    
    # Check if already has interpolated baseline and not forcing
    if 'interpolated_duplicate_baseline' in adata.uns and not force:
        print(f"   Already has interpolated baseline (skipping, use --force to recompute)")
        return True
    
    # Check if we have the necessary baselines
    if 'technical_duplicate_second_half_baseline' not in adata.uns:
        raise ValueError("   No technical duplicate baseline found")
    
    if 'pvals_adj_df_dict' not in adata.uns:
        raise ValueError("   No p-values found")
    
    # Get the technical duplicate predictor (second half)
    tech_dup_predictor = adata.uns['technical_duplicate_second_half_baseline']
    
    # Get p-values dictionary
    pvals_dict = adata.uns['pvals_adj_df_dict']
    # pvals_dict = adata.uns['pvals_unadj_df_dict']

    
    # Find all fold-specific mean baselines
    fold_baselines = []
    for key in adata.uns.keys():
        if '_mean_baseline' in key and 'split_fold_' in key:
            fold_num = key.split('_')[2]  # Extract fold number from split_fold_X_mean_baseline
            fold_baselines.append((int(fold_num), key))
    
    if not fold_baselines:
        raise ValueError("No fold-specific mean baselines found")
    
    # Sort by fold number
    fold_baselines.sort(key=lambda x: x[0] if x[0] is not None else -1)
    
    # Collect all fold-specific interpolated baselines
    all_fold_baselines = []
    
    # Process each fold
    for fold_num, mean_baseline_key in fold_baselines:
        print(f"\n  Processing fold {fold_num}...")
        
        # Get the split column name for this fold
        split_col = f'split_fold_{fold_num}'
        
        # Get test set perturbations for this fold
        test_mask = adata.obs[split_col] == 'test'
        test_conditions = adata.obs.loc[test_mask, 'condition'].unique()
        
        # Add dataset prefix to match tech_dup_predictor index format
        test_conditions_with_prefix = [f"{dataset_name}_{cond}" for cond in test_conditions 
                                       if cond != 'control' and 'ctrl' not in cond.lower()]
        
        print(f"    Found {len(test_conditions_with_prefix)} test perturbations for fold {fold_num}")
        
        # Get fold-specific mean baseline
        mean_baseline_df = adata.uns[mean_baseline_key]
        
        # Initialize interpolated baseline DataFrame for this fold's test perturbations
        fold_interpolated = pd.DataFrame(
            index=test_conditions_with_prefix,
            columns=tech_dup_predictor.columns
        )
        
        # Process each test condition for this fold
        n_with_pvals = 0
        n_without_pvals = 0
        
        for condition_key in tqdm(test_conditions_with_prefix, 
                                  desc=f"    Computing interpolated baseline for fold {fold_num}", 
                                  leave=False):
            
            # Skip if not in tech_dup_predictor
            if condition_key not in tech_dup_predictor.index:
                raise ValueError(f"{condition_key} not in technical duplicate baseline")
                
            
            # Get the dataset name from mean baseline (it should have one row)
            dataset_key = mean_baseline_df.index[0]
            
            if condition_key in pvals_dict:
                # Get p-values and gene names for this condition
                pvals_list = pvals_dict[condition_key]
                
                # Check if we have the corresponding gene names
                if 'names_df_dict' in adata.uns and condition_key in adata.uns['names_df_dict']:
                    names_list = adata.uns['names_df_dict'][condition_key]
                    
                    # Create a Series with genes as index and p-values as values
                    pvals_series = pd.Series(pvals_list, index=names_list)
                    
                    # Reorder to match the column order of tech_dup_predictor
                    # Fill missing genes with p-value of 1 (alpha = 0, use mean baseline)
                    pvals_ordered = pvals_series.reindex(tech_dup_predictor.columns, fill_value=1.0)
                    pvals = pvals_ordered.values
                else:
                    # If we don't have names_df_dict, can't properly align p-values
                    raise ValueError(f"names_df_dict not found or missing entry for {condition_key}")
                
                # Calculate alpha = 1 - pval for each gene
                # Higher alpha (lower p-value) = more weight to technical duplicate
                alphas = 1 - pvals
                
                # Handle any NaN p-values (set alpha to 0, use mean baseline)
                alphas = np.nan_to_num(alphas, nan=0.0)
                
                n_with_pvals += 1
                
            else:
                # No p-values available for this condition, use mean baseline only (alpha = 0)
                alphas = np.zeros(len(tech_dup_predictor.columns))
                n_without_pvals += 1
            
            # Perform interpolation: alpha * tech_dup + (1-alpha) * mean
            if dataset_key in mean_baseline_df.index:
                mean_values = mean_baseline_df.loc[dataset_key].values
            else:
                raise ValueError(f"{dataset_key} not in mean baseline")
            
            interpolated_values = (
                alphas * tech_dup_predictor.loc[condition_key].values + 
                (1 - alphas) * mean_values
            )
            
            # Store the interpolated values
            fold_interpolated.loc[condition_key] = interpolated_values
        
        # Clean up NaN rows (conditions not in tech_dup_predictor)
        fold_interpolated = fold_interpolated.dropna(how='all')
        
        print(f"    Interpolated baseline computed for {len(fold_interpolated)} conditions")
        print(f"    Conditions with p-values: {n_with_pvals}")
        print(f"    Conditions without p-values: {n_without_pvals}")
        
        # Add to collection
        all_fold_baselines.append(fold_interpolated)
    
    # Combine all fold baselines into one dataframe
    print("\n  Combining baselines from all folds...")
    combined_interpolated = pd.concat(all_fold_baselines, axis=0)
    combined_interpolated = combined_interpolated.astype(float)
    
    print(f"  Total interpolated baseline: {len(combined_interpolated)} conditions")
    
    # Store the combined interpolated baseline
    adata.uns['interpolated_duplicate_baseline'] = combined_interpolated

    
    print(f"  Saving updated adata to: {adata_path}")
    adata.write_h5ad(adata_path)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add interpolated duplicate baseline to processed adata files"
    )
    parser.add_argument(
        'adata_path',
        nargs='?',
        help='Path to single adata file to process'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all datasets that need interpolated baseline'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recomputation even if interpolated baseline already exists'
    )
    
    args = parser.parse_args()
    
    # Check for conflicting arguments
    if args.all and args.adata_path:
        parser.error("Cannot specify both --all and a specific adata path")
    
    if not args.all and not args.adata_path:
        parser.error("Must specify either --all or a path to an adata file")
    
    if args.all:
        # Process all datasets
        import yaml
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial
        
        config_dir = Path('cellsimbench/configs/dataset')
        yaml_files = list(config_dir.glob('*.yaml'))
        
        datasets_to_process = []
        for yaml_path in yaml_files:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            data_path = config.get('data_path')
            if not data_path or not Path(data_path).exists():
                continue

            # if not 'sunshine' in data_path and not 'wessels' in data_path:
            #     continue
            
            # If forcing, process all datasets with valid data paths
            # Otherwise, check if already has interpolated baseline
            if args.force:
                datasets_to_process.append(data_path)
                print(f"  ⟳ {yaml_path.stem} - will reprocess (force mode)")
            else:
                # Load adata to check if it has interpolated baseline
                try:
                    adata = sc.read_h5ad(data_path)
                    if 'interpolated_duplicate_baseline' in adata.uns:
                        print(f"  ✓ {yaml_path.stem} - already has interpolated baseline")
                    else:
                        datasets_to_process.append(data_path)
                        print(f"  ✗ {yaml_path.stem} - needs interpolated baseline")
                except Exception as e:
                    print(f"  ⚠ {yaml_path.stem} - error checking: {e}")
        
        if not datasets_to_process:
            print("All datasets already have interpolated baselines!")
            return
            
        
        print(f"\nProcessing {len(datasets_to_process)} datasets with {args.workers} workers...")
        # Process in parallel with force flag
        process_func = partial(add_interpolated_baseline_to_adata, force=args.force)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_func, path): path 
                      for path in datasets_to_process}
            
            for future in as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                    print(f"✓ Completed: {Path(path).parent.name}")
                except Exception as e:
                    print(f"✗ Failed: {Path(path).parent.name} - {e}")
        
        print("\nAll processing complete!")
        
    else:
        # Process single file
        adata_path = args.adata_path
        
        if not Path(adata_path).exists():
            print(f"Error: File not found: {adata_path}")
            sys.exit(1)
        
        try:
            success = add_interpolated_baseline_to_adata(adata_path, force=args.force)
            if success:
                print("\nSuccessfully added interpolated duplicate baseline!")
            else:
                print("\nFailed to add interpolated duplicate baseline.")
                sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
