#!/usr/bin/env python
"""
Add ground truth DEGs to existing processed adata files.

This script calculates DEGs from the FIRST HALF (ground truth) of technical duplicate
splits and stores them separately from the second half DEGs used for interpolation.
This is important because metrics should be calculated based on error vs the ground
truth, not the predictor half.

The ground truth DEGs are stored with "_gt" suffix:
  - deg_gene_dict_gt
  - names_df_dict_gt
  - pvals_adj_df_dict_gt
  - pvals_unadj_df_dict_gt
  - scores_df_dict_gt

Usage:
    python add_ground_truth_degs.py <path_to_adata.h5ad>  # Process single file
    python add_ground_truth_degs.py --all                  # Process all datasets
    python add_ground_truth_degs.py --all --workers 8      # Process with 8 parallel workers
    python add_ground_truth_degs.py --all --force          # Force recompute all datasets
    python add_ground_truth_degs.py <path> --force         # Force recompute single file
"""

import scanpy as sc
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import argparse


def calculate_ground_truth_degs(adata, dataset_name, min_cells=4):
    """
    Calculate differential expression genes using only FIRST HALF (ground truth) of technical duplicate split.
    
    Args:
        adata: AnnData object with tech_dup_split column
        dataset_name: Dataset name for condition prefixing
        min_cells: Minimum number of cells required per perturbation
        
    Returns:
        Dictionary containing DEG results to store in adata.uns
    """
    print("Calculating ground truth DEGs using first half of technical duplicate split...")
    
    # Filter to only use first_half cells (GROUND TRUTH)
    adata_first_half = adata[adata.obs['tech_dup_split'] == 'first_half'].copy()
    print(f"Using {adata_first_half.shape[0]} cells from first half for ground truth DEG calculation")
    
    # Get non-control conditions
    all_conditions = adata.obs['condition'].unique()
    non_control_conditions = [cond for cond in all_conditions 
                             if 'control' not in cond.lower() and 'ctrl' not in cond.lower()]
    
    # Filter perturbations with enough cells in first half
    pert_counts = adata_first_half.obs['condition'].value_counts()
    valid_perts = pert_counts[(pert_counts >= min_cells) & 
                              (pert_counts.index.isin(non_control_conditions))].index
    adata_deg = adata_first_half[adata_first_half.obs['condition'].isin(valid_perts)].copy()
    
    print(f"Calculating DEGs for {len(valid_perts)} perturbations with ≥{min_cells} cells in first half")
    
    if len(valid_perts) == 0:
        print("WARNING: No valid perturbations found in first half")
        return None
    
    # Calculate DEGs vs rest
    print("Computing DEGs vs rest...")
    sc.tl.rank_genes_groups(adata_deg, 'condition', method='t-test_overestim_var', reference='rest')
    
    # Store results - including both adjusted and unadjusted p-values
    names_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["names"])
    pvals_adj_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["pvals_adj"])
    pvals_unadj_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["pvals"])  # Unadjusted p-values
    scores_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["scores"])
    
    # Create DEG dictionary (only significant genes with p_adj < 0.05)
    deg_dict = {}
    for pert in tqdm(adata_deg.obs['condition'].unique(), desc="Processing ground truth DEGs"):
        if pert == 'control' or 'ctrl' in pert:
            continue
        pert_degs = names_df[pert]
        pert_degs_sig = pert_degs[pvals_adj_df[pert] < 0.05]  # Still use adjusted for significance
        deg_dict[f'{dataset_name}_{pert}'] = pert_degs_sig.tolist()
    
    # Convert dataframes to proper format for h5ad storage
    # Create flattened dictionaries that can be saved in h5ad format
    names_df_dict_final = {}
    pvals_adj_df_dict_final = {}
    pvals_unadj_df_dict_final = {}
    scores_df_dict_final = {}
    
    for pert in names_df.columns:
        if pert == 'control' or 'ctrl' in pert:
            continue
        # Convert to lists to avoid h5ad serialization issues
        names_df_dict_final[f'{dataset_name}_{pert}'] = names_df[pert].tolist()
        pvals_adj_df_dict_final[f'{dataset_name}_{pert}'] = pvals_adj_df[pert].tolist()
        pvals_unadj_df_dict_final[f'{dataset_name}_{pert}'] = pvals_unadj_df[pert].tolist()
        scores_df_dict_final[f'{dataset_name}_{pert}'] = scores_df[pert].tolist()
    
    print(f"Calculated ground truth DEGs for {len(deg_dict)} perturbations using first half only")
    
    return {
        'deg_gene_dict_gt': deg_dict,
        'names_df_dict_gt': names_df_dict_final,
        'pvals_adj_df_dict_gt': pvals_adj_df_dict_final,
        'pvals_unadj_df_dict_gt': pvals_unadj_df_dict_final,
        'scores_df_dict_gt': scores_df_dict_final
    }


def add_ground_truth_degs_to_adata(adata_path, force=False, min_cells=4):
    """
    Add ground truth DEGs to an existing processed adata file.
    
    Args:
        adata_path: Path to the processed adata file
        force: If True, recompute even if ground truth DEGs already exist
        min_cells: Minimum cells required per perturbation
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\nProcessing: {adata_path}")
    
    # Load the adata
    adata = sc.read_h5ad(adata_path)
    
    # Get dataset name from the parent directory
    dataset_name = Path(adata_path).parent.name
    print(f"Dataset: {dataset_name}")
    
    # Check if already has ground truth DEGs and not forcing
    if 'deg_gene_dict_gt' in adata.uns and not force:
        print(f"   Already has ground truth DEGs (skipping, use --force to recompute)")
        return True
    
    # Check if we have the necessary tech_dup_split column
    if 'tech_dup_split' not in adata.obs.columns:
        raise ValueError("   No tech_dup_split column found in adata.obs")
    
    # Check that first_half exists
    split_counts = adata.obs['tech_dup_split'].value_counts()
    if 'first_half' not in split_counts:
        raise ValueError("   'first_half' not found in tech_dup_split column")
    
    print(f"   First half cells: {split_counts.get('first_half', 0)}")
    print(f"   Second half cells: {split_counts.get('second_half', 0)}")
    
    # Calculate ground truth DEGs
    gt_deg_results = calculate_ground_truth_degs(adata, dataset_name, min_cells=min_cells)
    
    if gt_deg_results is None:
        raise ValueError("   Failed to calculate ground truth DEGs")
    
    # Store results in adata.uns
    for key, value in gt_deg_results.items():
        adata.uns[key] = value
        print(f"   Stored {key} with {len(value)} entries")
    
    # Create temporary output path
    output_path = str(adata_path).replace('.h5ad', '.with_gt_degs.h5ad')
    
    try:
        # Save to temporary file
        print(f"   Saving updated adata to temporary file...")
        adata.write_h5ad(output_path)
        
        # Replace original file with new file
        import os
        os.rename(output_path, adata_path)
        print(f"   ✓ Successfully added ground truth DEGs to {dataset_name}")
        
        return True
        
    except Exception as e:
        # Clean up temporary file if it exists
        if Path(output_path).exists():
            Path(output_path).unlink()
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="Add ground truth DEGs to processed adata files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python add_ground_truth_degs.py data/adamson16/adamson16_processed_complete.h5ad
  
  # Process all datasets
  python add_ground_truth_degs.py --all
  
  # Process with more workers
  python add_ground_truth_degs.py --all --workers 8
  
  # Force recomputation
  python add_ground_truth_degs.py --all --force
  
  # Custom minimum cells threshold
  python add_ground_truth_degs.py --all --min-cells 6
        """
    )
    parser.add_argument(
        'adata_path',
        nargs='?',
        help='Path to single adata file to process'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all datasets that need ground truth DEGs'
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
        help='Force recomputation even if ground truth DEGs already exist'
    )
    parser.add_argument(
        '--min-cells',
        type=int,
        default=4,
        help='Minimum cells required per perturbation per half (default: 4)'
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
        
        config_dir = Path('src/cellsimbench/configs/dataset')
        yaml_files = list(config_dir.glob('*.yaml'))
        
        datasets_to_process = []
        for yaml_path in yaml_files:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # if not 'gwps' in str(yaml_path):
            #     continue
            
            data_path = config.get('data_path')
            if not data_path or not Path(data_path).exists():
                continue
            
            # If forcing, process all datasets with valid data paths
            # Otherwise, check if already has ground truth DEGs
            if args.force:
                datasets_to_process.append(data_path)
                print(f"  ⟳ {yaml_path.stem} - will reprocess (force mode)")
            else:
                # Load adata to check if it has ground truth DEGs
                try:
                    adata = sc.read_h5ad(data_path)
                    if 'deg_gene_dict_gt' in adata.uns:
                        print(f"  ✓ {yaml_path.stem} - already has ground truth DEGs")
                    else:
                        datasets_to_process.append(data_path)
                        print(f"  ✗ {yaml_path.stem} - needs ground truth DEGs")
                except Exception as e:
                    print(f"  ⚠ {yaml_path.stem} - error checking: {e}")
        
        if not datasets_to_process:
            print("\nAll datasets already have ground truth DEGs!")
            return
        
        print(f"\nProcessing {len(datasets_to_process)} datasets with {args.workers} workers...")
        print(f"Minimum cells per perturbation: {args.min_cells}")
        print("=" * 80)
        
        # Process in parallel with force flag
        process_func = partial(add_ground_truth_degs_to_adata, 
                              force=args.force, 
                              min_cells=args.min_cells)
        
        successful = []
        failed = []
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_func, path): path 
                      for path in datasets_to_process}
            
            for future in as_completed(futures):
                path = futures[future]
                dataset_name = Path(path).parent.name
                try:
                    future.result()
                    successful.append(dataset_name)
                    print(f"✓ Completed: {dataset_name}")
                except Exception as e:
                    failed.append((dataset_name, str(e)))
                    print(f"✗ Failed: {dataset_name} - {e}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"✅ Successfully processed: {len(successful)} datasets")
        if successful:
            for name in successful:
                print(f"   - {name}")
        
        if failed:
            print(f"\n⚠️  Failed: {len(failed)} datasets")
            for name, error in failed:
                print(f"   - {name}: {error}")
        
        print("\n✨ All processing complete!")
        
    else:
        # Process single file
        adata_path = args.adata_path
        
        if not Path(adata_path).exists():
            print(f"Error: File not found: {adata_path}")
            sys.exit(1)
        
        try:
            success = add_ground_truth_degs_to_adata(adata_path, 
                                                     force=args.force,
                                                     min_cells=args.min_cells)
            if success:
                print("\n✅ Successfully added ground truth DEGs!")
            else:
                print("\n❌ Failed to add ground truth DEGs.")
                sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()

