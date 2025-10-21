#!/usr/bin/env python3
"""
Script to extract basic statistics from all datasets.

Extracts:
- Number of cells (n_obs)
- Number of genes (n_vars)
- Number of perturbations (unique conditions)
- Cell type (first unique cell_type)

Saves results to a CSV file.

Usage:
    python get_dataset_stats.py
    python get_dataset_stats.py --output dataset_stats.csv
    python get_dataset_stats.py --num_workers 8
"""

import os
import yaml
import scanpy as sc
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')


def load_dataset_configs(config_dir):
    """Load all dataset configurations from YAML files"""
    configs = {}
    yaml_files = list(Path(config_dir).glob("*.yaml"))
    
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            if config and 'name' in config and 'data_path' in config:
                configs[config['name']] = config
    
    return configs


def extract_dataset_stats(args):
    """Extract statistics from a single dataset - worker function for multiprocessing"""
    dataset_name, config = args
    
    try:
        data_path = config['data_path']
        
        # Check if data file exists
        if not os.path.exists(data_path):
            return {
                'dataset': dataset_name,
                'n_cells': None,
                'n_genes': None,
                'n_perturbations': None,
                'cell_type': None,
                'status': 'File not found'
            }
        
        # Load the dataset
        adata = sc.read_h5ad(data_path)
        
        # Extract basic info
        n_cells = adata.shape[0]
        n_genes = adata.shape[1]
        
        # Get number of unique perturbations
        n_perturbations = len(adata.obs['condition'].unique())
        
        # Get cell type (first unique value)
        cell_type = adata.obs['cell_type'].unique()[0]
        
        return {
            'dataset': dataset_name,
            'n_cells': n_cells,
            'n_genes': n_genes,
            'n_perturbations': n_perturbations,
            'cell_type': cell_type,
            'status': 'Success'
        }
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}"
        print(f"   ⚠️  Error processing {dataset_name}: {error_msg}")
        return {
            'dataset': dataset_name,
            'n_cells': None,
            'n_genes': None,
            'n_perturbations': None,
            'cell_type': None,
            'status': f'Error: {error_msg}'
        }


def main(output_path='dataset_stats.csv', num_workers=None):
    """Main function to extract stats from all datasets"""
    
    # Load all dataset configs
    config_dir = "cellsimbench/configs/dataset"
    configs = load_dataset_configs(config_dir)
    
    print(f"Found {len(configs)} dataset configurations")
    print("=" * 60)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count() - 1, 8)
    
    print(f"Using {num_workers} parallel workers")
    print("=" * 60)
    
    # Prepare arguments for parallel processing
    dataset_args = [
        (dataset_name, config)
        for dataset_name, config in configs.items()
    ]
    
    # Process datasets in parallel
    results = []
    
    with mp.Pool(processes=num_workers) as pool:
        # Use imap_unordered for better progress tracking
        results = list(tqdm(
            pool.imap_unordered(extract_dataset_stats, dataset_args),
            total=len(dataset_args),
            desc="Extracting dataset stats"
        ))
    
    # Create dataframe from results
    df = pd.DataFrame(results)
    
    # Sort by dataset name
    df = df.sort_values('dataset').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total datasets: {len(df)}")
    print(f"Successfully processed: {(df['status'] == 'Success').sum()}")
    print(f"Failed: {(df['status'] != 'Success').sum()}")
    
    if (df['status'] != 'Success').any():
        print("\nFailed datasets:")
        for _, row in df[df['status'] != 'Success'].iterrows():
            print(f"   - {row['dataset']}: {row['status']}")
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Print some basic stats
    if (df['status'] == 'Success').any():
        success_df = df[df['status'] == 'Success']
        print("\nDataset statistics:")
        print(f"   Total cells: {success_df['n_cells'].sum():,}")
        print(f"   Average cells per dataset: {success_df['n_cells'].mean():.0f}")
        print(f"   Average genes per dataset: {success_df['n_genes'].mean():.0f}")
        print(f"   Total unique perturbations: {success_df['n_perturbations'].sum()}")
        print(f"   Unique cell types: {success_df['cell_type'].nunique()}")
    
    print("\n✨ Processing complete!")


if __name__ == "__main__":
    import argparse
    
    # Set multiprocessing start method for better compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    parser = argparse.ArgumentParser(
        description="Extract basic statistics from all datasets"
    )
    parser.add_argument('--output', type=str, default='dataset_stats.csv',
                        help='Output CSV file path (default: dataset_stats.csv)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1, max 8)')
    args = parser.parse_args()
    
    main(output_path=args.output, num_workers=args.num_workers)



