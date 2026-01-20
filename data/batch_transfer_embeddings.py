#!/usr/bin/env python3
"""
Batch transfer gene embeddings from reference AnnData to all datasets.

This script transfers gene embeddings from a reference dataset (norman19) to all
other datasets specified in the config files.

Usage:
    python batch_transfer_embeddings.py                        # Process all datasets
    python batch_transfer_embeddings.py --workers 8            # Process with 8 parallel workers
    python batch_transfer_embeddings.py --force                # Force recompute all datasets
    python batch_transfer_embeddings.py --reference <path>     # Use custom reference file
"""

import scanpy as sc
import yaml
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import sys


def transfer_gene_embeddings(
    input_path: str,
    reference_adata_path: str,
    output_path: str
):
    """Transfer gene embeddings from reference AnnData to target AnnData.
    
    Args:
        input_path: Path to input h5ad file (target)
        reference_adata_path: Path to reference h5ad file containing embeddings
        output_path: Path to save output h5ad file with transferred embeddings
    """
    print('='*100)
    print("Transferring gene embeddings from reference")
    print('='*100)
    print(f"Input data: {input_path}")
    print(f"Reference data: {reference_adata_path}")
    print(f"Output data: {output_path}")
    print('='*100)
    
    # Load input and reference data
    print("Loading input AnnData...")
    adata = sc.read_h5ad(input_path)
    
    print("Loading reference AnnData...")
    ref_adata = sc.read_h5ad(reference_adata_path)
    
    # Find all embedding keys in reference data
    embedding_keys = [key for key in ref_adata.uns.keys() if key.startswith('embeddings_')]
    
    if not embedding_keys:
        print("Warning: No embedding keys found in reference data")
        print("Available keys in reference uns:", list(ref_adata.uns.keys()))
        # Just save the input as output if no embeddings found
        adata.write_h5ad(output_path)
        return
    
    print(f"Found {len(embedding_keys)} embedding keys in reference data:")
    for key in embedding_keys:
        print(f"  - {key}")
    
    # Transfer all embedding keys from reference to target
    embeddings_transferred = 0
    for key in embedding_keys:
        try:
            adata.uns[key] = ref_adata.uns[key].copy()
            embeddings_transferred += 1
            
            # Print info about transferred embedding
            emb_data = ref_adata.uns[key]
            print(f"Transferred {key}: {emb_data.shape[0]} genes x {emb_data.shape[1]} dimensions")
            
        except Exception as e:
            print(f"Warning: Failed to transfer {key}: {e}")
    
    # Save result
    adata.write_h5ad(output_path)
    
    print('='*100)
    print(f"Transfer complete: {embeddings_transferred}/{len(embedding_keys)} embeddings transferred")
    print('='*100)

def check_has_embeddings(adata_path):
    """Check if an adata file already has gene embeddings.
    
    Args:
        adata_path: Path to the adata file
        
    Returns:
        bool: True if embeddings are present, False otherwise
    """
    try:
        adata = sc.read_h5ad(adata_path)
        # Check for any keys starting with 'embeddings_'
        embedding_keys = [key for key in adata.uns.keys() if key.startswith('embeddings_')]
        return len(embedding_keys) > 0
    except Exception as e:
        print(f"    ⚠ Error checking embeddings: {e}")
        return False


def transfer_embeddings_single_dataset(adata_path, reference_path, force=False):
    """Transfer embeddings to a single dataset.
    
    Args:
        adata_path: Path to the target adata file
        reference_path: Path to the reference adata file with embeddings
        force: If True, transfer even if embeddings already exist
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\nProcessing: {adata_path}")
    
    # Get dataset name from the parent directory
    dataset_name = Path(adata_path).parent.name
    print(f"Dataset: {dataset_name}")
    
    # Check if already has embeddings and not forcing
    if not force and check_has_embeddings(adata_path):
        print(f"   Already has embeddings (skipping, use --force to recompute)")
        return True
    
    # Check if file exists
    if not Path(adata_path).exists():
        raise ValueError(f"   Data file not found: {adata_path}")
    
    if not Path(reference_path).exists():
        raise ValueError(f"   Reference file not found: {reference_path}")
    
    # Create temporary output path
    output_path = str(adata_path).replace('.h5ad', '.with_embeddings.h5ad')
    
    try:
        # Transfer embeddings
        print("   Transferring embeddings...")
        transfer_gene_embeddings(
            input_path=adata_path,
            reference_adata_path=reference_path,
            output_path=output_path
        )
        
        # Replace original file with new file
        import os
        os.rename(output_path, adata_path)
        print(f"   ✓ Successfully transferred embeddings to {dataset_name}")
        
        return True
        
    except Exception as e:
        # Clean up temporary file if it exists
        if Path(output_path).exists():
            Path(output_path).unlink()
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="Batch transfer gene embeddings from reference to all datasets"
    )
    parser.add_argument(
        '--reference',
        default='data/norman19/norman19_processed_complete.h5ad',
        help='Path to reference adata file with embeddings (default: norman19)'
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
        help='Force retransfer even if embeddings already exist'
    )
    
    args = parser.parse_args()
    
    # Resolve reference path
    reference_path = Path(args.reference)
    if not reference_path.exists():
        print(f"Error: Reference file not found: {reference_path}")
        sys.exit(1)
    
    print(f"Reference file: {reference_path}")
    print(f"Workers: {args.workers}")
    print(f"Force mode: {args.force}")
    print("=" * 80)
    
    # Load all dataset configs
    config_dir = Path('cellsimbench/configs/dataset')
    yaml_files = list(config_dir.glob('*.yaml'))
    
    datasets_to_process = []
    
    print("\nScanning datasets...")
    for yaml_path in yaml_files:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        data_path = config.get('data_path')
        if not data_path or not Path(data_path).exists():
            continue
        
        # If forcing, process all datasets with valid data paths
        # Otherwise, check if already has embeddings
        if args.force:
            datasets_to_process.append(data_path)
            print(f"  ⟳ {yaml_path.stem} - will reprocess (force mode)")
        else:
            # Check if it has embeddings
            if check_has_embeddings(data_path):
                print(f"  ✓ {yaml_path.stem} - already has embeddings")
            else:
                datasets_to_process.append(data_path)
                print(f"  ✗ {yaml_path.stem} - needs embeddings")
    
    if not datasets_to_process:
        print("\nAll datasets already have embeddings!")
        return
    
    print(f"\nProcessing {len(datasets_to_process)} datasets with {args.workers} workers...")
    print("=" * 80)
    
    # Process in parallel with force flag
    process_func = partial(
        transfer_embeddings_single_dataset, 
        reference_path=str(reference_path),
        force=args.force
    )
    
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
    
    print("\n✨ Processing complete!")


if __name__ == "__main__":
    main()
