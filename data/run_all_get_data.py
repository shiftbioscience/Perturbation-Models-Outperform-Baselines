#!/usr/bin/env python3
"""
Script to run all get_data.py scripts for all datasets in parallel.

This script processes all datasets by running their respective get_data.py scripts,
which includes downloading data, preprocessing, calculating baselines, and saving
the processed results.

Usage:
    python run_all_get_data.py                      # Run all datasets
    python run_all_get_data.py --force              # Force recompute even if output exists
    python run_all_get_data.py --workers 8          # Use 8 parallel workers
    python run_all_get_data.py --datasets norman19 wessels23  # Run specific datasets only
"""

import os
import sys
import yaml
import subprocess
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import argparse
import time
from datetime import datetime, timedelta
import traceback


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


def check_dataset_processed(config):
    """Check if a dataset has already been processed"""
    data_path = config.get('data_path', '')
    
    # Check if the output file exists
    if os.path.exists(data_path):
        # Check file size to ensure it's not empty
        file_size = os.path.getsize(data_path)
        if file_size > 1000:  # At least 1KB
            return True
    
    return False


def run_get_data_script(dataset_name, force_recompute=False):
    """Run the get_data.py script for a single dataset
    
    Args:
        dataset_name: Name of the dataset to process
        force_recompute: If True, rerun even if output exists
        
    Returns:
        Tuple of (dataset_name, success, error_msg, duration)
    """
    start_time = time.time()
    
    try:
        # Construct path to get_data.py script
        script_path = Path(f"data/{dataset_name}/get_data.py")
        
        if not script_path.exists():
            return (dataset_name, False, f"Script not found: {script_path}", 0)
        
        # Load config to check if already processed
        config_path = Path(f"cellsimbench/configs/dataset/{dataset_name}.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check if already processed (unless forcing)
            if not force_recompute and check_dataset_processed(config):
                duration = time.time() - start_time
                return (dataset_name, True, "Already processed (use --force to rerun)", duration)
        
        print(f"\n🚀 Starting {dataset_name}...")
        print(f"   Script: {script_path}")
        
        # Run the script using subprocess
        # Use python to ensure we're using the same environment
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=".",  # Run from project root
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   ✅ {dataset_name} completed in {duration:.1f}s")
            
            # Check if output was actually created
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    data_path = config.get('data_path', '')
                    if os.path.exists(data_path):
                        file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
                        print(f"   📊 Output: {data_path} ({file_size_mb:.1f} MB)")
            
            return (dataset_name, True, None, duration)
        else:
            error_msg = f"Script failed with return code {result.returncode}"
            if result.stderr:
                # Get last 500 chars of stderr for concise error reporting
                stderr_excerpt = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                error_msg += f"\nError: {stderr_excerpt}"
            print(f"   ❌ {dataset_name} failed: {error_msg}")
            return (dataset_name, False, error_msg, duration)
            
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"   ⚠️ Error processing {dataset_name}: {str(e)}")
        return (dataset_name, False, error_msg, duration)


def process_dataset_wrapper(args):
    """Wrapper function for multiprocessing"""
    dataset_name, force_recompute = args
    return run_get_data_script(dataset_name, force_recompute)


def run_all_datasets(force_recompute=False, num_workers=None, specific_datasets=None):
    """Main function to run all dataset processing scripts in parallel
    
    Args:
        force_recompute: If True, rerun all scripts even if outputs exist
        num_workers: Number of parallel workers (None = auto-detect)
        specific_datasets: List of specific dataset names to run (None = all)
    """
    
    # Load all dataset configs
    config_dir = "cellsimbench/configs/dataset"
    all_configs = load_dataset_configs(config_dir)
    
    # Filter to specific datasets if requested
    if specific_datasets:
        configs = {k: v for k, v in all_configs.items() if k in specific_datasets}
        if len(configs) < len(specific_datasets):
            missing = set(specific_datasets) - set(configs.keys())
            print(f"⚠️  Warning: Requested datasets not found: {missing}")
    else:
        configs = all_configs
    
    print(f"Found {len(configs)} dataset(s) to process")
    print("=" * 60)
    
    # List datasets
    for dataset_name in sorted(configs.keys()):
        status = "🔄 Will process"
        if not force_recompute and check_dataset_processed(configs[dataset_name]):
            status = "✅ Already processed"
        print(f"  {dataset_name:20s} {status}")
    
    print("=" * 60)
    
    # Determine number of workers
    if num_workers is None:
        # Default to CPU count - 1, max 8 for memory considerations
        num_workers = min(mp.cpu_count() - 1, 8)
    
    print(f"Using {num_workers} parallel workers")
    print("=" * 60)
    
    # Prepare arguments for parallel processing
    dataset_args = [
        (dataset_name, force_recompute)
        for dataset_name in sorted(configs.keys())
    ]
    
    # Track overall progress
    start_time = time.time()
    successful = []
    failed = []
    skipped = []
    
    # Process datasets in parallel
    with mp.Pool(processes=num_workers) as pool:
        # Use imap_unordered for better progress tracking
        results = list(tqdm(
            pool.imap_unordered(process_dataset_wrapper, dataset_args),
            total=len(dataset_args),
            desc="Processing datasets"
        ))
    
    # Collect and analyze results
    total_duration = 0
    for dataset_name, success, error_msg, duration in results:
        total_duration += duration
        
        if success:
            if error_msg and "already processed" in error_msg.lower():
                skipped.append((dataset_name, duration))
            else:
                successful.append((dataset_name, duration))
        else:
            failed.append((dataset_name, error_msg, duration))
    
    # Print summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    
    if successful:
        print(f"\n✅ Successfully processed: {len(successful)} dataset(s)")
        for name, duration in sorted(successful):
            print(f"   - {name:20s} ({duration:6.1f}s)")
    
    if skipped:
        print(f"\n⏭️  Skipped (already processed): {len(skipped)} dataset(s)")
        for name, duration in sorted(skipped):
            print(f"   - {name:20s}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} dataset(s)")
        for name, error, duration in failed:
            # Show first line of error only in summary
            error_first_line = error.split('\n')[0] if error else "Unknown error"
            print(f"   - {name:20s} ({duration:6.1f}s): {error_first_line[:50]}")
    
    # Performance statistics
    print("\n" + "=" * 60)
    print("PERFORMANCE STATISTICS")
    print("=" * 60)
    print(f"Total wall time:        {elapsed_time:.1f}s ({str(timedelta(seconds=int(elapsed_time)))})")
    print(f"Total processing time:  {total_duration:.1f}s ({str(timedelta(seconds=int(total_duration)))})")
    print(f"Speedup from parallel:  {total_duration/elapsed_time:.1f}x")
    print(f"Average time/dataset:   {total_duration/len(results):.1f}s")
    
    # Final status
    print("\n" + "=" * 60)
    if failed:
        print("⚠️  Processing completed with errors")
        print("   Run with --force to retry failed datasets")
    else:
        print("✨ All datasets processed successfully!")
    
    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run all dataset get_data.py scripts in parallel"
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force recomputation even if outputs already exist'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=None,
        help='Number of parallel workers (default: CPU count - 1, max 8)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specific dataset names to process (default: all)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually running'
    )
    
    args = parser.parse_args()
    
    # Set multiprocessing start method for better compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    if args.dry_run:
        # Just show what would be processed
        config_dir = "cellsimbench/configs/dataset"
        configs = load_dataset_configs(config_dir)
        
        if args.datasets:
            configs = {k: v for k, v in configs.items() if k in args.datasets}
        
        print("Datasets that would be processed:")
        for dataset_name in sorted(configs.keys()):
            script_path = Path(f"data/{dataset_name}/get_data.py")
            script_exists = "✅" if script_path.exists() else "❌"
            already_processed = check_dataset_processed(configs[dataset_name])
            
            status = "SKIP" if already_processed and not args.force else "RUN"
            print(f"  {script_exists} {dataset_name:20s} -> {status}")
        
        return
    
    # Run the processing
    success = run_all_datasets(
        force_recompute=args.force,
        num_workers=args.workers,
        specific_datasets=args.datasets
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

