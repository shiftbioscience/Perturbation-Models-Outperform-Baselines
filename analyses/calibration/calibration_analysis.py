

# %%
# ============================================================================
# IMPORTS AND SETUP
# ============================================================================
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import pickle
import json
from scperturb import edist
import warnings
warnings.filterwarnings('ignore')

# Import our existing components
from cellsimbench.core.data_manager import DataManager
from cellsimbench.core.baseline_runner import BaselineRunner
from cellsimbench.core.metrics_engine import MetricsEngine
from cellsimbench.core.benchmark import BenchmarkRunner
from omegaconf import OmegaConf
import argparse
from joblib import Parallel, delayed

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run calibration analysis')
parser.add_argument('--force', action='store_true', help='Force recalculation of all results')
parser.add_argument('--njobs', type=int, default=1, help='Number of parallel jobs for dataset processing (default: 1)')
args = parser.parse_args()

# Set the working directory to the root of the project
import os
from pathlib import Path
os.chdir(Path(__file__).parent.parent.parent)

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Define paths
DATA_DIR = Path("data")
RESULTS_DIR = Path("analyses/calibration/results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Define datasets to analyze
DATASETS = [
    'norman19', 'nadig25hepg2', 'nadig25jurkat',
    'replogle22rpe1', 'replogle22k562', 'replogle22k562gwps', 'adamson16', 'frangieh21', 'tian21crispri', 'tian21crispra', 
    'kaden25rpe1', 'kaden25fibroblast', 'sunshine23', 'wessels23'
]


# Define metrics configuration - INCLUDING CENTROID ACCURACY
METRICS_CONFIG = {
    'mse': {'higher_better': False, 'perfect': 0.0},
    'wmse': {'higher_better': False, 'perfect': 0.0},
    'pearson_deltactrl': {'higher_better': True, 'perfect': 1.0},
    'pearson_deltactrl_degs': {'higher_better': True, 'perfect': 1.0},
    'pearson_deltapert': {'higher_better': True, 'perfect': 1.0},
    'pearson_deltapert_degs': {'higher_better': True, 'perfect': 1.0},
    'r2_deltactrl': {'higher_better': True, 'perfect': 1.0},
    'r2_deltactrl_degs': {'higher_better': True, 'perfect': 1.0},
    'r2_deltapert': {'higher_better': True, 'perfect': 1.0},
    'r2_deltapert_degs': {'higher_better': True, 'perfect': 1.0},
    'weighted_r2_deltactrl': {'higher_better': True, 'perfect': 1.0},
    'weighted_r2_deltapert': {'higher_better': True, 'perfect': 1.0},
    'centroid_accuracy': {'higher_better': True, 'perfect': 1.0},
}

print("Calibration analysis initialized")
print(f"Will analyze {len(DATASETS)} datasets with {len(METRICS_CONFIG)} metrics")
print(f"Results will be saved to: {RESULTS_DIR}")
print(f"Force recalculation: {args.force}")
print(f"Parallel jobs: {args.njobs}")
print("Note: E-distance matrices will be cached for faster reruns. Delete cache files to recompute.")
print("Note: For datasets with >400 perturbations, we subsample to 400 for e-distance calculation (seed=42)")

# %%
# ============================================================================
# CHECK FOR EXISTING RESULTS AND LOAD IF AVAILABLE
# ============================================================================

# Check if final results already exist (unless --force is used)
final_results_exist = (
    (RESULTS_DIR / 'per_perturbation_results.csv').exists() and
    (RESULTS_DIR / 'drf_all_versions.csv').exists() and
    (RESULTS_DIR / 'dataset_quality.csv').exists()
)

if final_results_exist and not args.force and False:
    print("\nFinal results already exist. Loading from disk...")
    per_pert_df = pd.read_csv(RESULTS_DIR / 'per_perturbation_results.csv')
    # Load the FULL results with all three DRF types
    results_df = pd.read_csv(RESULTS_DIR / 'drf_all_versions.csv')
    quality_df = pd.read_csv(RESULTS_DIR / 'dataset_quality.csv')
    
    print(f"  - Loaded per-perturbation results: {len(per_pert_df)} rows")
    print(f"  - Loaded aggregated results: {len(results_df)} rows")
    print(f"  - Loaded quality measures: {len(quality_df)} rows")
    
    # Reconstruct the data structures needed for plotting from CSVs
    print("\nReconstructing data structures from loaded CSVs...")
    
    # Reconstruct per_pert_drf dictionaries for all three DRF versions
    per_pert_drf = {}
    per_pert_drf_mean = {}
    per_pert_drf_sparsemean = {}
    per_pert_drf_ctrl = {}
    
    for _, row in per_pert_df.iterrows():
        dataset = row['dataset']
        metric = row['metric']
        pert = row['perturbation']
        
        # Initialize nested dicts if needed
        for drf_dict in [per_pert_drf, per_pert_drf_mean, per_pert_drf_sparsemean, per_pert_drf_ctrl]:
            if dataset not in drf_dict:
                drf_dict[dataset] = {}
            if metric not in drf_dict[dataset]:
                drf_dict[dataset][metric] = {}
        
        # Reconstruct the key format
        pert_key = f"{dataset}_{pert}"
        
        # Get DRF values - check which columns exist
        if 'drf_mean' in row:
            per_pert_drf_mean[dataset][metric][pert_key] = row['drf_mean']
            per_pert_drf[dataset][metric][pert_key] = row['drf_mean']  # Legacy
        elif 'drf' in row:
            per_pert_drf[dataset][metric][pert_key] = row['drf']
            per_pert_drf_mean[dataset][metric][pert_key] = row['drf']
        
        if 'drf_sparsemean' in row and not pd.isna(row['drf_sparsemean']):
            per_pert_drf_sparsemean[dataset][metric][pert_key] = row['drf_sparsemean']
        
        if 'drf_ctrl' in row and not pd.isna(row['drf_ctrl']):
            per_pert_drf_ctrl[dataset][metric][pert_key] = row['drf_ctrl']
    
    # Reconstruct dataset_quality dictionary
    dataset_quality = {}
    for dataset in DATASETS:
        dataset_quality[dataset] = {
            'deg_counts': {},
            'per_pert_edist': {},
            'quality_measures': {}
        }
        
        # Extract quality measures from quality_df
        if dataset in quality_df['dataset'].values:
            quality_row = quality_df[quality_df['dataset'] == dataset].iloc[0]
            dataset_quality[dataset]['quality_measures'] = {
                'avg_degs': quality_row['avg_degs'],
                'cv_degs': quality_row['cv_degs'],
                'mean_edist': quality_row['mean_edist'],
                'median_edist': quality_row['median_edist'],
                'std_edist': quality_row['std_edist'],
                'n_cells': quality_row['n_cells'],
                'n_perturbations': quality_row['n_perturbations'],
                'cells_per_pert': quality_row['cells_per_pert']
            }
        
        # Extract per-perturbation data from per_pert_df
        dataset_data = per_pert_df[per_pert_df['dataset'] == dataset]
        if not dataset_data.empty:
            # Get unique perturbations for this dataset
            perturbations = dataset_data['perturbation'].unique()
            for pert in perturbations:
                pert_data = dataset_data[dataset_data['perturbation'] == pert].iloc[0]
                if not pd.isna(pert_data['deg_count']):
                    dataset_quality[dataset]['deg_counts'][pert] = pert_data['deg_count']
                if not pd.isna(pert_data['avg_edist']):
                    dataset_quality[dataset]['per_pert_edist'][pert] = pert_data['avg_edist']
    
    # Also need to reconstruct dataset_configs and dataset_metrics for some operations
    # These are lightweight - just create empty placeholders since they're not critical for plots
    dataset_configs = {}
    dataset_metrics = {}
    
    print("  - Reconstructed per_pert_drf dictionary")
    print("  - Reconstructed dataset_quality dictionary")
    
    # Merge quality_df with results_df when loading from existing files
    if 'drf_type' in results_df.columns:
        results_df = results_df.merge(quality_df, on='dataset', how='left')
        print("  - Merged quality measures with results_df containing ALL 3 DRF types")
    
    print("\nSkipping to visualization section...")
    
    # Jump to visualization section by setting a flag
    SKIP_COMPUTATION = True
else:
    print("\nComputing results from scratch...")
    SKIP_COMPUTATION = False

if not SKIP_COMPUTATION:
    # %%
    # ============================================================================
    # LOAD PRE-CALCULATED METRICS FROM CSV FILES
    # ============================================================================
    
    print("\nLoading pre-calculated metrics from baseline output CSVs...")
    dataset_metrics = {}
    dataset_configs = {}
    dataset_quality = {}
    
    # Base directory for baseline outputs
    baseline_outputs_dir = Path("analyses/calibration/baseline_outputs")
    
    missing_datasets = []
    for dataset_name in tqdm(DATASETS, desc="Loading pre-calculated metrics"):
        dataset_dir = baseline_outputs_dir / dataset_name
        metrics_csv = dataset_dir / "detailed_metrics.csv"
        
        if not metrics_csv.exists():
            print(f"  WARNING: No metrics CSV found for {dataset_name} at {metrics_csv}")
            missing_datasets.append(dataset_name)
            continue
        
        # Load the detailed metrics CSV
        print(f"  Loading metrics for {dataset_name}...")
        metrics_df = pd.read_csv(metrics_csv)
        
        # Restructure the data to match expected format
        # Expected format: dataset_metrics[dataset_name][baseline_name][metric_name][perturbation] = value
        dataset_metrics[dataset_name] = {}

        
        # Group by model (baseline type)
        for baseline_name in metrics_df['model'].unique():
            # Map CSV baseline names to expected names
            if baseline_name == 'dataset_mean':
                mapped_name = 'mean'
            elif baseline_name == 'sparse_mean':
                mapped_name = 'sparse_mean'
            elif baseline_name == 'control_mean':
                mapped_name = 'control'
            elif baseline_name == 'technical_duplicate':
                mapped_name = 'tech_dup'
            elif baseline_name == 'interpolated_duplicate':
                mapped_name = 'interp_dup'
            else:
                continue  # Skip other baselines we don't need (like additive)
            
            dataset_metrics[dataset_name][mapped_name] = {}
            baseline_data = metrics_df[metrics_df['model'] == baseline_name]
            
            # Group by metric
            for metric_name in baseline_data['metric'].unique():
                metric_data = baseline_data[baseline_data['metric'] == metric_name]
                
                # Create dictionary of perturbation -> value
                pert_dict = {}
                for _, row in metric_data.iterrows():
                    # Extract perturbation name (remove dataset prefix if present)
                    pert = row['perturbation']
                    if pert.startswith(dataset_name + '_'):
                        pert = pert[len(dataset_name) + 1:]
                    
                    # Create the expected key format: 'covariate_perturbation'
                    # Use dataset name as the covariate
                    pert_key = f"{dataset_name}_{pert}"
                    pert_dict[pert_key] = row['value']
                
                dataset_metrics[dataset_name][mapped_name][metric_name] = pert_dict
        
        # Load dataset config for later use
        config_path = Path(f"cellsimbench/configs/dataset/{dataset_name}.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                import yaml
                dataset_configs[dataset_name] = yaml.safe_load(f)
        
        # Filter metrics with too few perturbations 
        # This is important for metrics like pearson_deltapert_degs where many perturbations may have no DEGs
        # We specifically check the interpolated_duplicate baseline since that's what we use for DRF calculation
        MIN_PERTURBATIONS = 10
        metrics_to_filter = []
        
        # Check each metric for interpolated duplicate baseline only
        for metric_name in METRICS_CONFIG.keys():
            # Only check interpolated duplicate baseline
            if 'interp_dup' in dataset_metrics[dataset_name]:
                if metric_name in dataset_metrics[dataset_name]['interp_dup']:
                    valid_values = [v for v in dataset_metrics[dataset_name]['interp_dup'][metric_name].values() 
                                   if not np.isnan(v)]
                    valid_count = len(valid_values)
                    
                    # If interpolated duplicate has too few valid values, filter this metric
                    if valid_count < MIN_PERTURBATIONS:
                        print(f"    WARNING: {metric_name} has only {valid_count} valid perturbation(s) in interpolated_duplicate - filtering out (threshold: {MIN_PERTURBATIONS})")
                        metrics_to_filter.append(metric_name)
                        
                        # Set all values to NaN for this metric across all baselines
                        for baseline_name in ['mean', 'sparse_mean', 'control', 'tech_dup', 'interp_dup']:
                            if baseline_name in dataset_metrics[dataset_name]:
                                if metric_name in dataset_metrics[dataset_name][baseline_name]:
                                    for pert_key in dataset_metrics[dataset_name][baseline_name][metric_name].keys():
                                        dataset_metrics[dataset_name][baseline_name][metric_name][pert_key] = np.nan
        
        if metrics_to_filter:
            print(f"    Filtered {len(metrics_to_filter)} metric(s) due to insufficient data: {', '.join(metrics_to_filter)}")
        
        print(f"    Loaded {len(metrics_df)} metric values for {dataset_name}")
    
    if missing_datasets:
        print(f"\nWARNING: Missing pre-calculated metrics for {len(missing_datasets)} datasets: {missing_datasets}")
        print("Please run 'analyses/calibration/run_all_dataset_baselines.sh' to generate metrics for these datasets.")

    # %%
    # ============================================================================
    # EXTRACT QUALITY MEASURES FROM DATASETS (NOT IN CSVs)
    # ============================================================================
    print("\n" + "="*60)
    print("EXTRACTING QUALITY MEASURES FROM DATASETS")
    print("="*60)
    
    # Check for cached quality measures
    quality_cache = RESULTS_DIR / 'dataset_quality_cache.pkl'
    
    if quality_cache.exists() and not args.force:
        print("\nLoading cached quality measures...")
        with open(quality_cache, 'rb') as f:
            dataset_quality = pickle.load(f)
        print(f"  Loaded quality measures for {len(dataset_quality)} datasets")
        missing_quality = [d for d in DATASETS if d not in dataset_quality]
    else:
        print("\nNo cached quality measures found or force recalculation requested.")
        dataset_quality = {}
        missing_quality = [d for d in DATASETS if d not in missing_datasets]  # Only process datasets with metrics
    
    # Define function to extract quality measures from a single dataset
    def extract_quality_measures(dataset_name):
        """Extract quality measures from a dataset (DEGs, e-distance, etc.)"""
        print(f"\nExtracting quality measures for {dataset_name}...")
        
        # Load dataset config if not already loaded
        if dataset_name not in dataset_configs:
            config_path = Path(f"cellsimbench/configs/dataset/{dataset_name}.yaml")
            with open(config_path, 'r') as f:
                import yaml
                dataset_config = yaml.safe_load(f)
        else:
            dataset_config = dataset_configs[dataset_name]
        
        # Initialize DataManager to load dataset
        data_manager = DataManager(dataset_config)
        adata = data_manager.load_dataset()
        
        # 1. DEG statistics
        deg_counts = {}
        if 'deg_gene_dict_gt' in adata.uns:
            for pert, degs in adata.uns['deg_gene_dict_gt'].items():
                pert_clean = pert.split('_', 1)[1] if '_' in pert else pert
                if 'control' not in pert_clean.lower():
                    deg_counts[pert_clean] = len(degs)
        
        deg_values = list(deg_counts.values()) if deg_counts else [0]
        avg_degs = np.mean(deg_values)
        cv_degs = np.std(deg_values) / avg_degs if avg_degs > 0 else 0
        
        # 2. E-distance calculation using scperturb
        # Check for cached e-distance matrix
        edist_cache_path = RESULTS_DIR / f'{dataset_name}_edist_matrix.pkl'
        
        if edist_cache_path.exists():
            print(f"    Loading cached e-distance matrix from {edist_cache_path.name}")
            with open(edist_cache_path, 'rb') as f:
                estats = pickle.load(f)
        else:
            # Subsample if too many perturbations (for computational efficiency)
            unique_conditions = adata.obs['condition'].unique()
            non_control_conditions = [c for c in unique_conditions if 'control' not in c.lower()]
            N_SUBSAMPLE = 300
            if len(non_control_conditions) > N_SUBSAMPLE:
                print(f"    Dataset has {len(non_control_conditions)} perturbations - subsampling to {N_SUBSAMPLE} for e-distance calculation")
                np.random.seed(42)
                sampled_conditions = np.random.choice(non_control_conditions, size=N_SUBSAMPLE, replace=False)
                
                # Also include control conditions
                control_conditions = [c for c in unique_conditions if 'control' in c.lower()]
                all_selected_conditions = list(sampled_conditions) + control_conditions
                
                # Subset adata to selected conditions
                adata_subset = adata[adata.obs['condition'].isin(all_selected_conditions)].copy()
                print(f"      Subsampled to {len(all_selected_conditions)} conditions ({len(sampled_conditions)} perturbations + {len(control_conditions)} controls)")
            else:
                adata_subset = adata
                print(f"    Using all {len(non_control_conditions)} perturbations for e-distance calculation")
            
            print("    Calculating e-distance matrix (this may take a while)...")
            # Calculate e-distance in PCA space
            print("      Computing PCA...")
            sc.pp.pca(adata_subset, n_comps=50)
            
            # Calculate e-distance matrix
            estats = edist(adata_subset, obs_key='condition')
            
            # Cache the result
            print(f"    Saving e-distance matrix to {edist_cache_path.name}")
            with open(edist_cache_path, 'wb') as f:
                pickle.dump(estats, f)
        
        # Get all non-control perturbations
        non_control_perts = [p for p in estats.index if 'control' not in p.lower()]
        
        # Calculate per-perturbation average e-distance
        per_pert_edist = {}
        for pert in non_control_perts:
            # Get e-distances from this pert to all other non-control perts
            edist_to_others = []
            for other_pert in non_control_perts:
                if pert != other_pert:  # Exclude self
                    edist_to_others.append(estats.loc[pert, other_pert])
            if edist_to_others:
                per_pert_edist[pert] = np.mean(edist_to_others)
        
        # Calculate overall mean e-distance (excluding control and self-comparisons)
        edist_values = []
        for i, pert1 in enumerate(non_control_perts):
            for j, pert2 in enumerate(non_control_perts):
                if i < j:  # Upper triangle only, excluding diagonal
                    edist_values.append(estats.loc[pert1, pert2])
        
        mean_edist = np.mean(edist_values) if edist_values else 0
        median_edist = np.median(edist_values) if edist_values else 0
        std_edist = np.std(edist_values) if edist_values else 0
        
        # 3. Basic dataset statistics
        n_cells = adata.n_obs
        n_perturbations = len([c for c in adata.obs['condition'].unique() 
                              if 'control' not in c.lower()])
        cells_per_pert = n_cells / n_perturbations if n_perturbations > 0 else 0
        
        print(f"  Avg DEGs: {avg_degs:.1f}, CV: {cv_degs:.2f}")
        print(f"  Mean e-distance: {mean_edist:.2f} ± {std_edist:.2f}")
        print(f"  {n_perturbations} perturbations, {cells_per_pert:.1f} cells/pert")
        
        # Clear adata from memory
        del adata
        print(f"  Cleared adata from memory for {dataset_name}")
        
        # Return the quality measures
        return {
            'deg_counts': deg_counts,
            'per_pert_edist': per_pert_edist,
            'quality_measures': {
                'avg_degs': avg_degs,
                'cv_degs': cv_degs,
                'mean_edist': mean_edist,
                'median_edist': median_edist,
                'std_edist': std_edist,
                'n_cells': n_cells,
                'n_perturbations': n_perturbations,
                'cells_per_pert': cells_per_pert
            }
        }
    
    # Process missing quality datasets
    if missing_quality:
        if args.njobs == 1:
            print(f"\nExtracting quality measures for {len(missing_quality)} datasets sequentially...")
            for dataset_name in missing_quality:
                dataset_quality[dataset_name] = extract_quality_measures(dataset_name)
        else:
            print(f"\nExtracting quality measures for {len(missing_quality)} datasets in parallel with {args.njobs} jobs...")
            results = Parallel(n_jobs=args.njobs, verbose=1, backend='threading')(
                delayed(extract_quality_measures)(dataset_name) for dataset_name in missing_quality
            )
            for dataset_name, quality_data in zip(missing_quality, results):
                dataset_quality[dataset_name] = quality_data
        
        # Cache the quality measures
        print(f"\nCaching quality measures for {len(dataset_quality)} datasets...")
        with open(quality_cache, 'wb') as f:
            pickle.dump(dataset_quality, f)
        print(f"  Cached to {quality_cache}")
    else:
        print("\nAll quality measures are cached or no datasets to process")



# %%
# ============================================================================
# CALCULATE DRF (Dynamic Range Fraction) FOR EACH PERTURBATION
# ============================================================================

if not SKIP_COMPUTATION:
    drf_results = []
    per_pert_drf = {}  # Store per-perturbation DRFs for analysis
    # Store separate DRF versions
    per_pert_drf_mean = {}
    per_pert_drf_sparsemean = {}
    per_pert_drf_ctrl = {}
    per_pert_drf_interpolated = {}
    
    # Define metrics that should use control baseline instead of mean baseline for drf_mean and drf_interpolated
    # These metrics measure correlation/R2 with delta from perturbation mean, so the control baseline
    # is the appropriate negative control (not the dataset mean baseline)
    CONTROL_BASELINE_METRICS = [
        'pearson_deltapert', 
        'pearson_deltapert_degs',
        'r2_deltapert',
        'r2_deltapert_degs',
        'weighted_r2_deltapert'
    ]

    for dataset_name, baseline_metrics in dataset_metrics.items():
        print(f"\n{dataset_name} DRF Calculation (4 versions):")
        print("-" * 40)
        
        # Get all baseline metrics
        mean_metrics = baseline_metrics['mean']
        sparse_mean_metrics = baseline_metrics['sparse_mean']
        control_metrics = baseline_metrics['control']
        techdup_metrics = baseline_metrics['tech_dup']
        interp_dup_metrics = baseline_metrics['interp_dup']
        
        # Store per-perturbation DRFs for this dataset
        per_pert_drf_mean[dataset_name] = {}
        per_pert_drf_sparsemean[dataset_name] = {}
        per_pert_drf_ctrl[dataset_name] = {}
        per_pert_drf_interpolated[dataset_name] = {}

        
        for metric_name, config in METRICS_CONFIG.items():
            # Calculate DRF for EACH perturbation individually
            pert_drfs_mean = {}
            pert_drfs_sparsemean = {}
            pert_drfs_ctrl = {}
            pert_drfs_interpolated = {}
            
            # Get all perturbation keys from tech_dup metrics
            if metric_name not in techdup_metrics:
                continue

            
            for pert_key in techdup_metrics[metric_name].keys():
                techdup_perf = techdup_metrics[metric_name][pert_key]
                
                # Skip if techdup is NaN
                if np.isnan(techdup_perf):
                    continue
                
                # Helper function to calculate DRF given a baseline
                def calculate_drf(baseline_perf, techdup_perf, config):
                    if np.isnan(baseline_perf):
                        return np.nan
                    
                    if config['higher_better']:
                        if baseline_perf > config['perfect']:
                            raise ValueError(f"Baseline performance is perfect for {metric_name} {pert_key}")
                        else:
                            drf = (techdup_perf - baseline_perf) / ((config['perfect'] - baseline_perf) + 1e-6)
                    else:
                        if baseline_perf < config['perfect']:
                            raise ValueError(f"Baseline performance is perfect for {metric_name} {pert_key}")
                        else:
                            drf = (baseline_perf - techdup_perf) / (baseline_perf + 1e-6)
                    
                    # Clip DRF to [-1, 1] range
                    # DRF > 1 means better than tech_dup (cap at 1)
                    # DRF < -1 means severely worse than baseline (cap at -1)
                    drf = np.clip(drf, -1.0, 1.0)
                    return drf
                
                
                try:

                    # Calculate DRF with respect to dataset mean (or control for special metrics)
                    # For pearson_deltapert metrics, use control baseline as negative control
                    if metric_name in CONTROL_BASELINE_METRICS:
                        baseline_for_mean = control_metrics
                    else:
                        baseline_for_mean = mean_metrics
                    
                    if metric_name in baseline_for_mean and pert_key in baseline_for_mean[metric_name]:
                        drf_mean = calculate_drf(baseline_for_mean[metric_name][pert_key], techdup_perf, config)
                        if not np.isnan(drf_mean):
                            pert_drfs_mean[pert_key] = drf_mean
                    
                    # Calculate DRF with respect to sparse mean
                    if metric_name in sparse_mean_metrics and pert_key in sparse_mean_metrics[metric_name]:
                        drf_sparsemean = calculate_drf(sparse_mean_metrics[metric_name][pert_key], techdup_perf, config)
                        if not np.isnan(drf_sparsemean):
                            pert_drfs_sparsemean[pert_key] = drf_sparsemean
                    
                    # Calculate DRF with respect to control
                    if metric_name in control_metrics and pert_key in control_metrics[metric_name]:
                        drf_ctrl = calculate_drf(control_metrics[metric_name][pert_key], techdup_perf, config)
                        if not np.isnan(drf_ctrl):
                            pert_drfs_ctrl[pert_key] = drf_ctrl
                    
                    # Calculate DRF using interpolated duplicate instead of tech_dup
                    # For pearson_deltapert metrics, use control baseline as negative control
                    if metric_name in interp_dup_metrics and pert_key in interp_dup_metrics[metric_name]:
                        interp_dup_perf = interp_dup_metrics[metric_name][pert_key]
                        
                        # Use control baseline for special metrics, mean baseline for others
                        if metric_name in CONTROL_BASELINE_METRICS:
                            baseline_for_interpolated = control_metrics
                        else:
                            baseline_for_interpolated = mean_metrics
                        
                        if metric_name in baseline_for_interpolated and pert_key in baseline_for_interpolated[metric_name] and not np.isnan(interp_dup_perf):
                            # Use interpolated duplicate as the "best" instead of tech_dup
                            drf_interpolated = calculate_drf(baseline_for_interpolated[metric_name][pert_key], interp_dup_perf, config)
                            if not np.isnan(drf_interpolated):
                                pert_drfs_interpolated[pert_key] = drf_interpolated
                except ValueError as e:
                    print(e)
                    breakpoint()


            
            # Store per-perturbation DRFs for all four versions
            per_pert_drf_mean[dataset_name][metric_name] = pert_drfs_mean
            per_pert_drf_sparsemean[dataset_name][metric_name] = pert_drfs_sparsemean
            per_pert_drf_ctrl[dataset_name][metric_name] = pert_drfs_ctrl
            per_pert_drf_interpolated[dataset_name][metric_name] = pert_drfs_interpolated
            

            # Calculate overall statistics for reporting (for all four versions)
            versions = [
                ('drf_mean', pert_drfs_mean, mean_metrics),
                ('drf_sparsemean', pert_drfs_sparsemean, sparse_mean_metrics),
                ('drf_ctrl', pert_drfs_ctrl, control_metrics),
                ('drf_interpolated', pert_drfs_interpolated, interp_dup_metrics)
            ]
            
            for drf_type, pert_drfs, baseline_metrics_ref in versions:
                if pert_drfs:
                    drf_values = list(pert_drfs.values())
                    # Aggregate across perturbations for this dataset-metric-drf_type combination
                    median_drf = np.median(drf_values)
                    mean_drf = np.mean(drf_values)
                    std_drf = np.std(drf_values)
                    q25_drf = np.percentile(drf_values, 25)
                    q75_drf = np.percentile(drf_values, 75)
                    
                    # Calculate mean of the baseline performances for reporting
                    # For drf_mean and drf_interpolated with special metrics, the baseline is actually control_metrics
                    if drf_type in ['drf_mean', 'drf_interpolated'] and metric_name in CONTROL_BASELINE_METRICS:
                        actual_baseline_ref = control_metrics
                    else:
                        actual_baseline_ref = baseline_metrics_ref
                    
                    if metric_name in actual_baseline_ref:
                        baseline_perf = np.mean([actual_baseline_ref[metric_name][k] 
                                                for k in pert_drfs.keys() 
                                                if k in actual_baseline_ref[metric_name]])
                    else:
                        baseline_perf = np.nan
                    
                    techdup_perf_mean = np.mean([techdup_metrics[metric_name][k] 
                                                 for k in pert_drfs.keys()])
                    
                    if drf_type == 'drf_mean':  # Only print for main version to avoid clutter
                        print(f"  {metric_name:25s}: Median DRF={median_drf:+6.3f} (IQR:[{q25_drf:+.3f},{q75_drf:+.3f}]), {len(pert_drfs)} perts")
                    
                    # Store aggregated results for this dataset-metric-drf_type
                    drf_results.append({
                        'dataset': dataset_name,
                        'metric': metric_name,
                        'drf_type': drf_type,
                        'baseline': baseline_perf,
                        'tech_duplicate': techdup_perf_mean,
                        'median_drf': median_drf,
                        'mean_drf': mean_drf,
                        'q25_drf': q25_drf,
                        'q75_drf': q75_drf,
                        'std_drf': std_drf,
                        'n_perturbations': len(pert_drfs)
                    })

    
    drf_df = pd.DataFrame(drf_results)
    print("\nDRF calculation complete (4 versions: drf_mean, drf_sparsemean, drf_ctrl, drf_interpolated)!")
else:
    print("Skipping DRF calculation - already reconstructed from existing results")
    # drf_df will be loaded as results_df from CSV, no need to recalculate
    # Initialize empty DRF dictionaries if not already set
    if 'per_pert_drf_mean' not in locals():
        per_pert_drf_mean = per_pert_drf
    if 'per_pert_drf_sparsemean' not in locals():
        per_pert_drf_sparsemean = {}
    if 'per_pert_drf_ctrl' not in locals():
        per_pert_drf_ctrl = {}
    if 'per_pert_drf_interpolated' not in locals():
        per_pert_drf_interpolated = {}

# %%
# ============================================================================
# PREPARE QUALITY MEASURES DATAFRAME FROM CACHED DATA
# ============================================================================

if not SKIP_COMPUTATION:
    # Extract quality measures from the cached dataset_quality dictionary
    quality_measures = []
    for dataset_name in DATASETS:
        if dataset_name in dataset_quality and 'quality_measures' in dataset_quality[dataset_name]:
            quality_data = dataset_quality[dataset_name]['quality_measures'].copy()
            quality_data['dataset'] = dataset_name
            quality_measures.append(quality_data)

    quality_df = pd.DataFrame(quality_measures)
    print(f"\nQuality measures prepared for {len(quality_measures)} datasets")

# %%
# ============================================================================
# MERGE ALL RESULTS AND CALCULATE CORRELATIONS
# ============================================================================

if not SKIP_COMPUTATION:
    # First create per-perturbation results dataframe
    per_pert_results = []

    for dataset_name in DATASETS:
        if dataset_name not in per_pert_drf_mean:
            continue
        
        # Get per-perturbation quality metrics (already calculated)
        per_pert_edist = dataset_quality.get(dataset_name, {}).get('per_pert_edist', {})
        deg_counts = dataset_quality.get(dataset_name, {}).get('deg_counts', {})
        
        # Iterate through each metric and perturbation
        for metric_name in METRICS_CONFIG.keys():
            if metric_name not in per_pert_drf_mean[dataset_name]:
                continue
            
            for pert_key, drf in per_pert_drf_mean[dataset_name][metric_name].items():
                # Get condition name
                condition = pert_key.split('_', 1)[1] if '_' in pert_key else pert_key
                
                # Build result row for all three DRF versions
                base_row = {
                    'dataset': dataset_name,
                    'metric': metric_name,
                    'perturbation': condition,
                    'deg_count': deg_counts.get(condition, np.nan),
                    'avg_edist': per_pert_edist.get(condition, np.nan)
                }
                
                # Add drf_mean
                row = base_row.copy()
                row['drf'] = drf  # Legacy compatibility
                row['drf_mean'] = drf
                
                # Add drf_sparsemean if available
                if (dataset_name in per_pert_drf_sparsemean and 
                    metric_name in per_pert_drf_sparsemean[dataset_name] and
                    pert_key in per_pert_drf_sparsemean[dataset_name][metric_name]):
                    row['drf_sparsemean'] = per_pert_drf_sparsemean[dataset_name][metric_name][pert_key]
                else:
                    row['drf_sparsemean'] = np.nan
                
                # Add drf_ctrl if available
                if (dataset_name in per_pert_drf_ctrl and 
                    metric_name in per_pert_drf_ctrl[dataset_name] and
                    pert_key in per_pert_drf_ctrl[dataset_name][metric_name]):
                    row['drf_ctrl'] = per_pert_drf_ctrl[dataset_name][metric_name][pert_key]
                else:
                    row['drf_ctrl'] = np.nan
                
                # Add drf_interpolated if available
                if (dataset_name in per_pert_drf_interpolated and 
                    metric_name in per_pert_drf_interpolated[dataset_name] and
                    pert_key in per_pert_drf_interpolated[dataset_name][metric_name]):
                    row['drf_interpolated'] = per_pert_drf_interpolated[dataset_name][metric_name][pert_key]
                else:
                    row['drf_interpolated'] = np.nan
                    
                per_pert_results.append(row)

    per_pert_df = pd.DataFrame(per_pert_results)

    # IMPORTANT: Create separate results DataFrames for each DRF type
    # - We DO aggregate across perturbations (median/mean) to get one value per dataset-metric
    # - We NEVER mix the four DRF types together - they are always kept completely separate
    # - Each DRF type represents comparison to a different baseline or approach:
    #   * drf_mean: tech_dup vs dataset_mean
    #   * drf_sparsemean: tech_dup vs sparse_mean  
    #   * drf_ctrl: tech_dup vs control_mean
    #   * drf_interpolated: interpolated_dup vs dataset_mean
    results_df_mean = drf_df[drf_df['drf_type'] == 'drf_mean'].copy()
    results_df_sparsemean = drf_df[drf_df['drf_type'] == 'drf_sparsemean'].copy()
    results_df_ctrl = drf_df[drf_df['drf_type'] == 'drf_ctrl'].copy()
    results_df_interpolated = drf_df[drf_df['drf_type'] == 'drf_interpolated'].copy()
    
    # Use the full drf_df with ALL four types
    results_df = drf_df.copy()
    
    # Add quality measures to ALL results (results_df contains all four DRF types)
    results_df = results_df.merge(quality_df, on='dataset', how='left')
    # Also keep the separate versions for individual saving
    results_df_mean = results_df_mean.merge(quality_df, on='dataset', how='left')
    results_df_sparsemean = results_df_sparsemean.merge(quality_df, on='dataset', how='left')
    results_df_ctrl = results_df_ctrl.merge(quality_df, on='dataset', how='left')
    results_df_interpolated = results_df_interpolated.merge(quality_df, on='dataset', how='left')

    # Save all results
    per_pert_df.to_csv(RESULTS_DIR / 'per_perturbation_results.csv', index=False)
    
    # Save separate files for each DRF type (aggregated by dataset-metric)
    results_df_mean.to_csv(RESULTS_DIR / 'calibration_results_drf_mean.csv', index=False)
    results_df_sparsemean.to_csv(RESULTS_DIR / 'calibration_results_drf_sparsemean.csv', index=False)
    results_df_ctrl.to_csv(RESULTS_DIR / 'calibration_results_drf_ctrl.csv', index=False)
    results_df_interpolated.to_csv(RESULTS_DIR / 'calibration_results_drf_interpolated.csv', index=False)
    
    # Save all DRF versions in one file with drf_type column
    drf_df.to_csv(RESULTS_DIR / 'drf_all_versions.csv', index=False)
    
    quality_df.to_csv(RESULTS_DIR / 'dataset_quality.csv', index=False)
    
    print("\nResults saved to:", RESULTS_DIR)
    print(f"  - Per-perturbation results: {len(per_pert_df)} rows (all 4 DRF types as columns)")
    print(f"  - Aggregated results by DRF type:")
    print(f"    - calibration_results_drf_mean.csv: {len(results_df_mean)} rows")
    print(f"    - calibration_results_drf_sparsemean.csv: {len(results_df_sparsemean)} rows")
    print(f"    - calibration_results_drf_ctrl.csv: {len(results_df_ctrl)} rows")
    print(f"    - calibration_results_drf_interpolated.csv: {len(results_df_interpolated)} rows")
    print(f"  - All DRF versions combined: drf_all_versions.csv with {len(drf_df)} rows")
else:
    # When skipping computation, we already loaded these from CSV at the beginning
    print("\nUsing loaded results from CSV files")
    print(f"  - Per-perturbation results: {len(per_pert_df)} rows (all 4 DRF types as columns)")
    print(f"  - Aggregated results: {len(results_df)} rows (ALL 4 DRF types)")
    print(f"  - Quality measures: {len(quality_df)} rows")



# %%
# ============================================================================
# VISUALIZATION: DRF Heatmaps for All Four Types
# ============================================================================

# Always show all four DRF types
if 'drf_type' in results_df.columns:
    drf_types = ['drf_mean', 'drf_sparsemean', 'drf_ctrl', 'drf_interpolated']
    # Create separate heatmaps for each DRF type
    fig, axes = plt.subplots(1, len(drf_types), figsize=(12 * len(drf_types), 8))
    if len(drf_types) == 1:
        axes = [axes]
    
    for idx, drf_type in enumerate(drf_types):
        ax = axes[idx]
        
        # Filter data for this DRF type
        drf_type_data = results_df[results_df['drf_type'] == drf_type]
        
        # Create pivot table
        drf_pivot = drf_type_data.pivot(index='dataset', columns='metric', values='mean_drf')
        
        # Order rows and columns by their mean values
        # Order columns (metrics) by their mean across datasets (descending)
        col_means = drf_pivot.mean(axis=0).sort_values(ascending=False)
        drf_pivot = drf_pivot[col_means.index]
        
        # Order rows (datasets) by their mean across metrics (descending)
        row_means = drf_pivot.mean(axis=1).sort_values(ascending=False)
        drf_pivot = drf_pivot.loc[row_means.index]
        
        # Fixed color scale: -1 to 1, centered at 0
        # Red (negative) - White (0) - Blue (positive)
        vmin, vmax = -1.0, 1.0
        cmap = 'RdBu_r'  # Red for negative, Blue for positive
        
        sns.heatmap(drf_pivot, annot=True, fmt='.2f', cmap=cmap, 
                    vmin=vmin, vmax=vmax, ax=ax, center=0,
                    cbar_kws={'label': f'Mean DRF ({drf_type.replace("drf_", "")})'})
        ax.set_title(f'Mean Dynamic Range Fraction ({drf_type.replace("drf_", "").title()}) by Dataset and Metric', fontsize=14)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Dataset')
    
    plt.suptitle('DRF Types Shown SEPARATELY: Dataset Mean | Sparse Mean | Control | Interpolated (NOT averaged together)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'calibration_heatmaps_four_types_separate.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Heatmaps saved to", RESULTS_DIR / 'calibration_heatmaps_four_types_separate.png')
else:
    # Fallback to single heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    drf_pivot = results_df.pivot(index='dataset', columns='metric', values='mean_drf')
    
    # Order rows and columns by their mean values
    # Order columns (metrics) by their mean across datasets (descending)
    col_means = drf_pivot.mean(axis=0).sort_values(ascending=False)
    drf_pivot = drf_pivot[col_means.index]
    
    # Order rows (datasets) by their mean across metrics (descending)
    row_means = drf_pivot.mean(axis=1).sort_values(ascending=False)
    drf_pivot = drf_pivot.loc[row_means.index]
    
    # Fixed color scale: -1 to 1, centered at 0
    # Red (negative) - White (0) - Blue (positive)
    vmin, vmax = -1.0, 1.0
    cmap = 'RdBu_r'  # Red for negative, Blue for positive
        
    sns.heatmap(drf_pivot, annot=True, fmt='.2f', cmap=cmap, 
                vmin=vmin, vmax=vmax, ax=ax, center=0,
                cbar_kws={'label': 'Mean DRF'})
    ax.set_title('Mean Dynamic Range Fraction (DRF) by Dataset and Metric', fontsize=16)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Dataset')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'calibration_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Heatmap saved to", RESULTS_DIR / 'calibration_heatmap.png')

# %%
# ============================================================================
# VISUALIZATION: Dataset-level Scatter Plots - Mean E-dist vs Median DRF (PER METRIC)
# ============================================================================

print("\nGenerating dataset-level scatter plots (Mean E-distance vs Median DRF) for each metric...")

# Always plot all four DRF types
drf_types_to_plot = ['drf_mean', 'drf_sparsemean', 'drf_ctrl', 'drf_interpolated']

# Create a separate plot for each metric
for metric_name in METRICS_CONFIG.keys():
    print(f"\nCreating scatter plot for {metric_name}...")
    
    # Create subplots for each DRF type
    n_drf_types = len(drf_types_to_plot)
    fig, axes = plt.subplots(1, n_drf_types, figsize=(10 * n_drf_types, 8))
    if n_drf_types == 1:
        axes = [axes]
    
    for drf_idx, drf_type in enumerate(drf_types_to_plot):
        ax = axes[drf_idx]
        
        # Collect data for this metric and DRF type
        metric_scatter_data = []
        
        for dataset_name in DATASETS:
            # Get mean e-distance from quality_df
            dataset_quality_row = quality_df[quality_df['dataset'] == dataset_name]
            if dataset_quality_row.empty:
                continue
            
            mean_edist = dataset_quality_row['mean_edist'].iloc[0]
            median_edist = dataset_quality_row['median_edist'].iloc[0]
            
            # Get mean DRF for THIS SPECIFIC METRIC and DRF type for this dataset
            # results_df MUST have drf_type column with all three types
            metric_dataset_results = results_df[(results_df['dataset'] == dataset_name) & 
                                               (results_df['metric'] == metric_name) &
                                               (results_df['drf_type'] == drf_type)]
            
            if metric_dataset_results.empty:
                continue
            
            mean_drf_metric = metric_dataset_results['mean_drf'].iloc[0]
            
            # Calculate SEM for DRF (for this specific metric and DRF type)
            metric_drfs = []
            # Select the correct per_pert_drf dictionary based on DRF type
            if drf_type == 'drf_mean' and dataset_name in per_pert_drf_mean:
                if metric_name in per_pert_drf_mean[dataset_name]:
                    metric_drfs = list(per_pert_drf_mean[dataset_name][metric_name].values())
            elif drf_type == 'drf_sparsemean' and dataset_name in per_pert_drf_sparsemean:
                if metric_name in per_pert_drf_sparsemean[dataset_name]:
                    metric_drfs = list(per_pert_drf_sparsemean[dataset_name][metric_name].values())
            elif drf_type == 'drf_ctrl' and dataset_name in per_pert_drf_ctrl:
                if metric_name in per_pert_drf_ctrl[dataset_name]:
                    metric_drfs = list(per_pert_drf_ctrl[dataset_name][metric_name].values())
            elif drf_type == 'drf_interpolated' and dataset_name in per_pert_drf_interpolated:
                if metric_name in per_pert_drf_interpolated[dataset_name]:
                    metric_drfs = list(per_pert_drf_interpolated[dataset_name][metric_name].values())
            elif dataset_name in per_pert_drf and metric_name in per_pert_drf[dataset_name]:
                # Fallback to legacy per_pert_drf
                metric_drfs = list(per_pert_drf[dataset_name][metric_name].values())
            
            sem_drf = np.std(metric_drfs) / np.sqrt(len(metric_drfs)) if metric_drfs else 0
            
            # Calculate SEM for e-distance
            edist_values = []
            if dataset_name in dataset_quality:
                per_pert_edist = dataset_quality[dataset_name].get('per_pert_edist', {})
                edist_values = list(per_pert_edist.values())
            
            sem_edist = np.std(edist_values) / np.sqrt(len(edist_values)) if edist_values else 0
            
            metric_scatter_data.append({
                'dataset': dataset_name,
                'mean_edist': mean_edist,
                'median_edist': median_edist,
                'sem_edist': sem_edist,
                'mean_drf': mean_drf_metric,
                'sem_drf': sem_drf
            })
        
        # Create the scatter plot for this metric and DRF type
        if metric_scatter_data:
            # Use different colors for each dataset
            colors = plt.cm.Set2(np.linspace(0, 1, len(metric_scatter_data)))
            
            for i, data in enumerate(metric_scatter_data):
                ax.errorbar(
                    data['mean_edist'],
                    data['mean_drf'],
                    xerr=data['sem_edist'],
                    yerr=data['sem_drf'],
                    fmt='o',
                    markersize=12,
                    capsize=6,
                    capthick=2,
                    label=data['dataset'],
                    color=colors[i],
                    alpha=0.8,
                    linewidth=2
                )
                
                # Add dataset label next to point
                ax.annotate(
                    data['dataset'],
                    (data['mean_edist'], data['mean_drf']),
                    xytext=(8, 8),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    alpha=0.9
                )
            
            # Calculate and show correlation
            if len(metric_scatter_data) >= 3:
                x_vals = [d['mean_edist'] for d in metric_scatter_data]
                y_vals = [d['mean_drf'] for d in metric_scatter_data]
                
                # Pearson correlation
                corr_pearson, pval_pearson = stats.pearsonr(x_vals, y_vals)
                # Spearman correlation  
                corr_spearman, pval_spearman = stats.spearmanr(x_vals, y_vals)
                
                # Add trend line
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(x_vals) * 0.9, max(x_vals) * 1.1, 100)
                ax.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2)
                
                # Add correlation text box
                textstr = f'Pearson r = {corr_pearson:.3f} (p = {pval_pearson:.3f})\nSpearman ρ = {corr_spearman:.3f} (p = {pval_spearman:.3f})'
                ax.text(0.05, 0.95, textstr,
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                       fontsize=10)
            
            ax.set_xlabel('Average E-distance in Dataset', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Mean DRF ({drf_type.replace("drf_", "")})', fontsize=12, fontweight='bold')
            ax.set_title(f'{drf_type.replace("drf_", "").title()} Baseline', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Don't set fixed y limits since DRF can be negative now
            y_vals_all = [d['mean_drf'] for d in metric_scatter_data]
            if y_vals_all:
                ymin, ymax = min(y_vals_all), max(y_vals_all)
                y_range = ymax - ymin
                ax.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)
    
    # Add overall title and save
    if n_drf_types > 1:
        fig.suptitle(f'{metric_name}: Dataset Quality vs Calibration (4 DRF Types Shown SEPARATELY)', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'dataset_scatter_{metric_name}_four_drf_types_separate.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  Saved {metric_name} scatter plot to {RESULTS_DIR / f'dataset_scatter_{metric_name}_four_drf_types_separate.png'}")



# %%
# ============================================================================
# VISUALIZATION: Dataset-level Scatter Plots - Average DEGs vs Median DRF (PER METRIC)
# ============================================================================
# Only run if we have computed results or if we're not skipping computation
if not SKIP_COMPUTATION or (SKIP_COMPUTATION and 'dataset_quality' in locals()):
    print("\nGenerating dataset-level scatter plots (Average DEGs vs Median DRF) for each metric...")

    # Initialize per_pert_drf dictionaries if they don't exist
    if 'per_pert_drf_mean' not in locals():
        per_pert_drf_mean = per_pert_drf if 'per_pert_drf' in locals() else {}
    if 'per_pert_drf_sparsemean' not in locals():
        per_pert_drf_sparsemean = {}
    if 'per_pert_drf_ctrl' not in locals():
        per_pert_drf_ctrl = {}
    if 'per_pert_drf_interpolated' not in locals():
        per_pert_drf_interpolated = {}

    # Always plot all four DRF types
    drf_types_to_plot = ['drf_mean', 'drf_sparsemean', 'drf_ctrl', 'drf_interpolated']

    # Create a separate plot for each metric
    for metric_name in METRICS_CONFIG.keys():
        print(f"\nCreating DEG count scatter plot for {metric_name}...")
        
        # Create subplots for each DRF type
        n_drf_types = len(drf_types_to_plot)
        fig, axes = plt.subplots(1, n_drf_types, figsize=(10 * n_drf_types, 8))
        if n_drf_types == 1:
            axes = [axes]
        
        for drf_idx, drf_type in enumerate(drf_types_to_plot):
            ax = axes[drf_idx]
            
            # Collect data for this metric and DRF type
            metric_scatter_data = []
            
            for dataset_name in DATASETS:
                # Get average DEGs from quality_df
                dataset_quality_row = quality_df[quality_df['dataset'] == dataset_name]
                if dataset_quality_row.empty:
                    continue
                
                avg_degs = dataset_quality_row['avg_degs'].iloc[0]
                cv_degs = dataset_quality_row['cv_degs'].iloc[0]
            
                # Get mean DRF for THIS SPECIFIC METRIC and DRF type for this dataset
                # results_df MUST have drf_type column with all three types
                metric_dataset_results = results_df[(results_df['dataset'] == dataset_name) & 
                                                   (results_df['metric'] == metric_name) &
                                                   (results_df['drf_type'] == drf_type)]
                
                if metric_dataset_results.empty:
                    continue
                
                mean_drf_metric = metric_dataset_results['mean_drf'].iloc[0]
            
                # Calculate SEM for DRF (for this specific metric and DRF type)
                metric_drfs = []
                # Select the correct per_pert_drf dictionary based on DRF type
                if drf_type == 'drf_mean' and dataset_name in per_pert_drf_mean:
                    if metric_name in per_pert_drf_mean[dataset_name]:
                        metric_drfs = list(per_pert_drf_mean[dataset_name][metric_name].values())
                elif drf_type == 'drf_sparsemean' and dataset_name in per_pert_drf_sparsemean:
                    if metric_name in per_pert_drf_sparsemean[dataset_name]:
                        metric_drfs = list(per_pert_drf_sparsemean[dataset_name][metric_name].values())
                elif drf_type == 'drf_ctrl' and dataset_name in per_pert_drf_ctrl:
                    if metric_name in per_pert_drf_ctrl[dataset_name]:
                        metric_drfs = list(per_pert_drf_ctrl[dataset_name][metric_name].values())
                elif drf_type == 'drf_interpolated' and dataset_name in per_pert_drf_interpolated:
                    if metric_name in per_pert_drf_interpolated[dataset_name]:
                        metric_drfs = list(per_pert_drf_interpolated[dataset_name][metric_name].values())
                elif dataset_name in per_pert_drf and metric_name in per_pert_drf[dataset_name]:
                    # Fallback to legacy per_pert_drf
                    metric_drfs = list(per_pert_drf[dataset_name][metric_name].values())
                
                sem_drf = np.std(metric_drfs) / np.sqrt(len(metric_drfs)) if metric_drfs else 0
                
                # Calculate SEM for DEG counts
                deg_values = []
                if dataset_name in dataset_quality:
                    deg_counts = dataset_quality[dataset_name].get('deg_counts', {})
                    deg_values = list(deg_counts.values())
                
                sem_degs = np.std(deg_values) / np.sqrt(len(deg_values)) if deg_values else 0
                
                metric_scatter_data.append({
                    'dataset': dataset_name,
                    'avg_degs': avg_degs,
                    'cv_degs': cv_degs,
                    'sem_degs': sem_degs,
                    'mean_drf': mean_drf_metric,
                    'sem_drf': sem_drf
                })
            
            # Create the scatter plot for this metric and DRF type
            if metric_scatter_data:
                # Use different colors for each dataset
                colors = plt.cm.Set2(np.linspace(0, 1, len(metric_scatter_data)))
                
                for i, data in enumerate(metric_scatter_data):
                    ax.errorbar(
                        data['avg_degs'],
                        data['mean_drf'],
                        xerr=data['sem_degs'],
                        yerr=data['sem_drf'],
                        fmt='o',
                        markersize=12,
                        capsize=6,
                        capthick=2,
                        label=data['dataset'],
                        color=colors[i],
                        alpha=0.8,
                        linewidth=2
                    )
                    
                    # Add dataset label next to point
                    ax.annotate(
                        data['dataset'],
                        (data['avg_degs'], data['mean_drf']),
                        xytext=(8, 8),
                        textcoords='offset points',
                        fontsize=9,
                        fontweight='bold',
                        alpha=0.9
                    )
                
                # Calculate and show correlation
                if len(metric_scatter_data) >= 3:
                    x_vals = [d['avg_degs'] for d in metric_scatter_data]
                    y_vals = [d['mean_drf'] for d in metric_scatter_data]
                    
                    # Pearson correlation
                    corr_pearson, pval_pearson = stats.pearsonr(x_vals, y_vals)
                    # Spearman correlation  
                    corr_spearman, pval_spearman = stats.spearmanr(x_vals, y_vals)
                    
                    # Add trend line
                    z = np.polyfit(x_vals, y_vals, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(x_vals) * 0.9, max(x_vals) * 1.1, 100)
                    ax.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2)
                    
                    # Add correlation text box
                    textstr = f'Pearson r = {corr_pearson:.3f} (p = {pval_pearson:.3f})\nSpearman ρ = {corr_spearman:.3f} (p = {pval_spearman:.3f})'
                    ax.text(0.05, 0.95, textstr,
                           transform=ax.transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                           fontsize=10)
                
                ax.set_xlabel('Average DEG Count in Dataset', fontsize=12, fontweight='bold')
                ax.set_ylabel(f'Mean DRF ({drf_type.replace("drf_", "")})', fontsize=12, fontweight='bold')
                ax.set_title(f'{drf_type.replace("drf_", "").title()} Baseline', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Don't set fixed y limits since DRF can be negative now
                y_vals_all = [d['mean_drf'] for d in metric_scatter_data]
                if y_vals_all:
                    ymin, ymax = min(y_vals_all), max(y_vals_all)
                    y_range = ymax - ymin
                    ax.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)
        
        # Add overall title and save
        if n_drf_types > 1:
            fig.suptitle(f'{metric_name}: Dataset Signal Strength vs Calibration (4 DRF Types Shown SEPARATELY)', fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f'dataset_degs_scatter_{metric_name}_four_drf_types_separate.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  Saved {metric_name} DEG scatter plot to {RESULTS_DIR / f'dataset_degs_scatter_{metric_name}_four_drf_types_separate.png'}")

# %%
# ============================================================================
# VISUALIZATION: Per-Perturbation DRF vs DEG Count
# ============================================================================
# Plot how calibration (DRF) relates to perturbation strength (DEG count)

print("\nGenerating DRF vs DEG count plots...")

# Define available DRF columns
drf_cols_available = ['drf_mean', 'drf_sparsemean', 'drf_ctrl', 'drf_interpolated']
# Check if columns exist in the dataframe
drf_cols_available = [col for col in drf_cols_available if col in per_pert_df.columns]
if not drf_cols_available:
    # Fallback if the new columns don't exist
    drf_cols_available = ['drf'] if 'drf' in per_pert_df.columns else []

# Create plots for each dataset
for dataset_name in DATASETS:
    # Filter per_pert_df to this dataset
    dataset_data = per_pert_df[per_pert_df['dataset'] == dataset_name]
    
    if dataset_data.empty:
        print(f"  No data for {dataset_name}, skipping")
        continue
    
    # Create figure with subplots: rows for metrics, columns for DRF types
    n_metrics = len(METRICS_CONFIG)
    n_drf_types = len(drf_cols_available)
    n_metric_cols = 4  # Number of columns for metrics
    n_metric_rows = (n_metrics + n_metric_cols - 1) // n_metric_cols
    
    fig, axes = plt.subplots(n_metric_rows, n_metric_cols * n_drf_types, 
                             figsize=(6 * n_metric_cols * n_drf_types, n_metric_rows * 4))
    if n_metric_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'{dataset_name}: DRF vs DEG Count by Metric and DRF Type', fontsize=16)
    
    for metric_idx, (metric_name, metric_config) in enumerate(METRICS_CONFIG.items()):
        # Filter to this specific metric
        metric_data = dataset_data[dataset_data['metric'] == metric_name]
        
        for drf_idx, drf_col in enumerate(drf_cols_available):
            # Calculate subplot position
            row = metric_idx // n_metric_cols
            col = (metric_idx % n_metric_cols) * n_drf_types + drf_idx
            ax = axes[row, col] if n_metric_rows > 1 else axes[col]
            
            # Remove rows with NaN values for this specific DRF type
            valid_data = metric_data.dropna(subset=[drf_col, 'deg_count'])
            
            if len(valid_data) > 0:
                x_degs = valid_data['deg_count'].values
                y_drf = valid_data[drf_col].values
                
                # Create scatter plot
                scatter = ax.scatter(x_degs, y_drf, alpha=0.6, s=50)
                
                # Add correlation
                if len(x_degs) >= 3:
                    corr, pval = stats.spearmanr(x_degs, y_drf)
                    ax.text(0.95, 0.05, f'ρ = {corr:.3f}\np = {pval:.3f}',
                           transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
                
                # Add trend line
                if len(x_degs) >= 2:
                    z = np.polyfit(x_degs, y_drf, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(x_degs), max(x_degs), 100)
                    ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=1)
                
                ax.set_xlabel('DEG Count', fontsize=10)
                ax.set_ylabel(f'DRF ({drf_col.replace("drf_", "").replace("drf", "mean")})', fontsize=10)
                
                # Title shows metric name
                title = metric_name.replace('_', ' ').title()
                if drf_idx == 0:  # Only show metric name on first DRF type
                    ax.set_title(title, fontsize=11)
                
                ax.grid(True, alpha=0.3)
                
                # Dynamic y-axis limits since DRF can be negative
                ymin, ymax = y_drf.min(), y_drf.max()
                y_range = ymax - ymin
                ax.set_ylim(ymin - 0.1 * max(y_range, 0.1), ymax + 0.1 * max(y_range, 0.1))
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center')
                if drf_idx == 0:
                    ax.set_title(metric_name.replace('_', ' ').title(), fontsize=11)
    
    # Hide unused subplots
    total_subplots = n_metric_rows * n_metric_cols * n_drf_types
    for idx in range(n_metrics * n_drf_types, total_subplots):
        row = idx // (n_metric_cols * n_drf_types)
        col = idx % (n_metric_cols * n_drf_types)
        if n_metric_rows > 1:
            axes[row, col].axis('off')
        else:
            axes[col].axis('off')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{dataset_name}_drf_vs_degs.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved {dataset_name}_drf_vs_degs.png")

print("DRF vs DEG count visualization complete!")

# %%
# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("="*70)
print("CALIBRATION ANALYSIS SUMMARY")
print("="*70)

# Most robust metrics
print("\n1. MOST ROBUST METRICS (by mean DRF):")
print("-" * 40)

# Results_df contains all three DRF types  
# For the summary, we'll analyze them separately
if 'drf_type' not in results_df.columns:
    raise ValueError("drf_type column missing from results_df - this should not happen!")

# Show top metrics for each DRF type
for drf_type in results_df['drf_type'].unique():
    print(f"\n  {drf_type}:")
    drf_subset = results_df[results_df['drf_type'] == drf_type]
    mean_drf_by_metric = drf_subset.groupby('metric')['mean_drf'].mean().sort_values(ascending=False)
    for metric, drf_val in mean_drf_by_metric.head(3).items():
        print(f"    {metric:30s}: {drf_val:+.3f}")

# Continue with dataset analysis

# Best calibrated datasets
print("\n2. BEST CALIBRATED DATASETS (by mean DRF):")
print("-" * 40)

# Show for each DRF type
for drf_type in results_df['drf_type'].unique():
    print(f"\n  {drf_type}:")
    drf_subset = results_df[results_df['drf_type'] == drf_type]
    mean_drf_by_dataset = drf_subset.groupby('dataset')['mean_drf'].mean().sort_values(ascending=False)
    for dataset, drf_val in mean_drf_by_dataset.head(3).items():
        if dataset in quality_df['dataset'].values:
            quality_row = quality_df[quality_df['dataset'] == dataset].iloc[0]
            print(f"    {dataset:15s}: DRF={drf_val:.3f}, DEGs={quality_row['avg_degs']:.0f}, "
                  f"e-dist={quality_row['median_edist']:.1f}")

# Special analysis for centroid accuracy
print("\n3. CENTROID ACCURACY ANALYSIS:")
print("-" * 40)

# Show centroid accuracy for each DRF type
for drf_type in results_df['drf_type'].unique():
    print(f"\n  {drf_type}:")
    drf_subset = results_df[results_df['drf_type'] == drf_type]
    centroid_data = drf_subset[drf_subset['metric'] == 'centroid_accuracy']
    if not centroid_data.empty:
        print(f"    Mean DRF: {centroid_data['mean_drf'].mean():.3f}")
        print(f"    Range across datasets: {centroid_data['mean_drf'].min():.3f} - {centroid_data['mean_drf'].max():.3f}")

print("\n" + "="*70)
print("Analysis complete! Check the results/ directory for saved outputs.")
print("="*70)
