# %%

"""
Complete Norman19 data processing pipeline
Adapted from gene_map_v1 pipeline but simplified for single-dataset processing
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
import subprocess as sp
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
data_cache_dir = './data/norman19'
data_url = 'https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad?download=1'
MAX_CELLS_CONTROL = 8192
N_SYNTHETIC_CONTROLS = 500

print("Starting Norman19 complete processing pipeline...")

# ============================================================================
# STEP 1: Download and Basic Processing
# ============================================================================
print("\n" + "="*50)
print("STEP 1: Download and Basic Processing")
print("="*50)

# %%

if not os.path.exists(data_cache_dir):
    os.makedirs(data_cache_dir)

tmp_data_dir = f'{data_cache_dir}/norman19_downloaded.h5ad'

if not os.path.exists(tmp_data_dir):
    print("Downloading Norman19 data...")
    sp.call(f'wget -q {data_url} -O {tmp_data_dir}', shell=True)

print("Loading Norman19 data...")
adata = sc.read_h5ad(tmp_data_dir)

# Rename columns to standard format
adata.obs.rename(columns={
    'nCount_RNA': 'ncounts',
    'nFeature_RNA': 'ngenes',
    'percent.mt': 'percent_mito',
    'cell_line': 'cell_type',
}, inplace=True)

# %%

# Standardize perturbation names
adata.obs['perturbation'] = adata.obs['perturbation'].str.replace('_', '+')
adata.obs['perturbation'] = adata.obs['perturbation'].astype('category')
adata.obs['condition'] = adata.obs.perturbation.copy()

# Add donor_id column (required by pipeline but constant for norman19)
adata.obs['donor_id'] = 'norman19'

# Convert to sparse matrix
adata.X = csr_matrix(adata.X)

# Filter cells and genes
print("Filtering cells and genes...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Stash raw counts
adata.layers['counts'] = adata.X.copy()

# Library size normalization and log1p
print("Normalizing and log-transforming...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# %%

# Downsample each perturbation to have no more than N cells
print("Downsampling perturbations...")
pert_counts = adata.obs['condition'].value_counts()
# Keep only perturbations with at least 12 cells
perts_to_keep = pert_counts[pert_counts >= 12].index
adata = adata[adata.obs['condition'].isin(perts_to_keep)]
pert_counts = adata.obs['condition'].value_counts()
mean_cells = pert_counts.mean()
print(f"Mean cells per perturbation: {mean_cells:.1f}")

MAX_CELLS = round(mean_cells)

pert_counts_oversized = pert_counts[pert_counts > MAX_CELLS]
cells_to_keep = []

for pert in pert_counts.index:
    pert_cells = adata.obs[adata.obs['condition'] == pert].index.tolist()
    if pert == 'control':
        if len(pert_cells) > MAX_CELLS_CONTROL:
            pert_cells = np.random.choice(pert_cells, size=MAX_CELLS_CONTROL, replace=False)
    else:
        if len(pert_cells) > MAX_CELLS:
            pert_cells = np.random.choice(pert_cells, size=MAX_CELLS, replace=False)
    cells_to_keep.extend(pert_cells)

adata = adata[cells_to_keep]

# %%

# Filter perturbations with at least 4 cells (needed for technical duplicate baselines)
print("Filtering perturbations with at least 4 cells...")
pert_counts = adata.obs['condition'].value_counts()
valid_perts = pert_counts[pert_counts >= 4].index
print(f"Keeping {len(valid_perts)} perturbations with ≥4 cells out of {len(pert_counts)} total")

# Also keep control cells regardless of count
control_conditions = ['control', 'ctrl']
for ctrl in control_conditions:
    if ctrl in adata.obs['condition'].unique() and ctrl not in valid_perts:
        valid_perts = valid_perts.append(pd.Index([ctrl]))

adata = adata[adata.obs['condition'].isin(valid_perts)]
print(f"After perturbation filtering: {adata.shape[0]} cells, {adata.shape[1]} genes")

# %%

# Get highly variable genes
print("Finding highly variable genes...")
sc.pp.highly_variable_genes(adata, n_top_genes=8192, subset=False)

# Get the list of all genes that are in the perturbations
perts = adata.obs.condition.unique()
# Split by +
perts = [pert.split('+') for pert in perts]
perts = [gene for sublist in perts for gene in sublist]
perts = list(set(perts))

# Select the HVGs + the genes in the perturbations
hvg_genes = adata.var_names[adata.var.highly_variable]
genes_to_keep = list(set(hvg_genes) | set(perts))
genes_to_keep = [gene for gene in genes_to_keep if gene in adata.var_names]
adata = adata[:, genes_to_keep]

# Check if there are any genes in perturbations that are not in the new adata
missing_genes = set(perts) - set(adata.var_names)
print(f"Missing genes: {missing_genes}")

print(f"After processing: {adata.shape[0]} cells, {adata.shape[1]} genes")

# %%
# ============================================================================
# STEP 1.5: Pre-assign Technical Duplicate Splits
# ============================================================================
print("\n" + "="*50)
print("STEP 1.5: Pre-assign Technical Duplicate Splits")
print("="*50)

# Set seed for reproducibility
np.random.seed(42)

# Initialize the column with NaN for conditions with <2 cells
adata.obs['tech_dup_split'] = pd.NA

# Process each condition
unique_conditions = adata.obs['condition'].unique()
conditions_with_splits = 0
conditions_without_splits = 0

for condition in tqdm(unique_conditions, desc="Assigning technical duplicate splits"):
    condition_cells = adata.obs[adata.obs['condition'] == condition].index
    
    if len(condition_cells) >= 2:
        # Randomly shuffle and split
        cell_indices = np.random.permutation(condition_cells)
        split_idx = len(cell_indices) // 2
        
        # Assign first half
        adata.obs.loc[cell_indices[:split_idx], 'tech_dup_split'] = 'first_half'
        # Assign second half
        adata.obs.loc[cell_indices[split_idx:], 'tech_dup_split'] = 'second_half'
        
        conditions_with_splits += 1
    else:
        conditions_without_splits += 1
        print(f"  Warning: Condition '{condition}' has <2 cells, cannot create technical duplicates")

print(f"Technical duplicate splits assigned:")
print(f"  Conditions with splits: {conditions_with_splits}")
print(f"  Conditions without splits: {conditions_without_splits}")
print(f"  Split distribution: {adata.obs['tech_dup_split'].value_counts().to_dict()}")

# Validate split balance for each condition
print("\nValidating split balance...")
unbalanced_conditions = []
for condition in unique_conditions:
    cond_data = adata.obs[adata.obs['condition'] == condition]
    if len(cond_data) >= 2 and not pd.isna(cond_data['tech_dup_split'].iloc[0]):
        split_counts = cond_data['tech_dup_split'].value_counts()
        if 'first_half' in split_counts and 'second_half' in split_counts:
            if abs(split_counts['first_half'] - split_counts['second_half']) > 1:
                unbalanced_conditions.append(condition)
                print(f"  Note: Condition '{condition}' has unbalanced split: {split_counts.to_dict()}")

if not unbalanced_conditions:
    print("  All conditions have balanced splits (difference ≤ 1 cell)")

# %%

# ============================================================================
# STEP 2: Add Synthetic Controls
# ============================================================================
print("\n" + "="*50)
print("STEP 2: Add Synthetic Controls")
print("="*50)

def add_synthetic_controls(adata, n_controls=N_SYNTHETIC_CONTROLS):
    """Add synthetic mean controls only"""
    print("Adding synthetic mean controls...")
    
    # Get non-control cells
    non_ctrl_cells = adata[~adata.obs['condition'].str.contains('control')]
    
    # Create mean expression controls
    print("Creating mean expression controls...")
    n_mean_controls = min(n_controls // 4, 100)
    mean_controls = []
    for i in range(n_mean_controls):
        # Sample random cells and take mean
        sampled_for_mean = np.random.choice(non_ctrl_cells.obs_names, size=100, replace=False)
        mean_expr = adata[sampled_for_mean].X.mean(axis=0)
        
        # Create synthetic cell
        obs_df = pd.DataFrame({
            'condition': 'ctrl_synthetic_mean',
            'cell_type': adata.obs['cell_type'].iloc[0],
            'donor_id': 'norman19',
            'ncounts': np.sum(mean_expr),
            'ngenes': np.sum(mean_expr > 0)
        }, index=[f'synthetic_mean_{i}'])
        
        mean_adata = sc.AnnData(X=mean_expr.reshape(1, -1), obs=obs_df, var=adata.var)
        mean_controls.append(mean_adata)
    
    # Combine synthetic controls with original data
    if mean_controls:
        mean_adata_combined = sc.concat(mean_controls, join='outer', index_unique='_')
        adata_with_synthetic = sc.concat([adata, mean_adata_combined], join='outer', index_unique='_')
        adata_with_synthetic.var = adata.var.copy()
        return adata_with_synthetic
    
    return adata

adata = add_synthetic_controls(adata)
print(f"After adding synthetic controls: {adata.shape[0]} cells")

# %%

# ============================================================================
# STEP 3: Train/Test Split
# ============================================================================
print("\n" + "="*50)
print("STEP 3: Train/Test Split")
print("="*50)

def create_splits(adata):
    """Create train/test splits according to specifications (2 Fold cross validation)"""
    print("Creating train/test splits...")
    
    # Initialize split column
    adata.obs['split_fold_0'] = ''
    adata.obs['split_fold_1'] = ''
    
    # Get all conditions
    all_conditions = adata.obs['condition'].unique()
    
    # Separate single vs combo perturbations
    single_perts = [cond for cond in all_conditions if '+' not in cond and 'control' not in cond]
    combo_perts = [cond for cond in all_conditions if '+' in cond and 'control' not in cond]
    control_conditions = [cond for cond in all_conditions if 'control' in cond]
    
    print(f"Found {len(single_perts)} single perturbations, {len(combo_perts)} combo perturbations")
    
    # All single perturbations go to train
    for pert in single_perts:
        adata.obs.loc[adata.obs['condition'] == pert, 'split_fold_0'] = 'train'
        adata.obs.loc[adata.obs['condition'] == pert, 'split_fold_1'] = 'train'
    
    # 50% of combo perturbations go to test, rest split between train/val
    if combo_perts:
        np.random.shuffle(combo_perts)
        n_test = len(combo_perts) // 2
        n_val = (len(combo_perts) - n_test) // 2
        
        # Define test combos for each fold
        fold_0_test_combos = combo_perts[:n_test]
        fold_1_test_combos = combo_perts[n_test:]

        # Define val combos for each fold (50% of test combos of the other fold)
        fold_0_val_combos = fold_1_test_combos[:n_val]
        fold_1_val_combos = fold_0_test_combos[:n_val]

        # Define train combos for each fold (remaining combos of the other fold)
        fold_0_train_combos = fold_1_test_combos[n_val:]
        fold_1_train_combos = fold_0_test_combos[n_val:]
        
        # Assert that the folds are disjoint
        fold_0_int = set(fold_0_test_combos) & set(fold_0_val_combos) & set(fold_0_train_combos)
        fold_1_int = set(fold_1_test_combos) & set(fold_1_val_combos) & set(fold_1_train_combos)
        assert fold_0_int == set() and fold_1_int == set(), "Fold 0 or Fold 1 have overlapping perturbations"

        # Assign splits to each perturbation
        for pert in combo_perts:
            if pert in fold_0_test_combos:
                adata.obs.loc[adata.obs['condition'] == pert, 'split_fold_0'] = 'test'
            if pert in fold_0_val_combos:
                adata.obs.loc[adata.obs['condition'] == pert, 'split_fold_0'] = 'val'
            if pert in fold_0_train_combos:
                adata.obs.loc[adata.obs['condition'] == pert, 'split_fold_0'] = 'train'
            if pert in fold_1_test_combos:
                adata.obs.loc[adata.obs['condition'] == pert, 'split_fold_1'] = 'test'
            if pert in fold_1_val_combos:
                adata.obs.loc[adata.obs['condition'] == pert, 'split_fold_1'] = 'val'
            if pert in fold_1_train_combos:
                adata.obs.loc[adata.obs['condition'] == pert, 'split_fold_1'] = 'train'
    
    # Split control conditions 50/25/25 between train/val/test
    for ctrl_cond in control_conditions:
        ctrl_cells = adata.obs[adata.obs['condition'] == ctrl_cond].index
        n_cells = len(ctrl_cells)
        
        # Shuffle and split
        ctrl_cells_shuffled = np.random.permutation(ctrl_cells)
        n_train = int(0.5 * n_cells)
        n_val = int(0.25 * n_cells)
        
        train_cells_fold_0 = ctrl_cells_shuffled[:n_train]
        val_cells_fold_0 = ctrl_cells_shuffled[n_train:n_train + n_val]
        test_cells_fold_0 = ctrl_cells_shuffled[n_train + n_val:]

        train_cells_fold_1 = ctrl_cells_shuffled[n_train:]
        val_cells_fold_1 = ctrl_cells_shuffled[:n_val]
        test_cells_fold_1 = ctrl_cells_shuffled[n_val:n_train]
        
        adata.obs.loc[train_cells_fold_0, 'split_fold_0'] = 'train'
        adata.obs.loc[val_cells_fold_0, 'split_fold_0'] = 'val'
        adata.obs.loc[test_cells_fold_0, 'split_fold_0'] = 'test'

        adata.obs.loc[train_cells_fold_1, 'split_fold_1'] = 'train'
        adata.obs.loc[val_cells_fold_1, 'split_fold_1'] = 'val'
        adata.obs.loc[test_cells_fold_1, 'split_fold_1'] = 'test'

        # Assert that the folds are disjoint
        fold_0_int = set(train_cells_fold_0) & set(val_cells_fold_0) & set(test_cells_fold_0)
        fold_1_int = set(train_cells_fold_1) & set(val_cells_fold_1) & set(test_cells_fold_1)
        assert fold_0_int == set() and fold_1_int == set(), "Fold 0 or Fold 1 have overlapping control cells"

    # Print split statistics
    split_counts_fold_0 = adata.obs['split_fold_0'].value_counts()
    split_counts_fold_1 = adata.obs['split_fold_1'].value_counts()
    print("Split_fold_0 distribution:")
    for split_type, count in split_counts_fold_0.items():
        print(f"  {split_type}: {count} cells")
    print("Split_fold_1 distribution:")
    for split_type, count in split_counts_fold_1.items():
        print(f"  {split_type}: {count} cells")
    
    return adata

adata = create_splits(adata)

# %%

# ============================================================================
# STEP 4: Calculate DEGs
# ============================================================================
print("\n" + "="*50)
print("STEP 4: Calculate DEGs")
print("="*50)

MIN_CELLS_DEGS = 4  # Changed from 5 to 4

def calculate_degs(adata):
    """Calculate differential expression genes using only second half of technical duplicate split"""
    print("Calculating DEGs using second half of technical duplicate split...")
    
    # Filter to only use second_half cells
    adata_second_half = adata[adata.obs['tech_dup_split'] == 'second_half'].copy()
    print(f"Using {adata_second_half.shape[0]} cells from second half for DEG calculation")
    
    # Filter perturbations with enough cells in second half
    pert_counts = adata_second_half.obs['condition'].value_counts()
    valid_perts = pert_counts[pert_counts >= MIN_CELLS_DEGS].index
    adata_deg = adata_second_half[adata_second_half.obs['condition'].isin(valid_perts)].copy()
    
    print(f"Calculating DEGs for {len(valid_perts)} perturbations with ≥{MIN_CELLS_DEGS} cells in second half")
    
    # Calculate DEGs vs control
    print("Computing DEGs vs rest...")
    sc.tl.rank_genes_groups(adata_deg, 'condition', method='t-test_overestim_var', reference='rest')
    
    # Store results - now including both adjusted and unadjusted p-values
    names_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["names"])
    pvals_adj_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["pvals_adj"])
    pvals_unadj_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["pvals"])  # Unadjusted p-values
    scores_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["scores"])
    
    # Create DEG dictionary (following gene_map_v1 format)
    deg_dict = {}
    for pert in tqdm(adata_deg.obs['condition'].unique(), desc="Processing DEGs"):
        if pert == 'control' or 'ctrl' in pert:
            continue
        pert_degs = names_df[pert]
        pert_degs_sig = pert_degs[pvals_adj_df[pert] < 0.05]  # Still use adjusted for significance
        deg_dict[f'norman19_{pert}'] = pert_degs_sig.tolist()
    
    # Convert dataframes to proper format for h5ad storage (following gene_map_v1 approach)
    # Create flattened dictionaries that can be saved in h5ad format
    names_df_dict_final = {}
    pvals_adj_df_dict_final = {}
    pvals_unadj_df_dict_final = {}  # New: unadjusted p-values
    scores_df_dict_final = {}
    
    for pert in names_df.columns:
        if pert == 'control' or 'ctrl' in pert:
            continue
        # Convert to lists to avoid h5ad serialization issues
        names_df_dict_final[f'norman19_{pert}'] = names_df[pert].tolist()
        pvals_adj_df_dict_final[f'norman19_{pert}'] = pvals_adj_df[pert].tolist()
        pvals_unadj_df_dict_final[f'norman19_{pert}'] = pvals_unadj_df[pert].tolist()  # New
        scores_df_dict_final[f'norman19_{pert}'] = scores_df[pert].tolist()
    
    # Store in adata - now with both p-value types
    adata.uns['deg_gene_dict'] = deg_dict
    adata.uns['names_df_dict'] = names_df_dict_final
    adata.uns['pvals_adj_df_dict'] = pvals_adj_df_dict_final  # Renamed for clarity
    adata.uns['pvals_unadj_df_dict'] = pvals_unadj_df_dict_final  # New: unadjusted p-values
    adata.uns['scores_df_dict'] = scores_df_dict_final
    
    print(f"Calculated DEGs for {len(deg_dict)} perturbations using second half only")
    return adata

adata = calculate_degs(adata)

# %%

# ============================================================================
# STEP 5: Add Baselines
# ============================================================================
print("\n" + "="*50)
print("STEP 5: Add Baselines")
print("="*50)




def add_baselines(adata, split_col='split'):
    """Add various baseline predictions
    
    Args:
        adata: AnnData object to add baselines to
        split_col: Column name in adata.obs to use for train/test split
    """
    print(f"Adding baselines using split column: {split_col}")
    
    # Initialize baselines dictionary to store all baselines for this split
    baselines = {}
    
    # 1. Dataset mean baseline (following gene_map_v1 pattern)
    print("Computing dataset mean baseline...")
    train_cells = adata[adata.obs[split_col] == 'train']
    train_non_ctrl = train_cells[~train_cells.obs['condition'].str.contains('control')]
    
    # Step 1: Get unique donor_id + condition combinations in training set
    unique_donor_condition_combos = train_non_ctrl.obs[['donor_id', 'condition']].drop_duplicates()
    
    # Step 2: For each (donor_id, condition) combo, compute mean expression
    condition_means = pd.DataFrame(index=range(len(unique_donor_condition_combos)), columns=adata.var_names)
    for idx, (_, row) in enumerate(tqdm(unique_donor_condition_combos.iterrows(), desc="Computing condition means")):
        donor_id = row['donor_id']
        condition = row['condition']
        
        # Get cells for this donor_id + condition combination
        combo_cells = train_non_ctrl[
            (train_non_ctrl.obs['donor_id'] == donor_id) & 
            (train_non_ctrl.obs['condition'] == condition)
        ]
        
        if len(combo_cells) > 0:
            combo_mean = combo_cells.X.mean(axis=0)
            if hasattr(combo_mean, 'A1'):
                combo_mean = combo_mean.A1
            condition_means.iloc[idx] = combo_mean
    
    # Step 3: Average across all conditions within each donor_id to get dataset mean
    # For norman19, we only have one donor_id, but follow the general pattern
    mean_baseline_df = pd.DataFrame(index=['norman19'], columns=adata.var_names)
    dataset_mean = condition_means.mean(axis=0)
    mean_baseline_df.loc['norman19'] = dataset_mean
    mean_baseline_df = mean_baseline_df.astype(float)
    baselines['split_mean_baseline'] = mean_baseline_df
    
    # 2. Technical duplicate baseline
    # Only per-fold baselines go here (mean baseline)
    # Control baseline, technical duplicate, and additive are calculated once universally
    
    # Store per-fold baselines in adata.uns
    for key, value in baselines.items():
        adata.uns[f'{split_col}_{key}'] = value
    
    return adata

# ============================================================================
# Calculate Universal Baselines (once, not per-fold)
# ============================================================================

# 1. Control baseline (universal)
print("Computing control baseline...")
ctrl_cells = adata[adata.obs['condition'] == 'control']
if len(ctrl_cells) > 0:
    ctrl_mean = ctrl_cells.X.mean(axis=0)
    if hasattr(ctrl_mean, 'A1'):
        ctrl_mean = ctrl_mean.A1
    
    ctrl_baseline_df = pd.DataFrame(index=['norman19'], columns=adata.var_names)
    ctrl_baseline_df.loc['norman19'] = ctrl_mean
    ctrl_baseline_df = ctrl_baseline_df.astype(float)
    adata.uns['ctrl_baseline'] = ctrl_baseline_df
    print(f"Control baseline created from {len(ctrl_cells)} control cells")

# 2. Technical duplicate baseline (universal)
print("Computing technical duplicate baseline...")
# Use pre-assigned splits from adata.obs['tech_dup_split']

unique_conditions = adata.obs['condition'].unique()
tech_dup_first_half = pd.DataFrame(index=unique_conditions, columns=adata.var_names)
tech_dup_second_half = pd.DataFrame(index=unique_conditions, columns=adata.var_names)

for condition in tqdm(unique_conditions, desc="Computing technical duplicates"):
    condition_cells = adata[adata.obs['condition'] == condition]
    
    # Use pre-assigned splits
    first_half_cells = condition_cells[condition_cells.obs['tech_dup_split'] == 'first_half']
    second_half_cells = condition_cells[condition_cells.obs['tech_dup_split'] == 'second_half']
    
    if len(first_half_cells) > 0 and len(second_half_cells) > 0:
        first_half_mean = first_half_cells.X.mean(axis=0)
        second_half_mean = second_half_cells.X.mean(axis=0)
        
        if hasattr(first_half_mean, 'A1'):
            first_half_mean = first_half_mean.A1
        if hasattr(second_half_mean, 'A1'):
            second_half_mean = second_half_mean.A1
        
        tech_dup_first_half.loc[condition] = first_half_mean
        tech_dup_second_half.loc[condition] = second_half_mean

# Clean up and store
tech_dup_first_half = tech_dup_first_half.dropna().astype(float)
tech_dup_second_half = tech_dup_second_half.dropna().astype(float)

# Convert index to include donor prefix
tech_dup_first_half.index = [f'norman19_{cond}' for cond in tech_dup_first_half.index]
tech_dup_second_half.index = [f'norman19_{cond}' for cond in tech_dup_second_half.index]

adata.uns['technical_duplicate_first_half_baseline'] = tech_dup_first_half
adata.uns['technical_duplicate_second_half_baseline'] = tech_dup_second_half
print(f"Technical duplicate baselines created for {len(tech_dup_first_half)} conditions")

# 3. Additive baseline (universal, computed once from fold_0 training data)
print("Computing additive baseline...")

# Get all conditions
all_conditions = adata.obs['condition'].unique()
control_conditions = [cond for cond in all_conditions if 'control' in cond or 'ctrl' in cond]

# Separate single vs combo perturbations
single_conditions = []
combo_conditions = []
for cond in all_conditions:
    if cond not in control_conditions:
        if '+' in cond:
            combo_conditions.append(cond)
        else:
            single_conditions.append(cond)

# Get training data from fold_0
train_adata = adata[adata.obs['split_fold_0'] == 'train']
train_single_conditions = [cond for cond in train_adata.obs['condition'].unique() if cond in single_conditions]

# Get control baseline
ctrl_cells = train_adata[train_adata.obs['condition'] == 'control']
if len(ctrl_cells) > 0:
    ctrl_mean = ctrl_cells.X.mean(axis=0)
    if hasattr(ctrl_mean, 'A1'):
        ctrl_mean = ctrl_mean.A1
else:
    ctrl_mean = np.zeros(adata.shape[1])

# Compute single perturbation effects (delta from control)
single_effects = {}
for condition in tqdm(train_single_conditions, desc="Computing single effects"):
    condition_cells = train_adata[train_adata.obs['condition'] == condition]
    if len(condition_cells) > 0:
        condition_mean = condition_cells.X.mean(axis=0)
        if hasattr(condition_mean, 'A1'):
            condition_mean = condition_mean.A1
        # Store effect as delta from control
        single_effects[condition] = condition_mean - ctrl_mean

# Create additive predictions for all combo perturbations
additive_predictions = []
for combo_condition in combo_conditions:
    # Parse combo condition
    genes_in_combo = combo_condition.split('+')
    
    # Start with control baseline
    predicted_expression = ctrl_mean.copy()
    
    # Add effects of each single gene in the combo
    for gene in genes_in_combo:
        if gene in single_effects:
            predicted_expression += single_effects[gene]
    
    additive_predictions.append({
        'condition': f'norman19_{combo_condition}',
        'expression': predicted_expression
    })

# Convert to DataFrame
if additive_predictions:
    additive_df = pd.DataFrame(index=[pred['condition'] for pred in additive_predictions], 
                              columns=adata.var_names)
    for i, pred in enumerate(additive_predictions):
        additive_df.iloc[i] = pred['expression']
    
    additive_df = additive_df.astype(float)
    adata.uns['additive_baseline'] = additive_df
    print(f"Created additive baseline for {len(additive_predictions)} combo perturbations")

adata = add_baselines(adata, split_col='split_fold_0')
adata = add_baselines(adata, split_col='split_fold_1')


# %%

# ============================================================================
# STEP 6: Add Gene Embeddings (Optional)
# ============================================================================
print("\n" + "="*50)
print("STEP 6: Gene Embeddings (Optional)")
print("="*50)

# This would call the gather_embeddings.sh script if needed
# For now, we'll skip this step
print("Skipping gene embeddings - can be added later if needed")

# %%

# ============================================================================
# STEP 7: Final Processing and QC
# ============================================================================
print("\n" + "="*50)
print("STEP 7: Final Processing and QC")
print("="*50)

# Add PCA and UMAP for visualization
print("Computing PCA and UMAP...")
sc.pp.pca(adata)
# sc.pp.neighbors(adata)
# sc.tl.umap(adata)

# # Add leiden clustering
# sc.tl.leiden(adata, resolution=0.5)

# Print summary statistics
print("\nFinal dataset summary:")
print(f"Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
print(f"Conditions: {len(adata.obs['condition'].unique())}")
print(f"Split distribution fold 0:")
for split_type, count in adata.obs['split_fold_0'].value_counts().items():
    print(f"  {split_type}: {count} cells")
print(f"Split distribution fold 1:")
for split_type, count in adata.obs['split_fold_1'].value_counts().items():
    print(f"  {split_type}: {count} cells")

print(f"\nBaselines computed:")
for k, v in adata.uns.items():
    if 'baseline' in k:
        print(f"{k}:")
        print(f"  {v.shape[0]} conditions")

# %%

# ============================================================================
# STEP 8: Save Results
# ============================================================================
print("\n" + "="*50)
print("STEP 8: Save Results")
print("="*50)

# Ensure 'gene_name' column exists in adata.var
if 'gene_name' not in adata.var.columns:
    adata.var['gene_name'] = adata.var_names.copy()
adata.var.index.name = None

# Symbol to ensemble_id
import pandas as pd
try:
    converter = pd.read_csv("./data/ref/gene_info_scgpt.csv")
except:
    converter = pd.read_csv("../ref/gene_info_scgpt.csv")
converter = converter[['feature_name', 'feature_id']]
converter.set_index('feature_id', inplace=True)

# Safely map symbols, using a lambda function to handle missing IDs
adata.var['symbol_scgpt'] = adata.var.ensemble_id.map(lambda x: converter.loc[x]['feature_name'] if x in converter.index else pd.NA)


# Save the processed data
output_path = f'{data_cache_dir}/norman19_processed_complete.h5ad'
print(f"Saving processed data to {output_path}")
adata.write_h5ad(output_path)

print(f"\nProcessing complete!")
print(f"Dataset ready for benchmarking!")
