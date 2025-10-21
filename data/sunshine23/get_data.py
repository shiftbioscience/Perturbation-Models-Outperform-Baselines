# %%
import scanpy as sc
import os
import subprocess as sp
from scipy.sparse import csr_matrix
import pandas as pd
from tqdm import tqdm
import numpy as np

np.random.seed(42)

data_url = 'https://zenodo.org/record/10044268/files/SunshineHein2023.h5ad'
data_cache_dir = 'data/sunshine23'
DATASET_NAME = 'sunshine23'
CELL_TYPE = 'calu3'
COMBO_SEPARATOR = '_'

if not os.path.exists(data_cache_dir):
    os.makedirs(data_cache_dir)

tmp_data_dir = f'{data_cache_dir}/{DATASET_NAME}_downloaded.h5ad'

if not os.path.exists(tmp_data_dir):
    sp.call(f'wget -q {data_url} -O {tmp_data_dir}', shell=True)

adata = sc.read_h5ad(tmp_data_dir)

# %%

# Remove duplicates
non_dup_indices = adata.var_names.duplicated(keep=False)
adata = adata[:, ~non_dup_indices]
if 'ensembl_id' in adata.var.columns:
    adata.var['gene_id'] = adata.var.ensembl_id
adata.var.index.name = None


# Process perturbation labels - keep all perturbation labels as requested
adata.obs['perturbation'] = adata.obs['perturbation'].astype('category')
adata.obs['condition'] = adata.obs.perturbation.copy()

# Standardize combo separator to "+" as early as possible
if COMBO_SEPARATOR != '+':
    print(f"Replacing combo separator '{COMBO_SEPARATOR}' with '+' in condition labels...")
    adata.obs['condition'] = adata.obs['condition'].str.replace(COMBO_SEPARATOR, '+', regex=False)
    COMBO_SEPARATOR = '+'
    print(f"Combo separator standardized to '+'")

# Add donor_id column (required by pipeline but constant for sunshine23)
adata.obs['donor_id'] = DATASET_NAME

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

MAX_CELLS_CONTROL = 8192


# Calculate mean cells per perturbation and set MAX_CELLS to closest power of 2
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

for pert in tqdm(pert_counts.index, desc="Downsampling perturbations"):
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
# Split by underscore for combo separator
perts = [pert.split(COMBO_SEPARATOR) for pert in perts]
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

adata.obs['cell_type'] = CELL_TYPE

# %%
# ============================================================================
# STEP 3.5: Pre-assign Technical Duplicate Splits
# ============================================================================
print("\n" + "="*50)
print("STEP 3.5: Pre-assign Technical Duplicate Splits")
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

# %% Cross Validation for Combo Prediction (like Norman19)
N_FOLDS = 2  # Norman19 uses 2 folds for combo prediction
# ============================================================================
# STEP 3: Cross Validation for Combo Prediction Task
# ============================================================================

# For combo prediction, we split based on single vs combo perturbations
# Single perturbations go to training, combo perturbations are split between train/val/test

# Get all conditions
all_conditions = adata.obs['condition'].unique()
control_conditions = [cond for cond in all_conditions if 'control' in cond.lower() or 'ctrl' in cond.lower()]
non_control_conditions = [cond for cond in all_conditions if cond not in control_conditions]

# Separate single and combo perturbations
single_conditions = []
combo_conditions = []
for cond in non_control_conditions:
    if COMBO_SEPARATOR in cond:
        combo_conditions.append(cond)
    else:
        single_conditions.append(cond)

print(f"Single perturbations: {len(single_conditions)}")
print(f"Combo perturbations: {len(combo_conditions)}")
print(f"Control conditions: {len(control_conditions)}")

# For combo prediction task (following Norman19):
# - All single perturbations go to training set
# - 50% of combo perturbations go to test (split between folds)
# - Remaining 50% are split between train and val
# - Control cells are distributed across all splits

np.random.seed(42)
shuffled_combos = np.random.permutation(combo_conditions)

# 50% of combo perturbations go to test, rest split between train/val
n_combos = len(shuffled_combos)
n_test = n_combos // 2
n_val = (n_combos - n_test) // 2

# Define test combos for each fold
fold_0_test_combos = shuffled_combos[:n_test]
fold_1_test_combos = shuffled_combos[n_test:]

# Define val combos for each fold (50% of test combos of the other fold)
fold_0_val_combos = fold_1_test_combos[:n_val]
fold_1_val_combos = fold_0_test_combos[:n_val]

# Define train combos for each fold (remaining combos of the other fold)
fold_0_train_combos = fold_1_test_combos[n_val:]
fold_1_train_combos = fold_0_test_combos[n_val:]

print(f"\nFold 0 - Test combos: {len(fold_0_test_combos)}, Val combos: {len(fold_0_val_combos)}, Train combos: {len(fold_0_train_combos)}")
print(f"Fold 1 - Test combos: {len(fold_1_test_combos)}, Val combos: {len(fold_1_val_combos)}, Train combos: {len(fold_1_train_combos)}")

# Get control cells for splitting
control_cells = []
for ctrl_cond in control_conditions:
    control_cells.extend(adata.obs[adata.obs['condition'] == ctrl_cond].index.tolist())

shuffled_control_cells = np.random.permutation(control_cells)

# Create splits for combo prediction (following Norman19 exactly)
for fold in range(N_FOLDS):
    print(f"\nFold {fold}:")
    
    # Initialize split column
    adata.obs[f'split_fold_{fold}'] = 'unassigned'
    
    if fold == 0:
        test_conditions = fold_0_test_combos.tolist()
        val_conditions = fold_0_val_combos.tolist()
        train_combo_conditions = fold_0_train_combos.tolist()
    else:
        test_conditions = fold_1_test_combos.tolist()
        val_conditions = fold_1_val_combos.tolist()
        train_combo_conditions = fold_1_train_combos.tolist()
    
    # All single perturbations go to train
    train_conditions = single_conditions + train_combo_conditions
    
    # Split control cells across splits (roughly 50% train, 25% val, 25% test)
    n_control = len(shuffled_control_cells)
    train_control_end = int(0.5 * n_control)
    val_control_end = int(0.75 * n_control)
    
    train_control_cells = shuffled_control_cells[:train_control_end]
    val_control_cells = shuffled_control_cells[train_control_end:val_control_end]
    test_control_cells = shuffled_control_cells[val_control_end:]
    
    print(f"  Single perturbations in train: {len(single_conditions)}")
    print(f"  Combo perturbations in train: {len(train_combo_conditions)}")
    print(f"  Combo perturbations in val: {len(val_conditions)}")
    print(f"  Combo perturbations in test: {len(test_conditions)}")
    print(f"  Control cells - Train: {len(train_control_cells)}, Val: {len(val_control_cells)}, Test: {len(test_control_cells)}")
    
    # Assign conditions to splits
    adata.obs.loc[adata.obs['condition'].isin(train_conditions), f'split_fold_{fold}'] = 'train'
    adata.obs.loc[adata.obs['condition'].isin(val_conditions), f'split_fold_{fold}'] = 'val'
    adata.obs.loc[adata.obs['condition'].isin(test_conditions), f'split_fold_{fold}'] = 'test'
    
    # Assign control cells to splits
    adata.obs.loc[train_control_cells, f'split_fold_{fold}'] = 'train'
    adata.obs.loc[val_control_cells, f'split_fold_{fold}'] = 'val'
    adata.obs.loc[test_control_cells, f'split_fold_{fold}'] = 'test'
    
    # Verify the split
    cell_counts = adata.obs[f'split_fold_{fold}'].value_counts()
    total_assigned = cell_counts.sum()
    
    print(f"  Cell distribution:")
    for split_type in ['train', 'val', 'test']:
        if split_type in cell_counts:
            count = cell_counts[split_type]
            pct = count / total_assigned * 100
            print(f"    {split_type}: {count} cells ({pct:.1f}%)")

print(f"\nCombo prediction split assignment complete!")

# %%

# ============================================================================
# STEP 4: Calculate DEGs
# ============================================================================
print("\n" + "="*50)
print("STEP 4: Calculate DEGs")
print("="*50)
MIN_CELLS_DEGS = 4

def calculate_degs(adata):
    """Calculate differential expression genes using only second half of technical duplicate split"""
    print("Calculating DEGs using second half of technical duplicate split...")
    
    # Filter to only use second_half cells
    adata_second_half = adata[adata.obs['tech_dup_split'] == 'second_half'].copy()
    print(f"Using {adata_second_half.shape[0]} cells from second half for DEG calculation")
    
    # Filter perturbations with enough cells in second half (include both single and combo)
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
        if pert in control_conditions:
            continue
        pert_degs = names_df[pert]
        pert_degs_sig = pert_degs[pvals_adj_df[pert] < 0.05]  # Still use adjusted for significance
        deg_dict[f'{DATASET_NAME}_{pert}'] = pert_degs_sig.tolist()
    
    # Convert dataframes to proper format for h5ad storage (following gene_map_v1 approach)
    # Create flattened dictionaries that can be saved in h5ad format
    names_df_dict_final = {}
    pvals_adj_df_dict_final = {}
    pvals_unadj_df_dict_final = {}  # New: unadjusted p-values
    scores_df_dict_final = {}
    
    for pert in names_df.columns:
        if pert in control_conditions:
            continue
        # Convert to lists to avoid h5ad serialization issues
        names_df_dict_final[f'{DATASET_NAME}_{pert}'] = names_df[pert].tolist()
        pvals_adj_df_dict_final[f'{DATASET_NAME}_{pert}'] = pvals_adj_df[pert].tolist()
        pvals_unadj_df_dict_final[f'{DATASET_NAME}_{pert}'] = pvals_unadj_df[pert].tolist()  # New
        scores_df_dict_final[f'{DATASET_NAME}_{pert}'] = scores_df[pert].tolist()
    
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
    mean_baseline_df = pd.DataFrame(index=[DATASET_NAME], columns=adata.var_names)
    dataset_mean = condition_means.mean(axis=0)
    mean_baseline_df.loc[DATASET_NAME] = dataset_mean
    mean_baseline_df = mean_baseline_df.astype(float)
    baselines['mean_baseline'] = mean_baseline_df
    
    # Technical duplicate baseline is calculated once universally
    
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
    
    ctrl_baseline_df = pd.DataFrame(index=[DATASET_NAME], columns=adata.var_names)
    ctrl_baseline_df.loc[DATASET_NAME] = ctrl_mean
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
tech_dup_first_half.index = [f'{DATASET_NAME}_{cond}' for cond in tech_dup_first_half.index]
tech_dup_second_half.index = [f'{DATASET_NAME}_{cond}' for cond in tech_dup_second_half.index]

adata.uns['technical_duplicate_first_half_baseline'] = tech_dup_first_half
adata.uns['technical_duplicate_second_half_baseline'] = tech_dup_second_half
print(f"Technical duplicate baselines created for {len(tech_dup_first_half)} conditions")

# 3. Additive baseline (universal, computed once from fold_0 training data)
print("Computing additive baseline...")
train_adata = adata[adata.obs['split_fold_0'] == 'train']
train_single_conditions = [cond for cond in train_adata.obs['condition'].unique() if cond in single_conditions]

# Get control baseline
ctrl_cells = train_adata[train_adata.obs['condition'].isin(control_conditions)]
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
    # Parse combo condition (using underscore separator)
    genes_in_combo = combo_condition.split(COMBO_SEPARATOR)
    
    # Start with control baseline
    predicted_expression = ctrl_mean.copy()
    
    # Add effects of each single gene in the combo
    for gene in genes_in_combo:
        if gene in single_effects:
            predicted_expression += single_effects[gene]
    
    additive_predictions.append({
        'condition': f'{DATASET_NAME}_{combo_condition}',
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

# Add baselines for both folds
for fold in range(N_FOLDS):
    adata = add_baselines(adata, split_col=f'split_fold_{fold}')



# ============================================================================
# STEP 7: Add Sparse Mean Baselines
# ============================================================================
print("\n" + "="*50)
print("STEP 7: Add Sparse Mean Baselines")
print("="*50)

def calculate_universal_sparse_mean_baseline(adata, dataset_name, seed=0):
    """
    Calculate a single sparsity-matched mean baseline for all perturbations across all folds.
    This creates a fair comparison with technical duplicate baselines by 
    matching the sampling sparsity. For each test perturbation with N cells,
    we subsample N/2 cells from training perturbations to match the sparsity
    of the technical duplicate (which uses N/2 cells for its prediction).
    
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
            if n_subsample > 0 and len(train_non_ctrl) >= n_subsample:
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

# Calculate sparse mean baseline with three different seeds for reproducibility
SEEDS = [0, 1, 2]
for seed in SEEDS:
    print(f"Calculating sparse mean baseline with seed {seed}...")
    sparse_baseline = calculate_universal_sparse_mean_baseline(
        adata, DATASET_NAME, seed=seed
    )
    
    # Save with seed-specific key
    key = f'sparse_mean_baseline_seed_{seed}'
    adata.uns[key] = sparse_baseline
    print(f"  ✓ Created {key} for {len(sparse_baseline)} conditions")

# ============================================================================
# STEP 8: Save Results
# ============================================================================
print("\n" + "="*50)
print("STEP 8: Save Results")
print("="*50)

# Ensure 'gene_name' column exists in adata.var
if 'gene_name' not in adata.var.columns:
    adata.var['gene_name'] = adata.var_names

# Ensure ensemble_id column exists in adata.var
if 'ensemble_id' not in adata.var.columns:
    adata.var['ensemble_id'] = adata.var.gene_id.copy()

# Symbol to ensemble_id
import pandas as pd
try:
    converter = pd.read_csv("data/ref/gene_info_scgpt.csv")
except:
    converter = pd.read_csv("data/ref/gene_info_scgpt.csv")

converter = converter[['feature_name', 'feature_id']]
converter.set_index('feature_id', inplace=True)

# Safely map symbols, using a lambda function to handle missing IDs
adata.var['symbol_scgpt'] = adata.var.gene_id.map(lambda x: converter.loc[x]['feature_name'] if x in converter.index else pd.NA)


adata.var.index.name = None

# Save the processed data
output_path = f'{data_cache_dir}/{DATASET_NAME}_processed_complete.h5ad'
print(f"Saving processed data to {output_path}")
adata.write_h5ad(output_path)
