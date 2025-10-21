# %%
import scanpy as sc
import os
import subprocess as sp
from scipy.sparse import csr_matrix
import pandas as pd
from tqdm import tqdm
import numpy as np

np.random.seed(42)

data_url = 'https://zenodo.org/records/15213619/files/RPE1_CRISPRa_final_population.h5ad?download=1'
data_cache_dir = 'data/kaden25rpe1'
DATASET_NAME = 'kaden25rpe1'
CELL_TYPE = 'rpe1'

if not os.path.exists(data_cache_dir):
    os.makedirs(data_cache_dir)

tmp_data_dir = f'{data_cache_dir}/{DATASET_NAME}_downloaded.h5ad'

if not os.path.exists(tmp_data_dir):
    sp.call(f'wget -q "{data_url}" -O {tmp_data_dir}', shell=True)

adata = sc.read_h5ad(tmp_data_dir)
adata.obs['perturbation'] = adata.obs['guide_target']
adata.obs['perturbation'] = adata.obs['perturbation'].str.replace('non', 'control')
adata.var['ensemble_id'] = adata.var_names.copy()
adata.var.index = adata.var['gene_name'].copy()
adata.var.index.name = None

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


# Add donor_id column (required by pipeline but constant for kaden25rpe1)
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

# %% 5 Fold Cross Validation with Deterministic 70:10:20 split
N_FOLDS = 5
# ============================================================================
# STEP 3: Deterministic Cross Validation with Fixed 70:10:20 Split
# ============================================================================

# Get unique conditions (excluding controls)
all_conditions = adata.obs['condition'].unique()
non_control_conditions = sorted([cond for cond in all_conditions if 'control' not in cond and 'ctrl' not in cond])

print(f"Total conditions to split: {len(non_control_conditions)}")

# Get control cells
control_cells = sorted(adata.obs[adata.obs['condition'].isin(['control', 'ctrl'])].index.tolist())

# Deterministically shuffle conditions and control cells with fixed seed
np.random.seed(42)
shuffled_conditions = np.random.permutation(non_control_conditions)
shuffled_control_cells = np.random.permutation(control_cells)

# Calculate how many conditions per fold for test set (20% each fold)
n_conditions = len(shuffled_conditions)
conditions_per_fold = n_conditions // N_FOLDS
remainder_conditions = n_conditions % N_FOLDS

# Calculate how many control cells per fold for test set (20% each fold)
n_control_cells = len(shuffled_control_cells)
control_cells_per_fold = n_control_cells // N_FOLDS
remainder_control = n_control_cells % N_FOLDS

# Pre-assign all conditions to their test fold
condition_to_test_fold = {}
idx = 0
for fold in range(N_FOLDS):
    # Distribute remainder evenly across first folds
    fold_size = conditions_per_fold + (1 if fold < remainder_conditions else 0)
    fold_conditions = shuffled_conditions[idx:idx + fold_size]
    for cond in fold_conditions:
        condition_to_test_fold[cond] = fold
    idx += fold_size

# Pre-assign all control cells to their test fold
control_to_test_fold = {}
idx = 0
for fold in range(N_FOLDS):
    # Distribute remainder evenly across first folds
    fold_size = control_cells_per_fold + (1 if fold < remainder_control else 0)
    fold_control = shuffled_control_cells[idx:idx + fold_size]
    for cell in fold_control:
        control_to_test_fold[cell] = fold
    idx += fold_size

# Now create the splits for each fold
for fold in range(N_FOLDS):
    print(f"\nFold {fold}:")
    
    # Get test conditions for this fold
    test_conditions = [cond for cond, f in condition_to_test_fold.items() if f == fold]
    
    # Get remaining conditions (not in test)
    remaining_conditions = [cond for cond, f in condition_to_test_fold.items() if f != fold]
    
    # Split remaining conditions into val (10% of total) and train (70% of total)
    # Since test is 20%, remaining is 80%. We need val=10% and train=70% of total
    n_val_conditions = int(len(remaining_conditions) * 0.1375)  # 10% of total = 13.75% of remaining
    val_conditions = remaining_conditions[:n_val_conditions]
    train_conditions = remaining_conditions[n_val_conditions:]
    
    # Get test control cells for this fold
    test_control_cells = [cell for cell, f in control_to_test_fold.items() if f == fold]
    
    # Get remaining control cells (not in test)
    remaining_control = [cell for cell, f in control_to_test_fold.items() if f != fold]
    
    # Split remaining control cells similarly
    n_val_control = int(len(remaining_control) * 0.1375)  # 10% of total = 13.75% of remaining
    val_control_cells = remaining_control[:n_val_control]
    train_control_cells = remaining_control[n_val_control:]
    sum_of_all = len(test_conditions) + len(val_conditions) + len(train_conditions) 
    print(f"  Conditions - Train: {len(train_conditions)} ({len(train_conditions) / sum_of_all * 100:.1f}%), Val: {len(val_conditions)} ({len(val_conditions) / sum_of_all * 100:.1f}%), Test: {len(test_conditions)} ({len(test_conditions) / sum_of_all * 100:.1f}%)")
    print(f"  Control cells - Train: {len(train_control_cells)}, Val: {len(val_control_cells)}, Test: {len(test_control_cells)}")
    
    # Initialize split column
    adata.obs[f'split_fold_{fold}'] = 'unassigned'

    # Assert no overlap between train, val, and test
    assert set(train_conditions) & set(val_conditions) & set(test_conditions) == set(), "Train, val, and test conditions overlap"
    assert set(train_control_cells) & set(val_control_cells) & set(test_control_cells) == set(), "Train, val, and test control cells overlap"
    
    # Assign conditions to splits
    adata.obs.loc[adata.obs['condition'].isin(test_conditions), f'split_fold_{fold}'] = 'test'
    adata.obs.loc[adata.obs['condition'].isin(val_conditions), f'split_fold_{fold}'] = 'val'
    adata.obs.loc[adata.obs['condition'].isin(train_conditions), f'split_fold_{fold}'] = 'train'
    
    # Assign control cells to splits
    adata.obs.loc[test_control_cells, f'split_fold_{fold}'] = 'test'
    adata.obs.loc[val_control_cells, f'split_fold_{fold}'] = 'val'
    adata.obs.loc[train_control_cells, f'split_fold_{fold}'] = 'train'
    
    # Verify the split percentages
    cell_counts = adata.obs[f'split_fold_{fold}'].value_counts()
    total_assigned = cell_counts.sum()
    
    print(f"  Cell distribution:")
    for split_type in ['train', 'val', 'test']:
        if split_type in cell_counts:
            count = cell_counts[split_type]
            pct = count / total_assigned * 100
            print(f"    {split_type}: {count} cells ({pct:.1f}%)")

print(f"\nSplit assignment complete!")
print("Verifying mutual exclusivity of test sets...")

# Verify that each condition appears in test set exactly once
all_test_conditions = set()
for fold in range(N_FOLDS):
    fold_test_conditions = [cond for cond, f in condition_to_test_fold.items() if f == fold]
    overlap = all_test_conditions.intersection(fold_test_conditions)
    if overlap:
        print(f"WARNING: Conditions {overlap} appear in multiple test folds!")
    all_test_conditions.update(fold_test_conditions)

print(f"Total unique conditions in test sets: {len(all_test_conditions)}")
print(f"Total non-control conditions: {len(non_control_conditions)}")
print(f"All conditions covered: {len(all_test_conditions) == len(non_control_conditions)}")

# Assert that no CELLS overlap between test conditions of different folds
test_sets = [set(adata.obs.index[adata.obs[f'split_fold_{fold}'] == 'test']) for fold in range(N_FOLDS)]
for i in range(N_FOLDS):
    for j in range(i+1, N_FOLDS):
        overlap = test_sets[i].intersection(test_sets[j])
        assert len(overlap) == 0, f"Test sets of folds {i} and {j} have {len(overlap)} overlapping cells!"

# For every fold, assert that no cells are found in more than one split
for fold in range(N_FOLDS):
    train_cells = set(adata.obs.index[adata.obs[f'split_fold_{fold}'] == 'train'])
    val_cells = set(adata.obs.index[adata.obs[f'split_fold_{fold}'] == 'val'])
    test_cells = set(adata.obs.index[adata.obs[f'split_fold_{fold}'] == 'test'])
    
    train_val_overlap = train_cells.intersection(val_cells)
    train_test_overlap = train_cells.intersection(test_cells)
    val_test_overlap = val_cells.intersection(test_cells)
    
    assert len(train_val_overlap) == 0, f"Fold {fold}: {len(train_val_overlap)} cells in both train and val!"
    assert len(train_test_overlap) == 0, f"Fold {fold}: {len(train_test_overlap)} cells in both train and test!"
    assert len(val_test_overlap) == 0, f"Fold {fold}: {len(val_test_overlap)} cells in both val and test!"

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
    
    # Filter perturbations with enough cells in second half
    pert_counts = adata_second_half.obs['condition'].value_counts()
    valid_perts = pert_counts[(pert_counts >= MIN_CELLS_DEGS) & (pert_counts.index.isin(non_control_conditions))].index
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
        deg_dict[f'{DATASET_NAME}_{pert}'] = pert_degs_sig.tolist()
    
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
from sklearn.decomposition import PCA
from scipy import sparse

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
    
    # 2. Technical duplicate baseline is calculated once universally

    # 3. Linear baseline calculation starts here
    def solve_Y_GWP(Y, G, P, G_ridge=0.01, P_ridge=0.01):
        """
        Solves the bilinear equation Y = GWP using ridge regression. Based in the R implementation from:
        https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/main/benchmark/src/run_linear_pretrained_model.R
        """
        
        # Convert sparse matrices to dense if needed
        if sparse.issparse(Y):
            Y = Y.toarray()
        if G is not None and sparse.issparse(G):
            G = G.toarray()
        if P is not None and sparse.issparse(P):
            P = P.toarray()

        # Ensure all arrays are float64 type to avoid object arrays
        Y = np.asarray(Y, dtype=np.float64)
        if G is not None:
            G = np.asarray(G, dtype=np.float64)
        if P is not None:
            P = np.asarray(P, dtype=np.float64)
        
        # Center the data
        b = np.mean(Y, axis=1, keepdims=True)
        Y = Y - b
        
        if G is not None and P is not None:
            assert Y.shape[0] == G.shape[0], f"Rows of Y({Y.shape[0]}) must match rows of G({G.shape[0]})"
            assert Y.shape[1] == P.shape[0], f"Columns of Y({Y.shape[1]}) must match rows of P({P.shape[0]})"
            
            # Solve using the bilinear form
            GT_G = (G.T @ G) + (G_ridge * np.eye(G.shape[1]))
            PT_P = (P.T @ P) + (P_ridge * np.eye(P.shape[1]))
            Gm = np.linalg.inv(GT_G) @ G.T
            Pm = P @ np.linalg.inv(PT_P)
            W = Gm @ Y @ Pm
        
        # Replace NaN values with 0
        W = np.nan_to_num(W, nan=0)

        return {"W": W, "b": b}

    def compute_linear_baseline(y_train_df, n_components=10, G_ridge=0.1, P_ridge=0.1):
        """Compute unseen gene linear baseline for a single donor/dataset/covariate combination.
        This is able to predict every gene that is originally measured"""

        # Compute PCA
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(y_train_df)
        
        # Define G matrix
        G = pd.DataFrame(pca.components_.T, index=y_train_df.columns)
        
        # Get P matrix making sure every condition was a measured gene
        p_list = y_train_df.index.tolist()
        p_list = [p for p in p_list if p in G.index]
        P = G.loc[p_list, :].copy()

        # Refine y_train_df (Remove conditions that were not a measured gene)
        filtered_y_index = [g for g in y_train_df.index if g in G.index]
        y_train_df = y_train_df.loc[filtered_y_index, :]
        assert y_train_df.columns.tolist() == G.index.tolist()
        y_train_df = y_train_df.T

        # Get linear results and predictions
        linear_results = solve_Y_GWP(y_train_df.values, G.values, P.values, G_ridge=G_ridge, P_ridge=P_ridge)
        baseline_results = G.values @ linear_results['W'] @ G.T.values + linear_results['b']
        
        # Convert predictions to DataFrame
        baseline_predictions = pd.DataFrame(baseline_results.T, 
                                    index=G.index, 
                                    columns=y_train_df.index)
        baseline_predictions = baseline_predictions[baseline_predictions.index.notna()]
        baseline_predictions.index.name = 'condition'
        
        return baseline_predictions

    # 4. Linear baseline
    train_adata = adata[adata.obs[split_col] == 'train']
    # Get unique donor id condition combinations in train adata
    unique_donor_id_condition_combos = train_adata.obs[['donor_id', 'condition']].drop_duplicates()

    # Remove anything containing the ctrl substring and the + substring
    unique_donor_id_condition_combos = unique_donor_id_condition_combos[~unique_donor_id_condition_combos.condition.str.contains('control')]
    unique_donor_id_condition_combos = unique_donor_id_condition_combos[~unique_donor_id_condition_combos.condition.str.contains('ctrl')]
    unique_donor_id_condition_combos = unique_donor_id_condition_combos[~unique_donor_id_condition_combos.condition.str.contains('+', regex=False)]
    # Make a new dataframe to store mean expression for each donor id condition combination
    multi_index = pd.MultiIndex.from_frame(unique_donor_id_condition_combos)
    pseudobulk_df = pd.DataFrame(index=multi_index, columns=adata.var_names)
    # For each unique donor id condition combination, compute mean expression
    for _, row in tqdm(unique_donor_id_condition_combos.iterrows(), total=unique_donor_id_condition_combos.shape[0], leave=False, desc=f'pseudobulking'):
        curr_donor_id = row['donor_id']
        curr_condition = row['condition']

        # Get cells in train adata that match donor id and condition
        curr_cells_bool = (train_adata.obs['donor_id'] == curr_donor_id) & (train_adata.obs['condition'] == curr_condition)
        curr_cells = train_adata.obs[curr_cells_bool].index.tolist()

        # Compute mean expression
        curr_mean_expression = train_adata[curr_cells].X.mean(axis=0)
        pseudobulk_df.loc[(curr_donor_id, curr_condition), :] = curr_mean_expression

    # Keep donor information in pseudobulk_df
    pseudobulk_df['condition'] = pseudobulk_df.index.get_level_values(1)
    pseudobulk_df['donor_id'] = pseudobulk_df.index.get_level_values(0)
    pseudobulk_df_final = pseudobulk_df.copy()
    pseudobulk_df_final.reset_index(drop=True, inplace=True)

    # Get unique donors and conditions
    unique_donors = pseudobulk_df_final['donor_id'].unique()
    unique_conditions = pseudobulk_df_final['condition'].unique()

    # Initialize dictionary to store predictions per donor
    donor_predictions = {}

    # Process each donor separately
    for donor in tqdm(unique_donors, desc=f'Processing donors', leave=False):
        # Get data for current donor
        donor_data = pseudobulk_df_final[pseudobulk_df_final['donor_id'] == donor]
        donor_data.set_index('condition', inplace=True)
        
        # Create y_train_df for this donor
        y_train_df = pd.DataFrame(index=donor_data.index, columns=adata.var_names)
        for condition in y_train_df.index:
            y_train_df.loc[condition, :] = donor_data.loc[condition, :]

        # Define linear baseline parameters (Set as in the original paper)
        n_components = 10
        G_ridge = 0.1 #  0.1
        P_ridge = 0.1 # 0.1

        # Compute linear baseline (add control expression for delta model)
        curr_predictions = compute_linear_baseline(y_train_df, n_components, G_ridge, P_ridge)
        
        # Store predictions for this donor
        donor_predictions[donor] = curr_predictions

    # Combine all donor predictions for standard baseline
    all_predictions = []
    for donor, pred_df in donor_predictions.items():
        pred_df = pred_df.copy()
        pred_df.index = pd.MultiIndex.from_product([[donor], pred_df.index], 
                                                    names=['donor_id', 'condition'])
        all_predictions.append(pred_df)

    baseline_predictions = pd.concat(all_predictions)
    baseline_predictions = baseline_predictions.sort_index(level=[0, 1])
    baseline_predictions.columns = baseline_predictions.columns.astype(str)
    baseline_predictions = baseline_predictions.astype(float)
    baseline_predictions.index = baseline_predictions.index.map(lambda x: '_'.join(x))
    baselines['linear_baseline'] = baseline_predictions
    
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

# 3. Calculate per-fold baselines
for fold in range(N_FOLDS):
    adata = add_baselines(adata, split_col=f'split_fold_{fold}')



# ============================================================================
# STEP 8: Save Results
# ============================================================================
print("\n" + "="*50)
print("STEP 8: Save Results")
print("="*50)

# Ensure 'gene_name' column exists in adata.var
if 'gene_name' not in adata.var.columns:
    adata.var['gene_name'] = adata.var_names

adata.var.index.name = None

# Save the processed data
output_path = f'{data_cache_dir}/{DATASET_NAME}_processed_complete.h5ad'
print(f"Saving processed data to {output_path}")
adata.write_h5ad(output_path)
