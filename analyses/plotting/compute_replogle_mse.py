import scanpy as sc
import numpy as np
import pandas as pd

adata = sc.read_h5ad("../../data/cellsimbench_natmeth/replogle22k562gwps_processed_complete.h5ad") 

# %% MSE comparison plot: Mean Baseline vs Tech. Dup. errors across all perturbations
deg_gene_dict = adata.uns.get('deg_gene_dict_gt', {})

# Get all perturbation names (excluding controls)
all_perturbations = [pert for pert in adata.uns['technical_duplicate_first_half_baseline'].index 
                     if 'control' not in pert.lower()]

# Initialize arrays to store MSE values and DEG counts -- [Coding Agent]
mse_td_list = []
mse_mb_list = []
mse_id_list = []
deg_counts_list = []

# Get mean baseline (same for all perturbations) and convert to numpy once -- [Coding Agent]
mean_baseline_vals = adata.uns['split_fold_0_mean_baseline'].iloc[0].values

# Get all data at once to avoid repeated .loc calls -- [Coding Agent]
gt_df = adata.uns['technical_duplicate_first_half_baseline']
td_df = adata.uns['technical_duplicate_second_half_baseline']
id_df = adata.uns['interpolated_duplicate_baseline']

# Compute MSE for each perturbation
for pert in all_perturbations:
    # Get data as numpy arrays directly -- [Coding Agent]
    gt_vals = gt_df.loc[pert].values
    td_vals = td_df.loc[pert].values
    id_vals = id_df.loc[pert].values
    # Compute MSE(Tech. Dup., GT) and MSE(Mean Baseline, GT)
    mse_td = np.mean((td_vals - gt_vals) ** 2)
    mse_mb = np.mean((mean_baseline_vals - gt_vals) ** 2)
    mse_id = np.mean((id_vals - gt_vals) ** 2)

    mse_td_list.append(mse_td)
    mse_mb_list.append(mse_mb)
    mse_id_list.append(mse_id)

    # Get DEG count for this perturbation
    n_degs = len(deg_gene_dict.get(pert, []))
    deg_counts_list.append(n_degs)

# Convert to arrays
mse_td_array = np.array(mse_td_list)
mse_mb_array = np.array(mse_mb_list)
mse_id_array = np.array(mse_id_list)
deg_counts_array = np.array(deg_counts_list)

# Create dataframe with all the data
mse_comparison_df = pd.DataFrame({
    'perturbation': all_perturbations,
    'mse_tech_dup': mse_td_array,
    'mse_mean_baseline': mse_mb_array,
    'mse_interpolated_dup': mse_id_array,
    'deg_count': deg_counts_array
})

# Save to CSV
mse_comparison_df.to_csv('mse_comparison_data_replogle22k562gwps.csv', index=False)