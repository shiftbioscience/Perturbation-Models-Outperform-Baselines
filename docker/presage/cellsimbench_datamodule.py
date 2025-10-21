# SPDX-License-Identifier: LicenseRef-Genentech-NonCommercial-1.0
# Copyright (c) 2025 Genentech, Inc.
#
# Licensed under the Genentech Non-Commercial Software License Version 1.0 (September 2022).
# You may not use this file except in compliance with the License.
# A copy of the License is included in this directory as "docker/presage/LICENSE" and in built images at "/LICENSE".
#
# NOTICE OF MODIFICATIONS:
# This file was created or modified by the CellSimBench team to integrate PRESAGE with CellSimBench.
# See docker/presage/MODIFICATIONS.md for a summary of changes.
"""
Custom PRESAGE DataModule for CellSimBench data.
Inherits from PRESAGEDataModule but handles CellSimBench-specific data format.
"""

import os
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
from typing import Dict, List
import logging
from tqdm import tqdm

import sys
sys.path.insert(0, '/presage_src' if os.path.exists('/presage_src') else 'ref/PRESAGE/src')

from presage_datamodule import PRESAGEDataModule
import torch
from torch.utils.data import Dataset
from anndata import AnnData

log = logging.getLogger(__name__)


def compute_pseudobulk(
    adata: AnnData, condition_field: str = "perturbation"
) -> pd.DataFrame:

    return pd.DataFrame(
        adata.X, index=adata.obs[condition_field], columns=adata.var_names
    ).pipe(lambda df: df.groupby(df.index).mean())

class CellSimBenchscPerturbData(Dataset):
    """Interface for a preprocessed AnnData object derived from scPerturb.org

    Preprocessing defined in `prepare_data` and/or `setup` methods of `scPerturbDataModule`.

    Needs to identify and separate control cells.

    Needs to compute pseudobulk and implement option to generate samples from
    either pseudobulk or single cells

    Needs to implement an invertible mapping between
    perturbation keys and indicators over adata.var_names
    """

    def __init__(
        self,
        adata,
        pert_covariates=None,
        perturb_field="perturbation",
        control_key="control",
        use_pseudobulk=False,
        z_score=False,
    ):
        self.adata = adata

        self.pert_covariates = pert_covariates

        self.perturb_field = perturb_field
        self.control_key = control_key

        # separate control cells
        self.perturbs = adata

        perturb_keys = self.perturbs.obs[self.perturb_field].to_numpy()
        self.X = self.perturbs.X.astype(np.float32)

        self.perturb_keys = perturb_keys
        self.var_names = adata.var_names

        self.indmtx = np.vstack(
            [self.pert_to_ind(key) for key in self.perturb_keys]
        ).astype(np.float32)

        # for now, drop perturbations of genes that aren't measured
        not_observed = self.indmtx.sum(1) == 0
        if not_observed.any():
            not_observed_keys = set(self.perturb_keys[not_observed])
            print(
                f"WARNING: Data contain perturbations for {len(not_observed_keys)} genes "
                f"for which there is no mRNA expression measurement: {not_observed_keys}.\n"
                "They will be removed because they have all-0 indicator variables."
            )
            observed = ~not_observed
            self.X = self.X[observed]
            self.perturb_keys = self.perturb_keys[observed]
            self.indmtx = self.indmtx[observed]

        self.covmtx = np.zeros((self.indmtx.shape[0], 0))
        self.covmtx = self.covmtx.astype(np.float32)

    def pert_to_ind(self, pert_key):
        """Convert perturbation key to indicator vector."""
        ind = np.zeros(len(self.adata.var))
        gene_to_idx = {gene: i for i, gene in enumerate(self.adata.var.index)}
            
        # Handle single and combo perturbations
        if "_" in pert_key and pert_key != "control":
            # Combo perturbation
            genes = pert_key.split("_")
        else:
            # Single perturbation
            genes = [pert_key] if pert_key != "control" else []
            
        for gene in genes:
            if gene in gene_to_idx:
                ind[gene_to_idx[gene]] = 1
                
        return ind

    def ind_to_pert(self, ind) -> str:
        if hasattr(ind, "numpy"):
            ind = ind.numpy()
        ind = ind > 0
        key = "_".join(self.var_names[ind])
        if key not in self.perturb_keys:
            key = "_".join(reversed(self.var_names[ind]))
        assert key in self.perturb_keys, f"Could not find perturb key {key}"
        return key

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):

        batch_data = dict(
            inds=torch.tensor(self.indmtx[i]),
            expr=torch.tensor(self.X[i]),
            cov=torch.tensor(self.covmtx[i]),
        )
        
        # Include the perturbation key for identification
        pert_key = self.perturb_keys[i]
        batch_data['pert_key'] = pert_key
        
        return batch_data


class CellSimBenchDataModule(PRESAGEDataModule):
    """Custom PRESAGE DataModule for CellSimBench data."""
    
    def __init__(self, processed_adata_path: str, splits_path: str, 
                 data_dir: str, dataset: str = "cellsimbench", **kwargs):
        """
        Initialize CellSimBench DataModule.
        
        Args:
            processed_adata_path: Path to the processed AnnData file
            splits_path: Path to JSON file with train/val/test splits
            data_dir: Base data directory
            dataset: Dataset name (default: "cellsimbench")
            **kwargs: Additional arguments - ALL REQUIRED:
                - batch_size
                - use_pseudobulk
                - preprocessing_zscore
                - perturb_field
                - control_key
                - dataset_class
        """
        # Store our custom paths
        self.processed_adata_path = Path(processed_adata_path)
        self.splits_json_path = Path(splits_path)
        
        # Initialize LightningDataModule base class
        import pytorch_lightning as pl
        pl.LightningDataModule.__init__(self)
        
        # Skip PRESAGEDataModule's dataset validation
        # We need to set required attributes manually
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        
        # REQUIRED parameters - will raise KeyError if missing
        self.batch_size = kwargs['batch_size']
        self.use_pseudobulk = kwargs['use_pseudobulk']
        
        self.perturb_field = kwargs['perturb_field']
        self.control_key = kwargs['control_key']
        self.dataset_class = kwargs['dataset_class']
        
        # Set up directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.deg_dir, exist_ok=True)
        
        # Initialize parent attributes
        self.n_genes = None
        self.degs = None
        self.var_names = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._data_prepared = False
        self._data_setup = False
        
        # Additional parent class attributes
        self.nperturb_clusters = None  # Required by parent setup() method
        self.allow_list = None
        self.allow_list_out_genes = None
        self.X_train_pca = None
        
        # Set split path
        self.split_path = str(self.splits_json_path)
        
        log.info(f"Initialized CellSimBenchDataModule with dataset: {dataset}")
        
        # Store split name if provided (for proper control filtering)
        self.split_name = kwargs['split_name']  # Required
        # Store whether to use perturbation mean as delta reference (required)
        self.perts_as_delta_ref = kwargs['perts_as_delta_ref']

    def compute_means(self, adata):
        """Compute control and perturbation means."""
        ctrl_cells = adata[adata.obs[self.perturb_field] == self.control_key]
        pert_cells = adata[adata.obs[self.perturb_field] != self.control_key]
        
        if len(ctrl_cells) == 0:
            raise ValueError(f"No control cells found with {self.perturb_field}=={self.control_key}")
        
        # Compute control mean
        ctrl_x = ctrl_cells.X
        if hasattr(ctrl_x, 'toarray'):
            ctrl_x = ctrl_x.toarray()
        control_mean = np.mean(ctrl_x, axis=0, keepdims=True)
        control_mean_df = pd.DataFrame(
            control_mean, 
            columns=ctrl_cells.var.index
        )
        log.info(f"Computed control mean using {len(ctrl_cells)} cells")
        
        # Compute perturbation mean (balanced average across all perturbations)
        pert_means_list = []
        perturbations_included = []
        
        for pert in adata.obs[self.perturb_field].unique():
            if pert == self.control_key:
                continue
                
            # Get cells for this perturbation
            pert_cells_subset = adata[adata.obs[self.perturb_field] == pert]
            
            if len(pert_cells_subset) > 0:
                pert_x = pert_cells_subset.X
                if hasattr(pert_x, 'toarray'):
                    pert_x = pert_x.toarray()
                
                # Compute mean for this perturbation
                pert_mean = np.mean(pert_x, axis=0)
                pert_means_list.append(pert_mean)
                perturbations_included.append(pert)
        
        # Average across all perturbations (equal weight to each)
        if len(pert_means_list) == 0:
            raise ValueError("No perturbations found")
        
        # Balanced mean: each perturbation contributes equally
        perturbation_mean = np.mean(np.array(pert_means_list), axis=0, keepdims=True)
        
        perturbation_mean_df = pd.DataFrame(
            perturbation_mean,
            columns=adata.var.index
        )
        log.info(f"Computed perturbation mean using {len(perturbations_included)} perturbations")
        
        return control_mean_df, perturbation_mean_df
    
    def create_dataset(self, adata, train: bool):
        """Create dataset with pseudobulk processing."""
        # Compute control and perturbation means
        control_mean, perturbation_mean = self.compute_means(adata)
        
        # For pseudobulk, compute pseudobulk for each perturbation
        if self.use_pseudobulk:
            pseudobulk_data = []
            pseudobulk_keys = []
            
            # Process each perturbation
            for pert in tqdm(adata.obs[self.perturb_field].unique(), desc="Computing pseudobulk"):
                if pert == self.control_key:
                    continue
                    
                pert_cells = adata[adata.obs[self.perturb_field] == pert]
                
                # Compute pseudobulk for this perturbation
                if hasattr(pert_cells.X, 'toarray'):
                    expr = pert_cells.X.toarray()
                else:
                    expr = pert_cells.X
                
                # Average expression across cells
                pseudobulk_expr = np.mean(expr, axis=0)
                
                # Center using either control or perturbation mean based on configuration
                if self.perts_as_delta_ref:
                    reference_mean = perturbation_mean.values.flatten()
                else:
                    reference_mean = control_mean.values.flatten()
                centered_expr = pseudobulk_expr - reference_mean
                
                # Store the data
                pseudobulk_data.append(centered_expr)
                pseudobulk_keys.append(pert)
            
            # Convert to numpy arrays
            X = np.array(pseudobulk_data, dtype=np.float32)
            perturb_keys = np.array(pseudobulk_keys)
            
            # Create modified adata for the dataset
            modified_obs = pd.DataFrame({
                self.perturb_field: perturb_keys
            })
            modified_adata = sc.AnnData(X=X, obs=modified_obs, var=adata.var)
            
            # Create the dataset with the modified data
            dataset = CellSimBenchscPerturbData(
                modified_adata,
                use_pseudobulk=False,  # Set to False since we already computed pseudobulk
                pert_covariates=None
            )
            
        else:
            raise NotImplementedError("Non-pseudobulk not implemented")
        
        # Store control and perturbation means
        dataset.control_mean = control_mean
        dataset.perturbation_mean = perturbation_mean
        dataset.perts_as_delta_ref = self.perts_as_delta_ref

        return dataset
    
    @property
    def dataset_dir(self) -> Path:
        """Dataset-specific directory."""
        return self.data_dir / self.dataset
    
    @property
    def deg_dir(self) -> Path:
        """Directory for DEG files."""
        return self.dataset_dir / "degs"
    
    @property
    def preprocessed_path(self) -> str:
        """Path to preprocessed data."""
        return str(self.processed_adata_path)
    
    @property
    def raw_path(self) -> str:
        """Path to raw data (same as preprocessed for us)."""
        return str(self.processed_adata_path)
    
    @property
    def merged_deg_file(self) -> str:
        """Path to merged DEG JSON file."""
        return str(self.deg_dir / "merged.degs.json")
    
    def prepare_data(self) -> None:
        """
        Prepare data for PRESAGE training.
        For CellSimBench, data is already prepared, we just need to extract DEGs.
        """
        if not self._data_prepared:
            log.info("Preparing CellSimBench data...")
            
            # Load the processed data
            adata = sc.read(self.processed_adata_path)
            
            # Extract and process DEGs if not already done
            if not os.path.exists(self.merged_deg_file):
                log.info("Processing DEGs from CellSimBench data...")
                
                if 'deg_gene_dict' in adata.uns:
                    # CellSimBench format: covariate_key_perturbation -> list of DEGs
                    deg_dict = adata.uns['deg_gene_dict']
                    
                    # Process keys to extract just perturbation names
                    # Format: "replogle22rpe1_UTP23" -> "UTP23"
                    processed_degs = {}
                    for key, genes in deg_dict.items():
                        # Extract perturbation name after first underscore
                        match = re.match(r'^[^_]+_(.+)$', key)
                        if match:
                            pert_name = match.group(1)
                            # Convert perturbation format to match PRESAGE expectations
                            pert_name = pert_name.replace("+", "_")
                            processed_degs[pert_name] = list(genes) if hasattr(genes, '__iter__') else [genes]
                    
                    # Save processed DEGs
                    with open(self.merged_deg_file, 'w') as f:
                        json.dump(processed_degs, f)
                    log.info(f"Saved {len(processed_degs)} perturbation DEGs to {self.merged_deg_file}")
                    
                else:
                    log.warning("No DEG information found in adata.uns. Creating empty DEG file.")
                    with open(self.merged_deg_file, 'w') as f:
                        json.dump({}, f)
            else:
                log.info(f"Found existing DEG file at {self.merged_deg_file}")
            
            self._data_prepared = True
    
    def setup(self, stage=None):
        """
        Override parent setup to properly handle control samples by split assignment.
        This fixes the data leakage issue where all control samples were included in all splits.
        """
        if not self._data_setup:
            log.info(f"Setting up data for stage: {stage}")
            
            # Load preprocessed data and splits
            adata = self.load_preprocessed()
            
            # Load split assignments
            with open(self.split_path, "r") as f:
                splits = json.load(f)
            self.splits = splits
            
            # Get var names and degs from loaded data
            self.var_names = adata.var_names
            self.n_genes = len(self.var_names)
            
            # Load DEGs
            if os.path.exists(self.merged_deg_file):
                with open(self.merged_deg_file, "r") as f:
                    self.degs = json.load(f)
            else:
                self.degs = {}
            
            # Filter splits to only include perturbations that exist in adata
            for key in splits:
                valid_perturbations = []
                for pert in splits[key]:
                    # Check if this perturbation exists
                    exists = (adata.obs[self.perturb_field] == pert).any()
                    if exists:
                        valid_perturbations.append(pert)
                splits[key] = valid_perturbations

                        
            if stage == "fit":
                # Create train and validation datasets WITH PROPER CONTROL FILTERING
                subsets = {"train": splits["train"], "val": splits["val"]}
                for name, subset in subsets.items():
                    # Create mask for perturbations in this subset
                    condition_mask = pd.Series([False] * len(adata), index=adata.obs.index)
                    
                    for pert in subset:
                        # Add cells matching this perturbation
                        pert_mask = (adata.obs[self.perturb_field] == pert)
                        condition_mask |= pert_mask
                    
                    # Get control samples that belong to THIS SPECIFIC SPLIT
                    control_mask = (
                        (adata.obs[self.perturb_field] == self.control_key) &
                        (adata.obs[self.split_name] == name)  # Only controls from this split
                    )
                    
                    # Combine condition and control masks
                    combined_mask = condition_mask | control_mask
                    split_adata = adata[combined_mask]
                    # Remove any cells that are not in the split
                    split_adata = split_adata[split_adata.obs[self.split_name] == name]

                    # Create dataset for this split
                    setattr(
                        self,
                        f"{name}_dataset",
                        self.create_dataset(
                            split_adata,
                            train=(name == "train"),
                        ),
                    )

                    log.info(f"Created {name} dataset with {combined_mask.sum()} samples "
                            f"({condition_mask.sum()} conditions + {control_mask.sum()} controls)")

                self.train_perturb_labels = None
            
            if stage == "test":
                self.nperturb_clusters = None
                self.train_perturb_labels = None
                
                # Create test dataset WITH PROPER CONTROL FILTERING
                condition_mask = pd.Series([False] * len(adata), index=adata.obs.index)
                
                for pert in splits["test"]:
                    # Add cells matching this perturbation
                    pert_mask = (adata.obs[self.perturb_field] == pert)
                    condition_mask |= pert_mask
                
                # Get control samples that belong to TEST SPLIT ONLY
                control_mask = (
                    (adata.obs[self.perturb_field] == self.control_key) &
                    (adata.obs[self.split_name] == 'test')  # Only test-split controls
                )

                # Combine masks - NO TRAINING CONDITIONS ADDED!
                combined_mask = condition_mask | control_mask
                split_adata = adata[combined_mask]
                split_adata = split_adata[split_adata.obs[self.split_name] == 'test']
                self.test_dataset = self.create_dataset(
                    split_adata,
                    train=False,
                )
                
                log.info(f"Created test dataset with {combined_mask.sum()} samples "
                        f"({condition_mask.sum()} conditions + {control_mask.sum()} controls)")
                log.info("FIXED: Not adding training conditions to test set!")
            
            self._data_setup = True
    
    @classmethod
    def from_config(cls, config: Dict):
        """Create datamodule from config dict, compatible with PRESAGE."""
        from copy import deepcopy
        import random
        import numpy as np
        import torch
        
        config = deepcopy(config)
        
        # Set seed if provided
        if 'seed' in config:
            seed = config['seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Extract required paths - will raise KeyError if missing
        processed_adata_path = config.pop('processed_adata_path')
        splits_path = config.pop('splits_path')
        
        # Handle dataset_class like parent from_config
        config["dataset_class"] = CellSimBenchscPerturbData
        # Create instance with remaining config as kwargs
        return cls(
            processed_adata_path=processed_adata_path,
            splits_path=splits_path,
            **config
        ) 
    
    def load_preprocessed(self):
        log.info("Loading adata...")
        adata = sc.read(self.preprocessed_path)

        if hasattr(adata.X, "toarray"):
            adata.X = adata.X.toarray()
        self.n_genes = adata.shape[1]
        log.info("Loading DEGs...")
        with open(self.merged_deg_file) as fp:
            self.degs = json.load(fp)
        deg_dir = "/".join(self.merged_deg_file.split("/")[:-1])

        parent_data_dir = "/".join(deg_dir.split("/")[:-1]) + "/"

        # perturbation cluster file for eval
        self.pclust_file = parent_data_dir + "eval.stratification.clusters.json"
        # genesets for virtual screen
        self.gs_file = parent_data_dir + "virtual.screen.genesets.json"

        # Find the file matching f"{parent_data_dir}/ncells_per_perturbation*"
        import glob
        ncells_per_perturbation_files = glob.glob(f"{parent_data_dir}/ncells_per_perturbation*")
        if len(ncells_per_perturbation_files) == 0:
            self.ncells_per_perturbation_file = None
        else:
            self.ncells_per_perturbation_file = ncells_per_perturbation_files[0]

        if self.ncells_per_perturbation_file is not None:
            cells_per_perturbation = dict(adata.obs.value_counts("perturbation"))
            cells_per_perturbation_temp = {
                i: int(j) for i, j in cells_per_perturbation.items()
            }
            with open(self.ncells_per_perturbation_file, "w") as f:
                json.dump(cells_per_perturbation_temp, f)

        self.var_names = adata.var_names
        self.pseudobulk = compute_pseudobulk(adata, self.perturb_field)

        self.centered_pseudobulk = (
            self.pseudobulk - self.pseudobulk.loc[self.control_key]
        )

        return adata
