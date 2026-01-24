"""
Data management for CellSimBench framework.

This module provides the central data management system for handling AnnData objects,
DEG weights, baselines, and perturbation conditions.
"""

from typing import Dict, List, Tuple, Optional, Union
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# Import metrics functions from existing codebase
def mse(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculate Mean Squared Error.
    
    Args:
        x1: First array of values.
        x2: Second array of values.
        
    Returns:
        Mean squared error between x1 and x2.
    """
    return np.mean((x1 - x2) ** 2)

def wmse(x1: np.ndarray, x2: np.ndarray, weights: np.ndarray) -> float:
    """Calculate Weighted Mean Squared Error.
    
    Args:
        x1: First array of values.
        x2: Second array of values.
        weights: Weight array for each element.
        
    Returns:
        Weighted mean squared error between x1 and x2.
    """
    weights_arr = np.array(weights)
    x1_arr = np.array(x1)
    x2_arr = np.array(x2)
    normalized_weights = weights_arr / np.sum(weights_arr)
    return np.sum(normalized_weights * ((x1_arr - x2_arr) ** 2))

def pearson(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient.
    
    Args:
        x1: First array of values.
        x2: Second array of values.
        
    Returns:
        Pearson correlation coefficient between x1 and x2.
    """
    return np.corrcoef(x1, x2)[0, 1]

def r2_score_on_deltas(delta_true: np.ndarray, delta_pred: np.ndarray, 
                      weights: Optional[np.ndarray] = None) -> float:
    """Calculate R² score on deltas with optional weighting.
    
    Args:
        delta_true: True delta values.
        delta_pred: Predicted delta values.
        weights: Optional weight array for weighted R².
        
    Returns:
        R² score between true and predicted deltas.
    """
    from sklearn.metrics import r2_score
    if len(delta_true) < 2 or len(delta_pred) < 2 or delta_true.shape != delta_pred.shape:
        return np.nan
    if weights is not None and np.sum(weights) != 0:
        return r2_score(delta_true, delta_pred, sample_weight=weights)
    else:
        return r2_score(delta_true, delta_pred)


class DataManager:
    """Handles data loading and DEG weights extraction for CellSimBench.
    
    This class manages AnnData objects containing perturbation response data,
    computes and caches DEG weights, and provides utilities for accessing
    baselines, splits, and perturbation conditions.
    
    Attributes:
        config: Dataset configuration dictionary.
        adata: Loaded AnnData object with expression data.
        deg_names_dict: Dictionary mapping perturbations to DEG names.
        deg_scores_dict: Dictionary mapping perturbations to DEG scores.
        deg_pvals_dict: Dictionary mapping perturbations to DEG p-values.
        pert_normalized_abs_scores_vsrest: Precomputed normalized DEG weights.
        
    Example:
        >>> config = {'data_path': 'data.h5ad', 'covariate_key': 'donor_id'}
        >>> dm = DataManager(config)
        >>> adata = dm.load_dataset()
        >>> weights = dm.get_deg_weights('donor1', 'GENE1')
    """
    
    def __init__(self, dataset_config: Dict) -> None:
        """Initialize DataManager with dataset configuration.
        
        Args:
            dataset_config: Dictionary containing dataset configuration including
                           data_path, covariate_key, and baseline keys.
        """
        self.config = dataset_config
        self.adata: Optional[sc.AnnData] = None
        self.deg_names_dict: Optional[Dict[str, List[str]]] = None
        self.deg_scores_dict: Optional[Dict[str, np.ndarray]] = None
        self.deg_pvals_dict: Optional[Dict[str, np.ndarray]] = None
        self.pert_normalized_abs_scores_vsrest: Dict[str, np.ndarray] = {}
        
    def load_dataset(self) -> sc.AnnData:
        """Load the h5ad file and precompute DEG weights.
        
        Returns:
            Loaded AnnData object with precomputed DEG weights.
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If required DEG data is not found in dataset.
        """
        path = Path(self.config['data_path'])
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        print(f"Loading dataset from {path}...")
        self.adata = sc.read_h5ad(path)
        self.adata.var.index.name = None
        print(f"Loaded AnnData with shape: {self.adata.shape}")
        
        # Extract DEG dictionaries - MANDATORY for CellSimBench operation
        try:
            self.deg_names_dict = self.adata.uns['names_df_dict_gt']
            self.deg_scores_dict = self.adata.uns['scores_df_dict_gt']
            self.deg_pvals_dict = self.adata.uns.get('pvals_adj_df_dict_gt', self.adata.uns.get('pvals_adj_df_dict_gt'))
                        
            # Precompute normalized weights for all perturbations
            self._precompute_deg_weights()
            print(f"Precomputed DEG weights for {len(self.pert_normalized_abs_scores_vsrest)} perturbations")
            
        except KeyError as e:
            raise ValueError(f"Required DEG data not found in dataset: {e}. "
                           f"Dataset must contain precomputed DEG information in uns['names_df_dict_gt'] "
                           f"and uns['scores_df_dict_gt']. Please use a properly processed dataset.")
        
        return self.adata
    
    def _precompute_deg_weights(self) -> None:
        """Precompute normalized DEG weights following plotting.py logic.
        
        Computes min-max normalized and squared weights for each perturbation
        based on DEG scores. Weights are cached in pert_normalized_abs_scores_vsrest.
        """
        self.pert_normalized_abs_scores_vsrest_df = {}
        for cov_pert_key in tqdm(self.deg_scores_dict.keys(), desc="Calculating Weights"):
            if 'control' in cov_pert_key.lower():
                continue
            
            # Get scores and names for this covariate-perturbation combination
            scores = self.deg_scores_dict[cov_pert_key]
            gene_names = self.deg_names_dict[cov_pert_key]
            
            # Convert to absolute scores
            abs_scores = np.abs(scores)
            
            # Min-max normalization
            min_val = np.min(abs_scores)
            max_val = np.max(abs_scores)

            normalized_weights = (abs_scores - min_val) / (max_val - min_val)
            
            # Handle NaNs
            normalized_weights = np.nan_to_num(normalized_weights, nan=0.0)
            
            # Square the weights for stronger emphasis
            normalized_weights = np.square(normalized_weights)
            
            # Create series and handle duplicates by taking the maximum weight for each gene
            weights_df = pd.DataFrame({
                'gene': gene_names,
                'weight': normalized_weights
            })
            
            # Group by gene and take the maximum weight in case of duplicates
            weights_aggregated = weights_df.groupby('gene')['weight'].max()
            
            # Reindex to match adata.var_names
            weights = weights_aggregated.reindex(self.adata.var_names, fill_value=0.0)
            
            self.pert_normalized_abs_scores_vsrest[cov_pert_key] = weights.values
            self.pert_normalized_abs_scores_vsrest_df[cov_pert_key] = weights
    
    def get_available_controls(self) -> List[str]:
        """Get all available control types from the data.
        
        Returns:
            List of control condition names.
        """
        control_conditions = self.adata.obs['condition'][
            self.adata.obs['condition'].str.contains('ctrl', case=False, na=False)
        ].unique()
        return control_conditions.tolist()
    
    def get_available_splits(self) -> List[str]:
        """Get all available split columns from the data.
        
        Returns:
            List of split column names.
        """
        split_columns = [col for col in self.adata.obs.columns if 'split' in col.lower()]
        return split_columns
    
    def get_deg_weights(self, covariate_value: str, perturbation: str, gene_order: List[str]) -> np.ndarray:
        """
        Get DEG-based weights for a specific covariate-perturbation combination.
        
        Args:
            covariate_value: Value of the covariate (e.g., donor ID)
            perturbation: Perturbation identifier
            gene_order: Ordered list of gene names to align weights to.
            
        Returns:
            Array of weights aligned with gene_order
        """
        cov_pert_key = f"{covariate_value}_{perturbation}"
        
        if cov_pert_key in self.pert_normalized_abs_scores_vsrest:
            weights = self.pert_normalized_abs_scores_vsrest_df[cov_pert_key]
        else:
            # Return zero weights if no DEG data available
            return np.zeros(len(gene_order))

        # Reindex weights to match the target gene order
        weights = weights.reindex(gene_order, fill_value=0.0)

        return weights.values
    
    def get_deg_mask(self, covariate_value: str, perturbation: str, gene_order: List[str], pval_threshold: float = 0.05) -> np.ndarray:
        """
        Get DEG mask for a specific covariate-perturbation combination.
        
        Args:
            covariate_value: Value of the covariate (e.g., donor ID)
            perturbation: Perturbation identifier
            gene_order: Ordered list of gene names to align the mask to.
            pval_threshold: P-value threshold for significance
            
        Returns:
            Boolean array indicating DEG positions, aligned to gene_order
        """
        cov_pert_key = f"{covariate_value}_{perturbation}"
        
        if cov_pert_key not in self.deg_pvals_dict:
            return np.zeros(len(gene_order), dtype=bool)
        
        # Get p-values and gene names (these are in DEG rank order, not var_names order)
        pvals = self.deg_pvals_dict[cov_pert_key]
        gene_names = self.deg_names_dict[cov_pert_key]
        
        # Create boolean mask for significant genes
        sig_mask = pvals < pval_threshold
        
        # Handle duplicates by taking the minimum p-value (most significant) for each gene
        pvals_df = pd.DataFrame({
            'gene': gene_names,
            'pval': pvals,
            'significant': sig_mask
        })
        
        # Group by gene and take minimum p-value, then check significance
        pvals_aggregated = pvals_df.groupby('gene')['pval'].min()
        deg_mask_aggregated = pvals_aggregated < pval_threshold

        
        # Reindex to match the target gene order
        deg_mask = deg_mask_aggregated.reindex(gene_order, fill_value=False)
        
        return deg_mask.values
    
    def get_control_baseline(self, donor_id: Optional[str] = None) -> np.ndarray:
        """Get control baseline expression from uns using dataset-specific key.
        
        Args:
            donor_id: Optional donor/covariate ID for donor-specific baseline.
            
        Returns:
            Control baseline expression array.
            
        Raises:
            ValueError: If donor not found or control baseline not available.
        """
        control_baseline_key = self.config['control_baseline_key']
        if control_baseline_key not in self.adata.uns:
            # Fallback to calculating control mean from data
            warnings.warn(f"Control baseline key '{control_baseline_key}' not found in uns. Calculating from data.")
            control_conditions = self.get_available_controls()
            if control_conditions:
                return self.adata[self.adata.obs['condition'] == control_conditions[0]].X.mean(axis=0).A1
            else:
                raise ValueError("No control conditions found in data")
        
        baseline_data = self.adata.uns[control_baseline_key]
        
        # Handle DataFrame format (donor-specific baselines)
        if hasattr(baseline_data, 'index') and hasattr(baseline_data, 'columns'):
            if donor_id is not None:
                # Return specific donor baseline
                if donor_id in baseline_data.index:
                    return baseline_data.loc[donor_id].values
                else:
                    raise ValueError(f"Donor '{donor_id}' not found in control baseline. Available: {list(baseline_data.index)}")
            else:
                # Return mean across all donors
                return baseline_data.mean(axis=0).values
        else:
            # Handle array format (single baseline)
            return baseline_data
    
    def get_control_baseline_dict(self) -> Dict[str, np.ndarray]:
        """Get all donor-specific control baselines as a dictionary.
        
        Returns:
            Dictionary mapping donor IDs to control baseline arrays.
            
        Raises:
            ValueError: If control baseline not found in dataset.
        """
        control_baseline_key = self.config['control_baseline_key']
        if control_baseline_key not in self.adata.uns:
            raise ValueError(f"Control baseline key '{control_baseline_key}' not found in uns")
        
        baseline_data = self.adata.uns[control_baseline_key]
        
        # Handle DataFrame format (donor-specific baselines)
        if hasattr(baseline_data, 'index') and hasattr(baseline_data, 'columns'):
            return {donor_id: baseline_data.loc[donor_id].values for donor_id in baseline_data.index}
        else:
            # Handle array format (single baseline) - return with generic key
            return {'default': baseline_data}
    
    def get_ground_truth_baseline(self) -> Dict[str, np.ndarray]:
        """Get ground truth baseline expressions from uns using dataset-specific key.
        
        Returns:
            Dictionary mapping covariate-perturbation keys to ground truth arrays.
            
        Raises:
            ValueError: If ground truth baseline not found in dataset.
        """
        ground_truth_baseline_key = self.config['ground_truth_baseline_key']
        if ground_truth_baseline_key not in self.adata.uns:
            raise ValueError(f"Ground truth baseline key '{ground_truth_baseline_key}' not found in uns")
        
        baseline_data = self.adata.uns[ground_truth_baseline_key]
        
        # Handle DataFrame format (covariate_perturbation-specific baselines)
        if hasattr(baseline_data, 'index') and hasattr(baseline_data, 'columns'):
            return {cov_pert_key: baseline_data.loc[cov_pert_key].values 
                    for cov_pert_key in baseline_data.index}
        else:
            # Handle dictionary format (already properly structured)
            return baseline_data
    
    def get_dataset_mean_baseline(self) -> Dict[str, np.ndarray]:
        """Get technical duplicate second half baseline expressions from uns.
        
        This is ALWAYS required as it serves as the control for all delta metrics.
        
        Returns:
            Dictionary mapping covariate-perturbation keys to baseline arrays.
            
        Raises:
            ValueError: If dataset mean baseline not found (required for delta metrics).
        """
        # Try to get from config first, with fallback to standard key name
        dataset_mean_baseline_key = self.config.dataset.dataset_mean_baseline_key
        
        if dataset_mean_baseline_key not in self.adata.uns:
            raise ValueError(f"Dataset mean baseline key '{dataset_mean_baseline_key}' not found in uns. "
                           f"This baseline is REQUIRED for delta metrics calculation.")
        
        baseline_data = self.adata.uns[dataset_mean_baseline_key]
        
        # Handle DataFrame format (covariate_perturbation-specific baselines)
        if hasattr(baseline_data, 'index') and hasattr(baseline_data, 'columns'):
            return {cov_pert_key: baseline_data.loc[cov_pert_key].values 
                    for cov_pert_key in baseline_data.index}
        else:
            # Handle dictionary format (already properly structured)
            return baseline_data
    
    def load_obs_only(self) -> pd.DataFrame:
        """Load only the observation metadata without the full expression matrix.
        
        Efficient method for accessing metadata without loading expression data.
        
        Returns:
            DataFrame containing observation metadata.
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist.
        """
        import h5py
        from anndata.experimental import read_elem
        
        path = Path(self.config['data_path'])
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
                
        print(f"Loading obs metadata from {path}...")
        with h5py.File(path, 'r') as f:
            obs = read_elem(f['obs'])
        
        return obs
    
    def get_perturbation_conditions(self, split_name: str, obs: Optional[pd.DataFrame] = None) -> Dict[str, List[str]]:
        """
        Get train/test perturbation conditions for a given split.
        
        Args:
            split_name: Name of the split column
            obs: Optional obs DataFrame. If not provided, uses self.adata.obs
            
        Returns:
            Dict with 'train' and 'test' lists of conditions
        """
        if obs is None:
            if self.adata is None:
                raise ValueError("Either provide obs parameter or load dataset first")
            obs = self.adata.obs
            
        if split_name not in obs.columns:
            raise ValueError(f"Split '{split_name}' not found in obs")
        
        split_data = obs[split_name]
        
        train_conditions = obs[split_data == 'train']['condition'].unique().tolist()
        val_conditions = obs[split_data == 'val']['condition'].unique().tolist()
        test_conditions = obs[split_data == 'test']['condition'].unique().tolist()

        # Remove control conditions
        train_conditions = [condition for condition in train_conditions if 'ctrl' not in condition]
        val_conditions = [condition for condition in val_conditions if 'ctrl' not in condition]
        test_conditions = [condition for condition in test_conditions if 'ctrl' not in condition]
        
        return {
            'train': train_conditions,
            'val': val_conditions,
            'test': test_conditions
        }
    
    def get_covariate_condition_pairs(self, split_name: str, split_type: str = 'test') -> List[Tuple[str, str]]:
        """Get all covariate-condition pairs for a given split.
        
        Args:
            split_name: Name of the split column.
            split_type: Either 'train', 'val', or 'test'.
            
        Returns:
            List of (covariate_value, condition) tuples.
            
        Raises:
            ValueError: If split_name not found in obs columns.
        """
        if split_name not in self.adata.obs.columns:
            raise ValueError(f"Split '{split_name}' not found in obs")
        
        split_mask = self.adata.obs[split_name] == split_type
        split_data = self.adata.obs[split_mask]
        
        covariate_key = self.config['covariate_key']
        if covariate_key not in self.adata.obs.columns:
            raise ValueError(f"Covariate key '{covariate_key}' not found in obs")
        
        pairs = split_data[[covariate_key, 'condition']].drop_duplicates()
        
        return [(str(row[covariate_key]), row['condition']) for _, row in pairs.iterrows()] 