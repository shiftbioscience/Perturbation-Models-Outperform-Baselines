"""
Baseline model execution for CellSimBench framework.

Handles execution of hardcoded baseline models that are stored in the dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scanpy as sc
from .data_manager import DataManager

log = logging.getLogger(__name__)


class BaselineRunner:
    """Executes hardcoded baseline models efficiently.
    
    This class runs baseline models that are pre-computed and stored in the
    dataset's uns dictionary, such as control mean, dataset mean, and
    technical duplicate baselines.
    
    Attributes:
        data_manager: DataManager instance for accessing baseline data.
        
    Example:
        >>> runner = BaselineRunner(data_manager)
        >>> baselines = runner.run_all_baselines(['ctrl_baseline'], 'split')
    """
    
    def __init__(self, data_manager: DataManager) -> None:
        """Initialize BaselineRunner with a DataManager instance.
        
        Args:
            data_manager: DataManager for accessing baseline data.
        """
        self.data_manager = data_manager
        
    def run_all_baselines(self, baseline_names: List[str], split_name: str) -> Dict[str, sc.AnnData]:
        """Run all specified baselines for this dataset.
        
        Args:
            baseline_names: List of baseline names to run (e.g., 'ctrl_baseline',
                          'split_mean_baseline', 'technical_duplicate_baseline').
            split_name: Name of the split to use for evaluation.
            
        Returns:
            Dictionary mapping baseline_name to predictions AnnData objects.
        """
        baseline_results: Dict[str, sc.AnnData] = {}
        
        for baseline_name in baseline_names:
            log.info(f"Running baseline: {baseline_name}")
            predictions = self._run_baseline(baseline_name, split_name)
            baseline_results[baseline_name] = predictions
                
        return baseline_results
    
    def load_baseline(self, baseline_key: str, baseline_type: str, split_name: str) -> sc.AnnData:
        """Load a specific baseline from adata.uns.
        
        Args:
            baseline_key: Key of the baseline in adata.uns.
            baseline_type: Type of baseline ('control', 'dataset_mean', 'linear', 'ground_truth', 'additive').
            split_name: Name of the split to use.
            
        Returns:
            AnnData object with baseline predictions.
            
        Raises:
            ValueError: If baseline_key not found in adata.uns.
        """
        # Check baseline exists - NO FALLBACKS!
        if baseline_key not in self.data_manager.adata.uns:
            available_keys = [k for k in self.data_manager.adata.uns.keys() if not k.startswith('_')]
            raise ValueError(
                f"Baseline key '{baseline_key}' not found in adata.uns. "
                f"Available keys: {available_keys}"
            )
        
        baseline_data = self.data_manager.adata.uns[baseline_key]
        
        # Dispatch to correct generation method based on type
        if baseline_type in ['control', 'dataset_mean']:
            return self._generate_covariate_baseline(baseline_data, baseline_key, split_name)
        elif baseline_type in ['linear', 'ground_truth', 'technical_duplicate', 'additive', 'interpolated_duplicate']:
            return self._generate_perturbation_baseline(baseline_data, baseline_key, split_name)
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
        
    def _run_baseline(self, baseline_name: str, split_name: str) -> sc.AnnData:
        """Run a single baseline directly from adata.uns or built-in logic.
        
        Args:
            baseline_name: Name of the baseline to run.
            split_name: Name of the split to use.
            
        Returns:
            AnnData object with baseline predictions.
            
        Raises:
            ValueError: If baseline not found in adata.uns.
        """
        
        # Get test conditions
        conditions = self.data_manager.get_perturbation_conditions(split_name)
        test_conditions = conditions['test']
        
        # Load baseline from adata.uns
        if baseline_name not in self.data_manager.adata.uns:
            available_keys = [k for k in self.data_manager.adata.uns.keys() if not k.startswith('_')]
            raise ValueError(f"Baseline '{baseline_name}' not found in adata.uns. Available: {available_keys}")
        
        baseline_data = self.data_manager.adata.uns[baseline_name]
        
        # Detect format and generate predictions
        predictions = self._generate_predictions_from_baseline_data(
            baseline_data, baseline_name, test_conditions, split_name
        )
        
        return predictions
    
    def _generate_predictions_from_baseline_data(self, baseline_data: pd.DataFrame, baseline_name: str, 
                                               test_conditions: List[str], split_name: str) -> sc.AnnData:
        """Generate predictions from baseline data based on its structure.
        
        Args:
            baseline_data: Baseline data from adata.uns.
            baseline_name: Name of the baseline.
            test_conditions: List of test conditions.
            split_name: Name of the split.
            
        Returns:
            AnnData object with baseline predictions.
        """
        
        # Get covariate-condition pairs for test set
        cov_condition_pairs = self.data_manager.get_covariate_condition_pairs(split_name, 'test')
        
        expressions: Dict[str, np.ndarray] = {}
        covariate_info: Dict[str, Tuple[str, str]] = {}
        
        # Case 1 DataFrame with rows and columns - distinguish by index content
        if hasattr(baseline_data, 'index') and hasattr(baseline_data, 'columns'):
            
            # Check if index contains covariate_perturbation pairs (has underscores)
            sample_index = str(baseline_data.index[0]) if len(baseline_data.index) > 0 else ""
            if '_' in sample_index and any('_' in str(idx) for idx in baseline_data.index[:5]):
                # Case 2: Linear/Technical duplicate baseline (covariate_perturbation as rows)
                log.info(f"  Detected covariate-perturbation baseline format for {baseline_name}")
                
                for covariate_value, condition in cov_condition_pairs:
                    if condition in test_conditions:
                        cov_pert_key = f"{covariate_value}_{condition}"
                        pair_key = cov_pert_key
                        
                        if cov_pert_key in baseline_data.index:
                            expressions[pair_key] = baseline_data.loc[cov_pert_key].values
                            covariate_info[pair_key] = (covariate_value, condition)
                        else:
                            raise ValueError(
                                f"Missing key '{cov_pert_key}' in {baseline_name}. "
                                f"Available keys: {list(baseline_data.index[:10])}..."
                            )
            else:
                # Case 3: Control baseline (covariates as rows, genes as columns)  
                log.info(f"  Detected covariate baseline format for {baseline_name}")
                
                for covariate_value, condition in cov_condition_pairs:
                    if condition in test_conditions:
                        pair_key = f"{covariate_value}_{condition}"
                        
                        if covariate_value in baseline_data.index:
                            expressions[pair_key] = baseline_data.loc[covariate_value].values
                            covariate_info[pair_key] = (covariate_value, condition)
                        else:
                            raise ValueError(
                                f"Missing covariate '{covariate_value}' in {baseline_name}. "
                                f"Available covariates: {list(baseline_data.index)}"
                            )
        
        else:
            raise ValueError(f"Unrecognized baseline data format for '{baseline_name}'. "
                           f"Expected DataFrame with specific structure.")
        
        # Ensure we have predictions for all test covariate-condition pairs - NO FALLBACKS!
        for covariate_value, condition in cov_condition_pairs:
            if condition in test_conditions:
                pair_key = f"{covariate_value}_{condition}"
                if pair_key not in expressions:
                    raise ValueError(
                        f"No prediction found for '{pair_key}' in baseline. "
                        f"This baseline is incomplete."
                    )
        
        # Create predictions AnnData using built-in helper
        return self._create_predictions_adata(expressions, covariate_info)
    
    def _create_predictions_adata(self, expressions: Dict[str, np.ndarray], 
                                  covariate_info: Dict[str, Tuple[str, str]]) -> sc.AnnData:
        """Helper to create AnnData from expressions dict.
        
        Args:
            expressions: Dictionary mapping pair keys to expression arrays.
            covariate_info: Dictionary mapping pair keys to (covariate, condition) tuples.
            
        Returns:
            AnnData object with predictions and metadata.
        """
        # Stack all expression arrays
        expression_matrix = np.vstack(list(expressions.values()))
        
        # Create obs dataframe with covariate and condition information
        pair_keys = list(expressions.keys())
        covariates: List[str] = []
        conditions: List[str] = []
        
        for pair_key in pair_keys:
            covariate_value, condition = covariate_info[pair_key]
            covariates.append(covariate_value)
            conditions.append(condition)
        
        obs_df = pd.DataFrame({
            'covariate': covariates,
            'condition': conditions,
            'pair_key': pair_keys
        })
        
        # Create AnnData
        adata = sc.AnnData(X=expression_matrix, obs=obs_df)
        adata.var_names = self.data_manager.adata.var_names
        
        return adata
    
    def _generate_covariate_baseline(self, baseline_data: pd.DataFrame, 
                                    baseline_key: str, split_name: str) -> sc.AnnData:
        """Generate predictions from covariate-based baseline (control, dataset_mean).
        
        These baselines have one row per covariate that gets replicated for all perturbations.
        
        Args:
            baseline_data: Baseline data from adata.uns with covariates as rows.
            baseline_key: Name of the baseline.
            split_name: Name of the split.
            
        Returns:
            AnnData object with baseline predictions.
            
        Raises:
            ValueError: If required covariates are missing.
        """
        # Get test conditions and covariate-condition pairs
        conditions = self.data_manager.get_perturbation_conditions(split_name)
        test_conditions = conditions['test']
        cov_condition_pairs = self.data_manager.get_covariate_condition_pairs(split_name, 'test')
        
        expressions: Dict[str, np.ndarray] = {}
        covariate_info: Dict[str, Tuple[str, str]] = {}
        
        # For covariate-based baselines, use the same prediction for all perturbations within a covariate
        for covariate_value, condition in cov_condition_pairs:
            if condition in test_conditions:
                pair_key = f"{covariate_value}_{condition}"
                
                if covariate_value not in baseline_data.index:
                    raise ValueError(
                        f"Covariate '{covariate_value}' not found in {baseline_key}. "
                        f"Available covariates: {list(baseline_data.index)}"
                    )
                
                # Use the same covariate baseline for all perturbations
                expressions[pair_key] = baseline_data.loc[covariate_value].values
                covariate_info[pair_key] = (covariate_value, condition)
        
        return self._create_predictions_adata(expressions, covariate_info)
    
    def _generate_perturbation_baseline(self, baseline_data: pd.DataFrame,
                                       baseline_key: str, split_name: str) -> sc.AnnData:
        """Generate predictions from perturbation-based baseline (linear, ground_truth, additive).
        
        Args:
            baseline_data: Baseline data from adata.uns with covariate_perturbation as rows.
            baseline_key: Name of the baseline.
            split_name: Name of the split.
            
        Returns:
            AnnData object with baseline predictions.
            
        Raises:
            ValueError: If required perturbations are missing for critical baselines.
        """
        # Get test conditions and covariate-condition pairs
        conditions = self.data_manager.get_perturbation_conditions(split_name)
        test_conditions = conditions['test']
        cov_condition_pairs = self.data_manager.get_covariate_condition_pairs(split_name, 'test')
        
        expressions: Dict[str, np.ndarray] = {}
        covariate_info: Dict[str, Tuple[str, str]] = {}
        
        # Check if this is an additive baseline (which doesn't include control)
        is_additive = 'additive' in baseline_key.lower()
        for covariate_value, condition in cov_condition_pairs:
            if condition in test_conditions:
                cov_pert_key = f"{covariate_value}_{condition}"
                pair_key = cov_pert_key
                
                
                if cov_pert_key not in baseline_data.index:
                    log.warning(
                        f"Perturbation key '{cov_pert_key}' not found in {baseline_key}"
                    )
                    continue
                
                expressions[pair_key] = baseline_data.loc[cov_pert_key].values
                covariate_info[pair_key] = (covariate_value, condition)
        
        if not expressions:
            raise ValueError(f"No valid predictions generated from {baseline_key}")
        
        return self._create_predictions_adata(expressions, covariate_info) 