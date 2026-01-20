"""
Variance analysis helper for mode collapse detection.

Provides tools for analyzing per-gene variance across perturbations to detect
mode collapse in model predictions.
"""

import logging
import numpy as np
from typing import Dict, List, Optional
import scanpy as sc
from .data_manager import DataManager

log = logging.getLogger(__name__)


class VarianceAnalyzer:
    """Helper class for calculating per-gene variances across perturbations.
    
    Used to detect mode collapse by comparing variance patterns between models
    and ground truth.
    
    Attributes:
        data_manager: DataManager instance for accessing data.
        
    Example:
        >>> analyzer = VarianceAnalyzer(data_manager)
        >>> variances = analyzer.calculate_covariate_variances(predictions, 'donor1', 'split')
    """
    
    def __init__(self, data_manager: DataManager) -> None:
        """Initialize VarianceAnalyzer with DataManager.
        
        Args:
            data_manager: DataManager for accessing perturbation data.
        """
        self.data_manager = data_manager
    
    def calculate_covariate_variances(self, all_predictions: Dict[str, sc.AnnData], 
                                    covariate: str, split_name: str) -> Dict[str, np.ndarray]:
        """Calculate per-gene variance across perturbations for a specific covariate.
        
        Args:
            all_predictions: Dictionary mapping model names to predictions AnnData.
            covariate: Specific covariate value to analyze (e.g., donor ID).
            split_name: Name of the split being used.
            
        Returns:
            Dictionary mapping model names to per-gene variance arrays.
        """
        variance_data: Dict[str, np.ndarray] = {}
        
        # Get covariate-perturbation pairs for this covariate
        if split_name == 'aggregated_folds':
            # For aggregated results, extract pairs from predictions directly
            # Get one model's predictions to extract the pairs
            sample_predictions = next(iter(all_predictions.values()))
            obs_df = sample_predictions.obs
            # Get unique covariate-condition pairs for this specific covariate
            covariate_pairs = []
            for _, row in obs_df.iterrows():
                if row['covariate'] == covariate:
                    pair = (row['covariate'], row['condition'])
                    if pair not in covariate_pairs:
                        covariate_pairs.append(pair)
        else:
            # Use standard DataManager method for regular splits
            cov_pert_pairs = self.data_manager.get_covariate_condition_pairs(split_name, 'test')
            covariate_pairs = [(cov, pert) for cov, pert in cov_pert_pairs if cov == covariate]
        
        if not covariate_pairs:
            log.warning(f"No perturbations found for covariate {covariate}")
            return variance_data
        
        # For each model, calculate variances
        for model_name, predictions_adata in all_predictions.items():
            model_expressions = []
            
            for cov, condition in covariate_pairs:
                try:
                    # Get mask for this specific covariate-condition combination
                    mask = ((predictions_adata.obs['covariate'] == cov) & 
                        (predictions_adata.obs['condition'] == condition))
                except: 
                    mask = ((predictions_adata.obs[self.data_manager.config['covariate_key']] == cov) & 
                        (predictions_adata.obs['condition'] == condition))
                
                if mask.sum() > 0:
                    # Get mean expression for this condition
                    expr = predictions_adata[mask].X.mean(axis=0)
                    if hasattr(expr, 'A1'):  # Handle sparse matrices
                        expr = expr.A1
                    else:
                        expr = np.asarray(expr).flatten()
                    model_expressions.append(expr)
            
            # Calculate per-gene variance across perturbations
            if model_expressions:
                expression_matrix = np.array(model_expressions)  # (n_perturbations, n_genes)
                per_gene_variance = np.var(expression_matrix, axis=0)
                variance_data[model_name] = per_gene_variance
                log.debug(f"Calculated variances for {model_name} with {len(model_expressions)} perturbations")
            else:
                log.warning(f"No expressions found for model {model_name}, covariate {covariate}")
        
        return variance_data
    
    def create_variance_rank_data(self, variance_data: Dict[str, np.ndarray], 
                                top_k: int = 2048) -> Dict[str, np.ndarray]:
        """Prepare variance data for rank plotting.
        
        Sorts genes by reference model variance and returns top-k for plotting.
        
        Args:
            variance_data: Dictionary mapping model names to per-gene variances.
            top_k: Number of top genes to include in the plot.
            
        Returns:
            Dictionary with sorted variance data ready for plotting.
        """
        if not variance_data:
            return {}
        
        # Find reference model for sorting (prefer technical duplicate or ground truth)
        reference_key = None
        for key in variance_data.keys():
            if 'technical_duplicate' in key.lower() or 'ground_truth' in key.lower():
                reference_key = key
                break
        
        if reference_key is None:
            reference_key = list(variance_data.keys())[0]
        
        # Sort genes by reference variance
        reference_var = variance_data[reference_key]
        sorted_indices = np.argsort(reference_var)[::-1][:top_k]
        
        # Apply sorting to all models
        sorted_variance_data = {}
        for model_name, variances in variance_data.items():
            sorted_variance_data[model_name] = variances[sorted_indices]
        
        return sorted_variance_data 