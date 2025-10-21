"""
Abstract base classes for CellSimBench models.

Provides base classes that define the interface for all models integrated
into the CellSimBench framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import scanpy as sc
import pandas as pd
import numpy as np
from ..core.data_manager import DataManager


class BaseModel(ABC):
    """Abstract base class for all CellSimBench models.
    
    Defines the interface that all models must implement for integration
    with the benchmarking framework.
    
    Attributes:
        config: Model configuration dictionary.
        name: Model name from configuration.
    """
    
    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialize BaseModel with configuration.
        
        Args:
            model_config: Dictionary containing model configuration.
        """
        self.config = model_config
        self.name = model_config['name']
        
    @abstractmethod
    def predict(
        self,
        data_manager: DataManager,
        test_conditions: List[str],
        split_name: str,
        **kwargs: Any
    ) -> sc.AnnData:
        """Generate predictions for test conditions.
        
        Args:
            data_manager: DataManager instance for accessing data.
            test_conditions: List of conditions to predict.
            split_name: Name of the split being used.
            **kwargs: Additional hyperparameters.
            
        Returns:
            AnnData with predictions where obs['condition'] contains condition
            labels and X contains predicted expression values.
        """
        pass


class BuiltinModel(BaseModel):
    """Base class for built-in models that run locally.
    
    Provides helper methods for creating prediction AnnData objects
    with proper formatting for the benchmarking framework.
    """
    
    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialize BuiltinModel with configuration.
        
        Args:
            model_config: Dictionary containing model configuration.
        """
        super().__init__(model_config)
        
    def create_predictions_adata(
        self,
        expressions: Dict[str, np.ndarray],
        var_names: List[str]
    ) -> sc.AnnData:
        """Helper method to create AnnData object from predictions.
        
        Args:
            expressions: Dictionary mapping conditions to expression arrays.
            var_names: List of gene names.
            
        Returns:
            AnnData with predictions formatted for benchmarking.
        """
        # Stack all expression arrays - each row is one condition's prediction
        expression_matrix = np.vstack(list(expressions.values()))
        
        # Create obs dataframe with one row per condition
        conditions = list(expressions.keys())
        obs_df = pd.DataFrame({
            'condition': conditions
        })
        
        # Create AnnData
        adata = sc.AnnData(
            X=expression_matrix,
            obs=obs_df
        )
        adata.var_names = var_names
        
        return adata
    
    def create_predictions_adata_with_covariates(
        self,
        expressions: Dict[str, np.ndarray],
        covariate_info: Dict[str, Tuple[str, str]],
        var_names: List[str]
    ) -> sc.AnnData:
        """Create AnnData object from predictions with covariate information.
        
        Args:
            expressions: Dictionary mapping pair_key to expression arrays.
            covariate_info: Dictionary mapping pair_key to (covariate, condition) tuples.
            var_names: List of gene names.
            
        Returns:
            AnnData with predictions including covariate metadata.
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
        adata = sc.AnnData(
            X=expression_matrix,
            obs=obs_df
        )
        adata.var_names = var_names
        
        return adata 