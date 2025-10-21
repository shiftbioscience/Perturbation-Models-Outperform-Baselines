"""
Utility functions for PRESAGE wrapper.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from typing import Dict, List

def convert_perturbation_names(perturbations: List[str]) -> List[str]:
    """Convert CellSimBench perturbation names to PRESAGE format."""
    converted = []
    for pert in perturbations:
        # Convert "GENE1+GENE2" to "GENE1_GENE2"
        converted_pert = pert.replace("+", "_").replace("ctrl_iegfp", "control").replace("ctrl", "control")
        converted.append(converted_pert)
    return converted

def validate_presage_data(adata: sc.AnnData) -> bool:
    """Validate that data is in correct format for PRESAGE."""
    required_obs_columns = ['perturbation', 'nperts']
    required_var_columns = ['gene_name']
    
    for col in required_obs_columns:
        if col not in adata.obs.columns:
            raise ValueError(f"Missing required obs column: {col}")
    
    for col in required_var_columns:
        if col not in adata.var.columns:
            raise ValueError(f"Missing required var column: {col}")
    
    return True 