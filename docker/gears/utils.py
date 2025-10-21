"""
Utility functions for GEARS wrapper.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging

log = logging.getLogger(__name__)


def validate_config(config: Dict) -> bool:
    """Validate GEARS configuration."""
    
    required_keys = {
        'train': ['mode', 'data_path', 'split_name', 'covariate_key', 'output_dir'],
        'predict': ['mode', 'data_path', 'model_path', 'test_conditions', 'output_path']
    }
    
    mode = config.get('mode')
    if mode not in required_keys:
        log.error(f"Invalid mode: {mode}")
        return False
    
    for key in required_keys[mode]:
        if key not in config:
            log.error(f"Missing required key: {key}")
            return False
    
    return True


def parse_gene_combinations(conditions: List[str]) -> List[List[str]]:
    """Parse condition strings into gene combination lists."""
    
    gene_combinations = []
    for condition in conditions:
        if condition == 'control':
            continue  # Skip control
        elif '+' not in condition:
            # Single gene
            gene_combinations.append([condition])
        else:
            # Gene combination
            genes = condition.split('+')
            # Remove 'ctrl' if present
            genes = [g for g in genes if g != 'ctrl']
            if genes:
                gene_combinations.append(genes)
    
    return gene_combinations


def filter_by_gene_availability(conditions: List[str], available_genes: List[str]) -> List[str]:
    """Filter conditions by gene availability in GEARS."""
    
    filtered_conditions = []
    
    for condition in conditions:
        if condition == 'control':
            filtered_conditions.append(condition)
            continue
            
        genes = parse_gene_combinations([condition])[0] if condition != 'control' else []
        
        if all(gene in available_genes for gene in genes):
            filtered_conditions.append(condition)
        else:
            log.warning(f"Skipping condition {condition}: genes not available in GEARS")
    
    return filtered_conditions


def create_condition_mapping(cellsimbench_conditions: List[str]) -> Dict[str, str]:
    """Create mapping from CellSimBench to GEARS condition format."""
    
    mapping = {}
    
    for condition in cellsimbench_conditions:
        if condition == 'control':
            mapping[condition] = 'ctrl'
        elif '+' not in condition:
            # Single perturbation
            mapping[condition] = f"{condition}+ctrl"
        else:
            # Already in combo format
            mapping[condition] = condition
    
    return mapping


def log_data_statistics(adata, split_conditions: Dict[str, List[str]]):
    """Log dataset and split statistics."""
    
    log.info(f"Dataset shape: {adata.shape}")
    log.info(f"Number of conditions: {len(adata.obs['condition'].unique())}")
    
    for split_name, conditions in split_conditions.items():
        log.info(f"{split_name} split: {len(conditions)} conditions")
        
        # Count cells per split
        split_cells = adata[adata.obs['condition'].isin(conditions)]
        log.info(f"{split_name} cells: {len(split_cells)}")


def validate_predictions(predictions: Dict, expected_conditions: List[str]) -> bool:
    """Validate that predictions cover expected conditions."""
    
    prediction_keys = set(predictions.keys())
    
    # Convert expected conditions to GEARS format
    expected_keys = set()
    for condition in expected_conditions:
        if condition != 'control':
            genes = parse_gene_combinations([condition])
            if genes:
                expected_keys.add('_'.join(genes[0]))
    
    missing_keys = expected_keys - prediction_keys
    if missing_keys:
        log.warning(f"Missing predictions for: {missing_keys}")
        return False
    
    log.info(f"Generated predictions for {len(prediction_keys)} conditions")
    return True 