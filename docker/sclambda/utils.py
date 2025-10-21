"""
Utility functions for scLambda wrapper.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from pathlib import Path
import logging

log = logging.getLogger(__name__)


def compute_correlation(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient between two arrays.
    
    Args:
        x1: First array
        x2: Second array
        
    Returns:
        Pearson correlation coefficient
    """
    try:
        corr = np.corrcoef(x1, x2)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0


def validate_gene_embeddings(gene_embeddings: Dict[str, np.ndarray], 
                           expected_genes: List[str]) -> bool:
    """
    Validate that gene embeddings contain all expected genes.
    
    Args:
        gene_embeddings: Dictionary of gene embeddings
        expected_genes: List of expected gene names
        
    Returns:
        True if all genes are present, False otherwise
    """
    missing_genes = set(expected_genes) - set(gene_embeddings.keys())
    if missing_genes:
        log.warning(f"Missing embeddings for genes: {missing_genes}")
        return False
    return True


def create_gene_embedding_summary(gene_embeddings: Dict[str, np.ndarray]) -> Dict:
    """
    Create a summary of gene embeddings for logging/debugging.
    
    Args:
        gene_embeddings: Dictionary of gene embeddings
        
    Returns:
        Summary dictionary with statistics
    """
    if not gene_embeddings:
        return {}
    
    embedding_arrays = list(gene_embeddings.values())
    stacked_embeddings = np.stack(embedding_arrays)
    
    return {
        'n_genes': len(gene_embeddings),
        'embedding_dim': embedding_arrays[0].shape[0],
        'mean_norm': np.mean([np.linalg.norm(emb) for emb in embedding_arrays]),
        'std_norm': np.std([np.linalg.norm(emb) for emb in embedding_arrays]),
        'mean_embedding': np.mean(stacked_embeddings, axis=0).tolist()[:5],  # First 5 dims only
        'genes_with_zero_embeddings': [gene for gene, emb in gene_embeddings.items() 
                                      if np.allclose(emb, 0)]
    }


def safe_tensor_to_numpy(tensor):
    """
    Safely convert PyTorch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor or numpy array
        
    Returns:
        Numpy array
    """
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().numpy()
    elif hasattr(tensor, 'numpy'):
        return tensor.numpy()
    else:
        return np.array(tensor)


def check_data_consistency(adata, split_name: str) -> Dict[str, int]:
    """
    Check data consistency for scLambda training.
    
    Args:
        adata: AnnData object
        split_name: Name of split column
        
    Returns:
        Dictionary with data statistics
    """
    stats = {
        'total_cells': adata.shape[0],
        'total_genes': adata.shape[1],
        'n_conditions': len(adata.obs['condition'].unique()),
        'has_ctrl': 'ctrl' in adata.obs['condition'].values,
        'split_coverage': {}
    }
    
    if split_name in adata.obs.columns:
        for split in ['train', 'val', 'test']:
            n_cells = (adata.obs[split_name] == split).sum()
            n_conditions = len(adata.obs[adata.obs[split_name] == split]['condition'].unique())
            stats['split_coverage'][split] = {
                'n_cells': n_cells,
                'n_conditions': n_conditions
            }
    
    return stats


def estimate_memory_usage(adata, batch_size: int, embedding_dim: int) -> Dict[str, float]:
    """
    Estimate memory usage for scLambda training.
    
    Args:
        adata: AnnData object
        batch_size: Training batch size
        embedding_dim: Gene embedding dimensions
        
    Returns:
        Dictionary with memory estimates in GB
    """
    n_genes = adata.shape[1]
    n_cells = adata.shape[0]
    
    # Estimate memory usage (in bytes)
    expression_data = n_cells * n_genes * 4  # float32
    embeddings = n_cells * embedding_dim * 4  # float32
    batch_memory = batch_size * (n_genes + embedding_dim) * 4 * 2  # input + gradient
    model_params = (n_genes + embedding_dim) * 512 * 4 * 4  # rough estimate
    
    total_bytes = expression_data + embeddings + batch_memory + model_params
    
    return {
        'expression_data_gb': expression_data / (1024**3),
        'embeddings_gb': embeddings / (1024**3),
        'batch_memory_gb': batch_memory / (1024**3),
        'model_params_gb': model_params / (1024**3),
        'total_estimated_gb': total_bytes / (1024**3)
    } 