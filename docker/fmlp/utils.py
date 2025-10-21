"""
Utility functions for fMLP wrapper.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
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


def validate_fm_embeddings(adata, embedding_key: str) -> Tuple[bool, List[str]]:
    """
    Validate that foundation model embeddings exist in adata.varm.
    
    Args:
        adata: AnnData object
        embedding_key: Key for embeddings in varm
        
    Returns:
        Tuple of (success, list of genes with missing embeddings)
    """
    if embedding_key not in adata.varm:
        log.error(f"Embedding key '{embedding_key}' not found in adata.varm")
        return False, []
    
    emb_matrix = adata.varm[embedding_key].values
    gene_names = adata.var_names
    
    missing_genes = []
    for i, gene in enumerate(gene_names):
        if np.isnan(emb_matrix[i]).any():
            missing_genes.append(gene)
    
    if missing_genes:
        log.warning(f"Found {len(missing_genes)} genes with NaN embeddings")
        
    return True, missing_genes


def create_embedding_summary(adata, embedding_key: str) -> Dict:
    """
    Create a summary of foundation model embeddings for logging/debugging.
    
    Args:
        adata: AnnData object
        embedding_key: Key for embeddings in varm
        
    Returns:
        Summary dictionary with statistics
    """
    if embedding_key not in adata.varm:
        return {'error': f"Embedding key '{embedding_key}' not found"}
    
    emb_matrix = adata.varm[embedding_key].values
    valid_mask = ~np.isnan(emb_matrix).any(axis=1)
    valid_embeddings = emb_matrix[valid_mask]
    
    if len(valid_embeddings) == 0:
        return {'error': 'No valid embeddings found'}
    
    return {
        'embedding_key': embedding_key,
        'n_genes_total': len(adata.var_names),
        'n_genes_with_embeddings': valid_mask.sum(),
        'n_genes_missing': (~valid_mask).sum(),
        'embedding_dim': emb_matrix.shape[1],
        'mean_norm': np.mean([np.linalg.norm(emb) for emb in valid_embeddings]),
        'std_norm': np.std([np.linalg.norm(emb) for emb in valid_embeddings]),
        'mean_embedding': np.mean(valid_embeddings, axis=0).tolist()[:5],  # First 5 dims only
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


def check_data_consistency(adata, split_name: str, covariate_key: str) -> Dict[str, int]:
    """
    Check data consistency for fMLP training.
    
    Args:
        adata: AnnData object
        split_name: Name of split column
        covariate_key: Name of covariate column
        
    Returns:
        Dictionary with data statistics
    """
    stats = {
        'total_cells': adata.shape[0],
        'total_genes': adata.shape[1],
        'n_conditions': len(adata.obs['condition'].unique()),
        'n_covariates': len(adata.obs[covariate_key].unique()) if covariate_key in adata.obs else 0,
        'has_ctrl': 'ctrl' in adata.obs['condition'].values,
        'split_coverage': {}
    }
    
    if split_name in adata.obs.columns:
        for split in ['train', 'val', 'test']:
            split_mask = adata.obs[split_name] == split
            n_cells = split_mask.sum()
            
            if n_cells > 0:
                split_data = adata.obs[split_mask]
                n_conditions = len(split_data['condition'].unique())
                n_covariates = len(split_data[covariate_key].unique()) if covariate_key in adata.obs else 0
                
                stats['split_coverage'][split] = {
                    'n_cells': n_cells,
                    'n_conditions': n_conditions,
                    'n_covariates': n_covariates
                }
            else:
                stats['split_coverage'][split] = {
                    'n_cells': 0,
                    'n_conditions': 0,
                    'n_covariates': 0
                }
    
    return stats


def estimate_memory_usage(adata, batch_size: int, embedding_dim: int, 
                          latent_dim: int = 128, hidden_dim: int = 512) -> Dict[str, float]:
    """
    Estimate memory usage for fMLP training.
    
    Args:
        adata: AnnData object
        batch_size: Training batch size
        embedding_dim: Foundation model embedding dimensions
        latent_dim: Latent space dimensions
        hidden_dim: Hidden layer dimensions
        
    Returns:
        Dictionary with memory estimates in GB
    """
    n_genes = adata.shape[1]
    n_cells = adata.shape[0]
    
    # Estimate memory usage (in bytes)
    expression_data = n_cells * n_genes * 4  # float32
    fm_embeddings = n_genes * embedding_dim * 4  # float32 (stored in varm)
    batch_memory = batch_size * (n_genes + embedding_dim) * 4 * 2  # input + gradient
    
    # Model parameters (simplified architecture)
    encoder_params = embedding_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * latent_dim
    decoder_params = (latent_dim + 16) * hidden_dim + hidden_dim * hidden_dim + hidden_dim * n_genes  # +16 for covariate
    model_params = (encoder_params + decoder_params) * 4
    
    total_bytes = expression_data + fm_embeddings + batch_memory + model_params
    
    return {
        'expression_data_gb': expression_data / (1024**3),
        'fm_embeddings_gb': fm_embeddings / (1024**3),
        'batch_memory_gb': batch_memory / (1024**3),
        'model_params_gb': model_params / (1024**3),
        'total_estimated_gb': total_bytes / (1024**3)
    }


def get_foundation_model_info(foundation_model: str) -> Dict[str, Union[str, int]]:
    """
    Get information about foundation model embeddings.
    
    Args:
        foundation_model: Name of foundation model ('geneformer', 'esm2', 'uce')
        
    Returns:
        Dictionary with model information
    """
    fm_info = {
        'geneformer': {
            'embedding_key': 'geneformer_gene_embeddings',
            'embedding_dim': 512,
            'description': 'Geneformer pre-trained on single-cell RNA-seq data'
        },
        'esm2': {
            'embedding_key': 'esm2_mean_gene_embeddings',  # Can also use bos/sum/max variants
            'embedding_dim': 640,  # For ESM2-150M model
            'description': 'ESM2 protein language model embeddings'
        },
        'uce': {
            'embedding_key': 'uce_gene_embeddings',
            'embedding_dim': 1280,
            'description': 'Universal Cell Embeddings'
        }
    }
    
    if foundation_model not in fm_info:
        raise ValueError(f"Unknown foundation model: {foundation_model}. "
                        f"Available models: {list(fm_info.keys())}")
    
    return fm_info[foundation_model]


def format_training_summary(epoch: int, loss: float, val_corr: Optional[float] = None) -> str:
    """
    Format training summary for logging.
    
    Args:
        epoch: Current epoch
        loss: Training loss
        val_corr: Validation correlation (optional)
        
    Returns:
        Formatted string
    """
    summary = f"Epoch {epoch}: Loss = {loss:.5f}"
    if val_corr is not None:
        summary += f", Val Correlation = {val_corr:.5f}"
    return summary
