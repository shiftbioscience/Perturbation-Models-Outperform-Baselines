"""
Shared utilities for calculating input hashes.

Provides hash-based caching mechanisms for training and inference to avoid
redundant computation.
"""

import json
import hashlib
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)


def calculate_input_hash(dataset_config: DictConfig, model_config: DictConfig, 
                        training_config: DictConfig) -> str:
    """Calculate a hash of all training inputs to detect changes.
    
    Creates a unique hash based on dataset, model, and training configurations
    to enable cache-based training checkpoints.
    
    Args:
        dataset_config: Dataset configuration from Hydra.
        model_config: Model configuration from Hydra.
        training_config: Training configuration from Hydra.
        
    Returns:
        MD5 hash string of all relevant training inputs.
    """
    
    # Collect all inputs that should trigger retraining if changed
    hash_inputs = {
        # Dataset configuration
        'dataset_config': OmegaConf.to_object(dataset_config),
        
        # Model configuration
        'model_config': OmegaConf.to_object(model_config),
        
        # Training configuration  
        'training_config': OmegaConf.to_object(training_config),
        
        # Data file modification time (to detect dataset changes)
        'data_mtime': None,
    }
    
    # Get data file modification time
    try:
        data_path = Path(dataset_config.data_path)
        if data_path.exists():
            hash_inputs['data_mtime'] = data_path.stat().st_mtime
    except Exception as e:
        log.warning(f"Could not get data file modification time: {e}")
    
    # Convert to JSON string for consistent hashing
    hash_string = json.dumps(hash_inputs, sort_keys=True, default=str)
    
    # Calculate MD5 hash
    return hashlib.md5(hash_string.encode()).hexdigest()


def get_model_path_for_config(dataset_config: DictConfig, model_config: DictConfig, 
                              training_config: DictConfig) -> Path:
    """Get the model path for a given configuration using hash logic.
    
    Determines the unique directory path where a model with the given
    configuration should be stored.
    
    Args:
        dataset_config: Dataset configuration from Hydra.
        model_config: Model configuration from Hydra.
        training_config: Training configuration from Hydra.
        
    Returns:
        Path to the model directory based on configuration hash.
    """
    input_hash = calculate_input_hash(dataset_config, model_config, training_config)
    base_output_dir = Path(training_config.output_dir)
    return base_output_dir / input_hash[:12]  # Use first 12 chars of hash


def calculate_inference_hash(pred_config: Dict[str, Any], training_checkpoint_path: Path) -> str:
    """Calculate hash for inference caching based on prediction config.
    
    Creates a unique hash for inference results to enable caching of predictions.
    
    Args:
        pred_config: Complete prediction configuration dictionary.
        training_checkpoint_path: Path to training_checkpoint.json file.
        
    Returns:
        MD5 hash string for inference caching.
        
    Raises:
        RuntimeError: If training was not completed.
    """
    # Load training checkpoint - let it fail if not found
    with open(training_checkpoint_path, 'r') as f:
        training_checkpoint = json.load(f)
    
    # Require training to be completed
    if not training_checkpoint['training_completed']:
        raise RuntimeError("Cannot generate inference hash - training was not completed")
    
    # Create hash inputs
    inference_hash_inputs: Dict[str, Any] = {
        'training_input_hash': training_checkpoint['input_hash'],
        'training_timestamp': training_checkpoint['timestamp'],
        'prediction_config': json.dumps(pred_config, sort_keys=True),
        'data_mtime': Path(pred_config['data_path']).stat().st_mtime
    }
    
    # Convert to JSON string and hash
    hash_string = json.dumps(inference_hash_inputs, sort_keys=True, default=str)
    return hashlib.md5(hash_string.encode()).hexdigest() 