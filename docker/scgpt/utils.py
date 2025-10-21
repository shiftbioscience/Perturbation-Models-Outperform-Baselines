"""
Utility functions for scGPT Docker container.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        log.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        log.error(f"Failed to load configuration: {e}")
        raise


def validate_config(config: Dict[str, Any], mode: str) -> bool:
    """Validate configuration for the given mode."""
    required_keys = ['mode', 'data_path', 'covariate_key', 'hyperparameters']
    
    if mode == 'train':
        required_keys.extend(['output_dir', 'checkpoint_dir', 'train_conditions', 'val_conditions', 'test_conditions'])
    elif mode == 'predict':
        required_keys.extend(['model_path', 'test_conditions', 'output_path'])
    
    for key in required_keys:
        if key not in config:
            log.error(f"Missing required configuration key: {key}")
            return False
    
    # Validate mode matches
    if config.get('mode') != mode:
        log.error(f"Configuration mode '{config.get('mode')}' does not match expected mode '{mode}'")
        return False
    
    # Validate paths exist
    data_path = Path(config['data_path'])
    if not data_path.exists():
        log.error(f"Data file not found: {data_path}")
        return False
    
    if mode == 'predict':
        model_path = Path(config['model_path'])
        if not model_path.exists():
            log.error(f"Model path not found: {model_path}")
            return False
    
    log.info(f"Configuration validation passed for mode: {mode}")
    return True


def check_gpu_availability():
    """Check if GPU is available for training."""
    import torch
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        log.info(f"GPU available: {device_name} (device {current_device}/{gpu_count})")
        return True
    else:
        log.warning("No GPU available, using CPU")
        return False


def print_system_info():
    """Print system information for debugging."""
    import torch
    import sys
    import platform
    
    log.info("=== System Information ===")
    log.info(f"Python version: {sys.version}")
    log.info(f"Platform: {platform.platform()}")
    log.info(f"PyTorch version: {torch.__version__}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        log.info(f"CUDA version: {torch.version.cuda}")
        log.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    log.info("==========================") 