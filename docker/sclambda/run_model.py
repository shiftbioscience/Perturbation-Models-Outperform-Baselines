#!/usr/bin/env python3

"""
Main entry point for scLambda Docker container.
Supports both training and prediction modes.
"""

import sys
import json
import logging
from pathlib import Path
from sclambda_wrapper import SCLambdaWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def main():
    """Main entry point for scLambda container."""
    
    if len(sys.argv) != 3:
        print("Usage: python run_model.py {train|predict} /path/to/config.json")
        sys.exit(1)
    
    mode = sys.argv[1]  # 'train' or 'predict'
    config_path = sys.argv[2]  # '/config.json'
    
    # Validate mode
    if mode not in ['train', 'predict']:
        log.error(f"Unknown mode: {mode}. Must be 'train' or 'predict'")
        sys.exit(1)
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        log.info(f"Loaded configuration from {config_path}")
        log.info(f"Config keys: {list(config.keys())}")
        log.info(f"Config mode: {config.get('mode')}")
        if 'hyperparameters' in config:
            log.info(f"Hyperparameters keys: {list(config['hyperparameters'].keys())}")
    except Exception as e:
        log.error(f"Failed to load configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Validate configuration mode matches
    if config.get('mode') != mode:
        log.error(f"Configuration mode '{config.get('mode')}' does not match command mode '{mode}'")
        sys.exit(1)
    
    # Fix config structure for scLambda (same as local script)
    if mode == 'train' and 'hyperparameters' in config and 'gene_embedding_config' in config['hyperparameters']:
        log.info("Moving gene_embedding_config from hyperparameters to top level")
        config['gene_embedding_config'] = config['hyperparameters'].pop('gene_embedding_config')
        log.info(f"Updated config keys: {list(config.keys())}")
        log.info(f"Updated hyperparameters keys: {list(config['hyperparameters'].keys())}")
    
    # Initialize scLambda wrapper
    try:
        log.info("Creating SCLambdaWrapper...")
        wrapper = SCLambdaWrapper(config)
        log.info(f"Initialized scLambda wrapper for mode: {mode}")
    except Exception as e:
        log.error(f"Failed to initialize scLambda wrapper: {e}")
        import traceback
        log.error("Full traceback:")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the requested operation
    try:
        if mode == 'train':
            log.info("Starting scLambda training...")
            wrapper.train()
            log.info("Training completed successfully")
        elif mode == 'predict':
            log.info("Starting scLambda prediction...")
            wrapper.predict()
            log.info("Prediction completed successfully")
    except Exception as e:
        log.error(f"Operation failed: {e}")
        log.error("Full traceback:")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 