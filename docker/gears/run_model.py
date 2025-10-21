#!/usr/bin/env python3
"""
Main entry point for GEARS Docker container.
Supports both training and prediction modes.
"""

import sys
import json
import logging
from pathlib import Path
from gears_wrapper import GEARSWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def main():
    """Main entry point for GEARS container."""
    
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
    except Exception as e:
        log.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Validate configuration mode matches
    if config.get('mode') != mode:
        log.error(f"Configuration mode '{config.get('mode')}' does not match command mode '{mode}'")
        sys.exit(1)
    
    # Initialize GEARS wrapper
    try:
        wrapper = GEARSWrapper(config)
        log.info(f"Initialized GEARS wrapper for mode: {mode}")
    except Exception as e:
        log.error(f"Failed to initialize GEARS wrapper: {e}")
        sys.exit(1)
    
    # Execute the requested operation
    try:
        if mode == 'train':
            log.info("Starting GEARS training...")
            wrapper.train()
            log.info("Training completed successfully")
        elif mode == 'predict':
            log.info("Starting GEARS prediction...")
            wrapper.predict()
            log.info("Prediction completed successfully")
    except Exception as e:
        log.error(f"Operation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 