"""
Command-line interface for CellSimBench.

Provides the main entry points for training models and running benchmarks
through Hydra-based configuration management.
"""

import logging
import sys
from typing import Optional
import hydra
from omegaconf import DictConfig
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

log = logging.getLogger(__name__)

# Get the absolute path to the configs directory
CONFIGS_PATH = str(Path(__file__).parent / "configs")


def main() -> None:
    """Main CLI entrypoint for CellSimBench with subcommands.
    
    Provides 'train' and 'benchmark' subcommands for model training
    and evaluation respectively.
    """
    
    if len(sys.argv) < 2:
        print("Usage: cellsimbench {train|benchmark} [options]")
        print("  train     - Train models independently")
        print("  benchmark - Benchmark pre-trained models")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "train":
        # Remove 'train' from argv and call train main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        train_main()
    elif command == "benchmark":
        # Remove 'benchmark' from argv and call benchmark main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        benchmark_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, benchmark")
        sys.exit(1)


@hydra.main(version_base=None, config_path=CONFIGS_PATH, config_name="config")
def train_main(cfg: DictConfig) -> None:
    """Training entrypoint for CellSimBench.
    
    Trains a single model based on Hydra configuration.
    
    Args:
        cfg: Hydra configuration for training including model, dataset,
             and training parameters.
             
    Raises:
        Exception: If training fails for any reason.
    """
    
    try:
        from cellsimbench.core.training_runner import TrainingRunner
        
        # Add default training configuration if not present
        if not hasattr(cfg, 'training'):
            from omegaconf import OmegaConf, open_dict
            with open_dict(cfg):
                cfg.training = {
                    'output_dir': f'models/{cfg.model.name}_{cfg.dataset.name}/',
                    'save_intermediate': True
                }
        
        # Update experiment name for training
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.experiment.name = f"train_{cfg.model.name}_{cfg.dataset.name}"
            cfg.experiment.description = f"Train {cfg.model.name} on {cfg.dataset.name} dataset"
        
        # Create and run training
        runner = TrainingRunner(cfg)
        model_path = runner.train_model()
        
        log.info(f"Training completed. Model saved to: {model_path}")
        
    except Exception as e:
        log.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@hydra.main(version_base=None, config_path=CONFIGS_PATH, config_name="config")
def benchmark_main(cfg: DictConfig) -> float:
    """Benchmarking entrypoint for CellSimBench.
    
    Runs benchmarks on pre-trained models and generates evaluation metrics.
    
    Args:
        cfg: Hydra configuration for benchmarking including model(s),
             dataset, and output settings.
        
    Returns:
        Primary metric value (0.0 on success, -inf on failure) for
        Hydra optimization compatibility.
    """
    
    try:
        from cellsimbench.core.benchmark import BenchmarkRunner
        from cellsimbench.utils.hash_utils import get_model_path_for_config
        from omegaconf import OmegaConf
        
        # Handle modelgroup configs that reference multiple models
        if hasattr(cfg, 'modelgroup'):
            # Load modelgroup config and expand to individual model configs
            from hydra import compose
            
            # Create models list from modelgroup config
            model_configs = []
            for model_name in cfg.modelgroup.models:
                # Load the individual model config
                try:
                    model_cfg = compose(config_name="config", overrides=[f"model={model_name}"])
                    model_configs.append(model_cfg.model)
                except Exception as e:
                    log.error(f"Failed to load model config '{model_name}': {e}")
                    raise
            
            from omegaconf import open_dict
            with open_dict(cfg):
                cfg.models = model_configs
                # Remove modelgroup and model configs to avoid conflicts
                # Use delattr to fully remove from config
                if 'modelgroup' in cfg:
                    delattr(cfg, 'modelgroup')
                if 'model' in cfg:
                    delattr(cfg, 'model')
        
        # Handle single model config - just ensure training config exists
        elif hasattr(cfg, 'model') and not hasattr(cfg, 'models'):
            if cfg.model.type == 'docker':
                # Add default training config if not present for hash calculation
                if not hasattr(cfg, 'training'):
                    from omegaconf import open_dict
                    with open_dict(cfg):
                        cfg.training = {
                            'output_dir': f'models/{cfg.model.name}_{cfg.dataset.name}/',
                            'save_intermediate': True
                        }
        
        # Handle models list (from modelgroup)
        if hasattr(cfg, 'models'):
            # Calculate model_path for each model that doesn't have one
            for i, model_config in enumerate(cfg.models):
                if model_config.type == 'docker' and 'model_path' not in model_config:
                    # Create individual training config for each model (same as single model approach)
                    individual_training_config = OmegaConf.create({
                        'output_dir': f'models/{model_config.name}_{cfg.dataset.name}/',
                        'save_intermediate': True
                    })
                    
                    # Calculate the model path using the full model config and individual training config
                    model_path = str(get_model_path_for_config(cfg.dataset, model_config, individual_training_config))
                    
                    # Add the calculated path to the config
                    from omegaconf import open_dict
                    with open_dict(cfg):
                        cfg.models[i]['model_path'] = model_path
            
            # Add default training config if not present (for other parts of the code)
            if not hasattr(cfg, 'training'):
                from omegaconf import open_dict
                with open_dict(cfg):
                    cfg.training = {
                        'output_dir': f'models/{cfg.dataset.name}/',
                        'save_intermediate': True
                    }
            
            # Update experiment name for multi-model benchmarking
            from omegaconf import open_dict
            with open_dict(cfg):
                model_names = [getattr(m, 'display_name', m.name) for m in cfg.models]
                cfg.experiment.name = f"benchmark_{'_'.join(model_names)}_{cfg.dataset.name}"
                cfg.experiment.description = f"Benchmark {', '.join(model_names)} on {cfg.dataset.name} dataset"
        
        # Create and run benchmark
        runner = BenchmarkRunner(cfg)
        results = runner.run_benchmark()
        
        log.info(f"Benchmark completed.")
        
        return 0.0
        
    except Exception as e:
        log.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return float('-inf')  # Return very low score on failure for optimization


if __name__ == "__main__":
    main() 