"""
Training orchestration for CellSimBench framework.

Manages model training independently of benchmarking with hash-based checkpointing.
"""

import logging
import json
import tempfile
import hashlib
import pandas as pd
import os
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
from omegaconf import DictConfig, OmegaConf

from .data_manager import DataManager
from .docker_runner import DockerRunner
from .gpu_utils import get_available_gpus, calculate_gpu_assignment
from ..utils.hash_utils import calculate_input_hash, get_model_path_for_config

log = logging.getLogger(__name__)


class TrainingRunner:
    """Orchestrates model training independently of benchmarking.
    
    This class manages the training process for models, including data preparation,
    Docker container execution, and checkpoint management using hash-based paths.
    
    Attributes:
        config: Hydra configuration for training.
        data_manager: DataManager instance for data handling.
        docker_runner: DockerRunner instance for container execution.
        
    Example:
        >>> runner = TrainingRunner(config)
        >>> model_path = runner.train_model()
    """
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize TrainingRunner with configuration.
        
        Args:
            config: Hydra configuration containing model, dataset, and training settings.
        """
        self.config = config
        self.data_manager: Optional[DataManager] = None
        self.docker_runner = DockerRunner()
    
    def train_model(self) -> Union[Path, List[Path]]:
        """Train models on folds with optional parallelism.
        
        Creates unique directories based on configuration hash and trains
        models using Docker container execution for each fold.
        
        Returns:
            Path to single trained model or list of Paths for multiple models.
            
        Raises:
            FileNotFoundError: If dataset file not found.
            RuntimeError: If Docker not available for docker-type models.
            ValueError: If unknown model type.
        """
        from omegaconf import open_dict
        
        # Check data path exists
        data_path = Path(self.config.dataset.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        # Determine which folds to train on
        fold_indices = self._determine_fold_indices()
        
        # Check if we should use parallel training
        should_use_parallel = self._should_use_parallel_training(fold_indices)
        
        if should_use_parallel:
            log.info("🚀 Using parallel fold training")
            return self._train_parallel(fold_indices)
        else:
            log.info("🔄 Using sequential fold training")
            return self._train_sequential(fold_indices)
    
    def _determine_fold_indices(self) -> List[int]:
        """Determine which fold indices to train on."""
        fold_indices = []
        
        if hasattr(self.config.dataset, 'split'):
            # Explicit split override - find which fold or use as-is
            split_name = self.config.dataset.split
            if hasattr(self.config.dataset, 'folds'):
                # Check if this is a fold split
                for i, fold in enumerate(self.config.dataset.folds):
                    if fold.split == split_name:
                        fold_indices = [i]
                        break
            
            if not fold_indices:
                # Not a fold split - train single model (backward compatibility)
                # This will be handled by _train_single_model in sequential mode
                fold_indices = []
                
        elif hasattr(self.config, 'fold_indices') and self.config.fold_indices is not None:
            # Explicit fold indices provided
            fold_indices = self.config.fold_indices
        elif hasattr(self.config.dataset, 'folds'):
            # Train on ALL folds by default
            fold_indices = list(range(len(self.config.dataset.folds)))
        
        return fold_indices
    
    def _should_use_parallel_training(self, fold_indices: List[int]) -> bool:
        """Determine whether to use parallel training based on config and conditions."""
        
        # Check config setting
        parallel_enabled = getattr(self.config.execution, 'parallel_folds', True)
        
        if not parallel_enabled:
            log.info("Parallel training disabled in execution config")
            return False
        
        # Only makes sense with multiple folds
        if len(fold_indices) <= 1:
            log.info("Single or no fold training - using sequential")
            return False
        
        # Check if GPUs are available
        available_gpus = get_available_gpus()
        if not available_gpus:
            log.info("No GPUs available - falling back to sequential training")
            return False
        
        # Check Docker availability
        if self.docker_runner.docker_client is None:
            log.info("Docker not available - cannot use parallel training")
            return False
        
        log.info(f"Parallel training conditions met: {len(fold_indices)} folds, {len(available_gpus)} GPUs")
        return True
    
    def _train_parallel(self, fold_indices: List[int]) -> List[Path]:
        """Execute parallel fold training."""
        from concurrent.futures import ThreadPoolExecutor
        
        # GPU assignment
        available_gpus = get_available_gpus()
        gpu_assignment = calculate_gpu_assignment(fold_indices, available_gpus)
        
        # Use as many workers as folds (GPUs will be assigned round-robin)
        max_workers = len(fold_indices)
        
        # Limit parallelism if requested by user
        max_parallel = getattr(self.config.execution, 'max_parallel_folds', None)
        if max_parallel is not None:
            max_workers = min(max_workers, max_parallel)
        
        log.info(f"Training {len(fold_indices)} folds in parallel using {max_workers} workers")
        
        # Parallel training execution
        trained_models = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fold training jobs
            future_to_fold = {}
            for fold_idx in fold_indices:
                gpu_id = gpu_assignment[fold_idx]
                future = executor.submit(self._train_single_fold_with_gpu, fold_idx, gpu_id)
                future_to_fold[future] = fold_idx
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_fold):
                fold_idx = future_to_fold[future]
                try:
                    model_path = future.result()  # No timeout
                    trained_models[fold_idx] = model_path
                    log.info(f"✅ Fold {fold_idx} training completed: {model_path}")
                except Exception as e:
                    log.error(f"❌ Fold {fold_idx} training failed: {e}")
                    raise RuntimeError(f"Training failed for fold {fold_idx}: {e}")
        
        # Return results in original order
        result_list = [trained_models[fold_idx] for fold_idx in fold_indices]
        return result_list
    
    def _train_sequential(self, fold_indices: List[int]) -> Union[Path, List[Path]]:
        """Execute sequential fold training (existing logic)."""
        from omegaconf import open_dict
        
        # Handle backward compatibility case with no folds
        if not fold_indices:
            if hasattr(self.config.dataset, 'split'):
                return self._train_single_model(self.config.dataset.split)
            else:
                return self._train_single_model('split')
        
        # Train on each fold sequentially
        trained_models = []
        for fold_idx in fold_indices:
            fold_config = self.config.dataset.folds[fold_idx]
            split_name = fold_config.split
            log.info(f"\n{'='*60}")
            log.info(f"Training fold {fold_idx} sequentially (split: {split_name})")
            log.info(f"{'='*60}")
            
            trained_path = self._train_single_fold(fold_idx)
            trained_models.append(trained_path)
        
        # Return single path if only one model, otherwise list
        return trained_models[0] if len(trained_models) == 1 else trained_models
    
    def _train_single_fold_with_gpu(self, fold_idx: int, gpu_id: int) -> Path:
        """Train a single fold with specific GPU assignment."""
        fold_config = self.config.dataset.folds[fold_idx]
        split_name = fold_config.split
        
        log.info(f"Training fold {fold_idx} on GPU {gpu_id} (split: {split_name})")
        
        return self._train_single_fold(fold_idx, gpu_id)
    
    def _train_single_fold(self, fold_idx: int, gpu_id: Optional[int] = None) -> Path:
        """Train a single fold (shared logic for both parallel and sequential)."""
        from omegaconf import open_dict
        
        fold_config = self.config.dataset.folds[fold_idx]
        split_name = fold_config.split
        
        # Create dataset config with this specific split
        fold_dataset_config = OmegaConf.create(OmegaConf.to_object(self.config.dataset))
        with open_dict(fold_dataset_config):
            fold_dataset_config.split = split_name
        
        # Calculate model path with fold-specific config
        model_path = get_model_path_for_config(fold_dataset_config, self.config.model, self.config.training)
        input_hash = calculate_input_hash(fold_dataset_config, self.config.model, self.config.training)
        
        log.info(f"Model for fold {fold_idx} will be saved to: {model_path}")
        log.info(f"Parameter hash: {input_hash}")
        
        # Check if already trained and not forcing retrain
        force_retrain = True
        if not force_retrain and model_path.exists() and (model_path / "model.pth").exists():
            log.info(f"Model for fold {fold_idx} already trained at: {model_path}")
            return model_path
        elif force_retrain and model_path.exists() and (model_path / "model.pth").exists():
            log.info(f"🔄 Retraining fold {fold_idx} (force_retrain=True)")
        else:
            log.info(f"🔨 Training fold {fold_idx} for the first time")
        
        # Create output directory
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data manager for this fold
        self.data_manager = DataManager(OmegaConf.to_object(fold_dataset_config))
        
        # Train the model
        model_type = self.config.model.get('type', 'docker')
        if model_type == 'docker':
            if self.docker_runner.docker_client is None:
                raise RuntimeError("Docker is not available but model type is 'docker'")
            return self._train_docker_model(model_path, split_name, gpu_id)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_single_model(self, split_name: str) -> Path:
        """Train a single model on a specific split.
        
        Args:
            split_name: Name of the split to train on.
            
        Returns:
            Path to the trained model directory.
        """
        # Calculate model path
        model_path = get_model_path_for_config(self.config.dataset, self.config.model, self.config.training)
        input_hash = calculate_input_hash(self.config.dataset, self.config.model, self.config.training)
        
        log.info(f"Model will be saved to: {model_path}")
        log.info(f"Parameter hash: {input_hash}")
        log.info(f"Using split: {split_name}")
        
        # Check if already trained and not forcing retrain
        force_retrain = True
        if not force_retrain and model_path.exists() and (model_path / "model.pth").exists():
            log.info(f"Model already trained at: {model_path}")
            return model_path
        elif force_retrain and model_path.exists() and (model_path / "model.pth").exists():
            log.info(f"🔄 Retraining model (force_retrain=True)")
        else:
            log.info(f"🔨 Training model for the first time")
        
        # Create output directory
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data manager
        self.data_manager = DataManager(OmegaConf.to_object(self.config.dataset))
        
        # Train the model
        model_type = self.config.model.get('type', 'docker')
        if model_type == 'docker':
            if self.docker_runner.docker_client is None:
                raise RuntimeError("Docker is not available but model type is 'docker'")
            return self._train_docker_model(model_path, split_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_docker_model(self, model_path: Path, split_name: str = None, gpu_id: Optional[int] = None) -> Path:
        """Execute model training in Docker container.
        
        Args:
            model_path: Path where trained model will be saved.
            split_name: Optional split name override.
            
        Returns:
            Path to the trained model directory.
            
        Raises:
            Exception: If Docker training fails.
        """
        
        # Create temporary directory for config file
        temp_path = Path(tempfile.mkdtemp())
        
        # Get split name - use provided or determine from config
        if split_name is None:
            if hasattr(self.config.dataset, 'split'):
                split_name = self.config.dataset.split
            elif hasattr(self.config.dataset, 'folds') and hasattr(self.config.dataset, 'default_fold'):
                default_fold_idx = self.config.dataset.default_fold
                split_name = self.config.dataset.folds[default_fold_idx].split
            else:
                split_name = 'split'
        
        # Get perturbation conditions for training (load only obs metadata)
        obs = self.data_manager.load_obs_only()
        conditions = self.data_manager.get_perturbation_conditions(split_name, obs)
        
        # Get the actual data file path and filename
        data_file_path = Path(self.data_manager.config['data_path']).resolve()
        data_filename = data_file_path.name
        
        log.info(f"Using data file: {data_filename}")
        
        # Create training configuration for the model
        train_config = {
            'mode': 'train',
            'data_path': f'/data/{data_filename}',
            'split_name': split_name,
            'covariate_key': self.data_manager.config['covariate_key'],
            'hyperparameters': OmegaConf.to_object(self.config.model.hyperparameters),
            'output_dir': '/model_output/',
            'checkpoint_dir': '/model_output/checkpoints/',
            'train_conditions': conditions['train'],
            'val_conditions': conditions['val'],
            'test_conditions': conditions['test']
        }
        
        # Create training config file
        config_path = temp_path / 'train_config.json'
        with open(config_path, 'w') as f:
            json.dump(train_config, f, indent=2)
            print(f"Saved training config to {config_path}")
        
        # Set up volumes - mount original data file and output directory
        volumes = {
            str(data_file_path.parent): {'bind': '/data', 'mode': 'ro'},
            str(model_path.resolve()): {'bind': '/model_output', 'mode': 'rw'},
            str(config_path.resolve()): {'bind': '/config.json', 'mode': 'ro'}
        }
        if "model_loc_local" in self.config.model.hyperparameters:
            pretrained_model_path = Path(self.config.model.hyperparameters.model_loc_local).resolve()
            volumes[str(pretrained_model_path)] = {'bind': '/pretrained_model/', 'mode': 'ro'}
        
        log.info(f"Volume mounts:")
        log.info(f"  Data directory: {data_file_path.parent} -> /data")
        log.info(f"  Model output: {model_path.resolve()} -> /model_output")
        log.info(f"  Config file: {config_path.resolve()} -> /config.json")
        if "model_loc_local" in self.config.model.hyperparameters:
            log.info(f"  Pretrained model: {pretrained_model_path} -> /pretrained_model/")
        
        # Set up environment variables for models that need them
        environment = {}
        
        # Note: GPU assignment is now handled via Docker device_requests instead of CUDA_VISIBLE_DEVICES
        import os
        
        # For scLambda and other models that need OpenAI API key
        if 'sclambda' in self.config.model.name:
            # Check if .env file path is specified in config
            env_file_path = None
            if hasattr(self.config.model, 'hyperparameters') and 'gene_embedding_config' in self.config.model.hyperparameters:
                env_file_path = self.config.model.hyperparameters.gene_embedding_config.get('env_file_path')
            
            if env_file_path:
                env_file_full_path = Path(env_file_path).resolve()
                if env_file_full_path.exists():
                    # Mount the .env file to the container
                    volumes[str(env_file_full_path)] = {'bind': '/app/.env', 'mode': 'ro'}
                    log.info(f"Mounting .env file: {env_file_full_path} -> /app/.env")
                else:
                    raise FileNotFoundError(f"scLambda requires OpenAI API key but specified .env file not found: {env_file_full_path}")
            
            # Also try to pass from host environment as fallback
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                environment['OPENAI_API_KEY'] = api_key
                log.info("OpenAI API key found in host environment and will be passed to container")
            elif not env_file_path:
                raise ValueError("scLambda requires OpenAI API key but no OPENAI_API_KEY environment variable found and no env_file_path specified in config")
        
        # Use shared Docker runner
        try:
            self.docker_runner.run_container(
                image=self.config.model.docker.image,
                command=['train', '/config.json'],
                volumes=volumes,
                docker_config=OmegaConf.to_object(self.config.model.docker),
                container_name="Model training",
                environment=environment if environment else None,
                gpu_id=gpu_id
            )
        except Exception as e:
            log.error(f"Docker training failed: {e}")
            # Mark training as failed in checkpoint
            self._save_training_checkpoint(model_path, training_completed=False)
            raise
        
        # Save training checkpoint with input hash
        self._save_training_checkpoint(model_path, training_completed=True)
        
        log.info(f"Training completed successfully. Model saved to: {model_path}")
        return model_path
    

    
    def _save_training_checkpoint(self, model_path: Path, training_completed: bool = True) -> None:
        """Save training checkpoint with input hash and completion status.
        
        Args:
            model_path: Path to the model directory.
            training_completed: Whether training completed successfully.
        """
        
        checkpoint: Dict[str, Any] = {
            'input_hash': calculate_input_hash(self.config.dataset, self.config.model, self.config.training),
            'training_completed': training_completed,
            'timestamp': pd.Timestamp.now().isoformat(),
            'config_snapshot': {
                'dataset': OmegaConf.to_object(self.config.dataset),
                'model': OmegaConf.to_object(self.config.model),
                'training': OmegaConf.to_object(self.config.training),
            }
        }
        
        checkpoint_file = model_path / 'training_checkpoint.json'
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            log.info(f"Training checkpoint saved to {checkpoint_file}")
        except Exception as e:
            log.warning(f"Failed to save training checkpoint: {e}") 