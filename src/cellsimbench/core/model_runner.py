"""
Model execution engine for CellSimBench framework.

Handles execution of models in Docker containers for prediction/inference.
"""

from pathlib import Path
from typing import Dict, Optional, List, Any
import tempfile
import json
import scanpy as sc
import warnings
import shutil
import logging
from datetime import datetime

from .docker_runner import DockerRunner
from .data_manager import DataManager

log = logging.getLogger(__name__)


class ModelRunner:
    """Executes models in Docker containers or locally.
    
    This class manages model prediction/inference using pre-trained models,
    handling Docker container execution and result caching.
    
    Attributes:
        docker_runner: DockerRunner instance for container execution.
        config: Optional configuration dictionary.
        
    Example:
        >>> runner = ModelRunner()
        >>> predictions = runner.run_model(model_config, data_manager, 'split', output_dir)
    """
    
    def __init__(self) -> None:
        """Initialize ModelRunner with Docker support."""
        self.docker_runner = DockerRunner()
        self.config: Dict[str, Any] = {}  # Can be set by BenchmarkRunner if needed
    
    def run_model(
        self,
        model_config: Dict[str, Any],
        data_manager: DataManager,
        split_name: str,
        output_dir: Path,
        gpu_id: Optional[int] = None
    ) -> Path:
        """Run prediction using pre-trained model and return path to predictions.
        
        Supports caching of inference results based on configuration hash.
        
        Args:
            model_config: Model configuration including model_path, type, and hyperparameters.
            data_manager: DataManager instance for accessing data.
            split_name: Name of the split to use for prediction.
            output_dir: Directory to save outputs.
            
        Returns:
            Path to predictions h5ad file.
            
        Raises:
            RuntimeError: If Docker is not available for docker-type models.
            ValueError: If model type is unknown or baselines_only.
        """
        model_type = model_config.get('type', 'docker')
        
        if model_type == 'docker':
            if self.docker_runner.docker_client is None:
                raise RuntimeError("Docker is not available but model type is 'docker'")
            return self._run_docker_prediction(
                model_config, data_manager, split_name, output_dir, gpu_id
            )
        elif model_type == 'baselines_only':
            raise ValueError("Baselines are handled by BaselineRunner, not ModelRunner")
        else:
            raise ValueError(f"Unknown model type: {model_type}. Currently only 'docker' is supported.")
    
    def _run_docker_prediction(self, model_config: Dict[str, Any], 
                               data_manager: DataManager, 
                               split_name: str, 
                               output_dir: Path,
                               gpu_id: Optional[int] = None) -> Path:
        """Execute model prediction in Docker container with caching.
        
        Args:
            model_config: Model configuration.
            data_manager: DataManager instance.
            split_name: Name of the split.
            output_dir: Output directory.
            
        Returns:
            Path to predictions file.
            
        Raises:
            FileNotFoundError: If model or checkpoint not found.
        """
        
        # Verify model path and training checkpoint exist
        model_path = Path(model_config['model_path'])
        training_checkpoint_path = model_path / 'training_checkpoint.json'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        if not training_checkpoint_path.exists():
            raise FileNotFoundError(f"Training checkpoint not found at {training_checkpoint_path}")
        
        # Get test conditions and data info
        data_file_path = Path(data_manager.config['data_path']).resolve()
        data_filename = data_file_path.name
        conditions = data_manager.get_perturbation_conditions(split_name)
        test_conditions = conditions['test']
        
        # Create prediction configuration for hashing
        pred_config = {
            'mode': 'predict',
            'data_path': str(data_file_path),
            'model_path': str(model_path),
            'split_name': split_name,
            'test_conditions': sorted(test_conditions),
            'covariate_key': data_manager.config['covariate_key'],
            'hyperparameters': model_config['hyperparameters'],
        }
        
        # Calculate inference hash
        from ..utils.hash_utils import calculate_inference_hash
        inference_hash = calculate_inference_hash(pred_config, training_checkpoint_path)
        
        # Define cache paths
        cache_dir = model_path / 'predictions_cache' / f'inf_{inference_hash[:12]}'
        cached_predictions = cache_dir / 'predictions.h5ad'
        
        # Check if we have cached predictions
        if cached_predictions.exists():
            log.info(f"Found cached predictions (hash: {inference_hash[:12]})")
            
            # Copy to output directory
            output_path = output_dir / 'predictions.h5ad'
            
            # Remove existing OUTPUT file if it exists (might be owned by root from previous Docker run)
            if output_path.exists():
                output_path.unlink()
            
            shutil.copy(cached_predictions, output_path)
            
            # Copy metadata
            shutil.copy(cache_dir / 'inference_metadata.json', output_dir / 'inference_metadata.json')
            
            log.info(f"Using cached predictions - saved significant compute time!")
            return output_path
        
        # No cache - must run inference
        log.info(f"No cached predictions found (hash: {inference_hash[:12]}), running inference...")
        
        # Create temporary directory for config file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create Docker prediction config
            docker_config = {
                'mode': 'predict',
                'data_path': f'/data/{data_filename}',
                'model_path': '/pretrained_model/',
                'split_name': split_name,
                'test_conditions': test_conditions,
                'covariate_key': data_manager.config['covariate_key'],
                'hyperparameters': model_config['hyperparameters'],
                'output_path': '/output/predictions.h5ad'
            }
            
            # Write config file
            config_path = temp_path / 'pred_config.json'
            with open(config_path, 'w') as f:
                json.dump(docker_config, f, indent=2)
            # Set up volumes
            volumes = {
                str(data_file_path.parent): {'bind': '/data', 'mode': 'ro'},
                str(model_path.resolve()): {'bind': '/pretrained_model', 'mode': 'ro'},
                str(output_dir.resolve()): {'bind': '/output', 'mode': 'rw'},
                str(config_path.resolve()): {'bind': '/config.json', 'mode': 'ro'}
            }
            
            # Set up environment variables
            environment = {}
            import os
            
            # Note: GPU assignment is now handled via Docker device_requests instead of CUDA_VISIBLE_DEVICES
            
            if model_config.get('name') == 'sclambda':
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    environment['OPENAI_API_KEY'] = api_key
                    log.info("OpenAI API key found and will be passed to container")
            
            # Run Docker container
            self.docker_runner.run_container(
                image=model_config['docker']['image'],
                command=['predict', '/config.json'],
                volumes=volumes,
                docker_config=model_config['docker'],
                container_name="Model prediction",
                environment=environment if environment else None,
                gpu_id=gpu_id
            )
        
        # Cache the results
        predictions_path = output_dir / 'predictions.h5ad'
        
        log.info(f"Caching predictions to: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(predictions_path, cached_predictions)
        
        # Save inference metadata
        inference_metadata = {
            'inference_hash': inference_hash,
            'inference_hash_short': inference_hash[:12],
            'timestamp': datetime.now().isoformat(),
            'prediction_config': pred_config,
            'docker_config': docker_config,
            'model_name': model_config.get('name'),
            'dataset_name': data_manager.config.get('name')
        }
        
        with open(cache_dir / 'inference_metadata.json', 'w') as f:
            json.dump(inference_metadata, f, indent=2)
        
        with open(output_dir / 'inference_metadata.json', 'w') as f:
            json.dump(inference_metadata, f, indent=2)
        
        return predictions_path
    
 