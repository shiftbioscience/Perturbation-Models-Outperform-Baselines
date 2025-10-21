"""
Main benchmark orchestration for CellSimBench framework.

Coordinates the complete benchmarking pipeline including model execution,
metrics calculation, and result visualization.
"""

import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import json
import warnings
from tqdm import tqdm
import pickle
import scanpy as sc
import re
from omegaconf import DictConfig, OmegaConf

from .data_manager import DataManager
from .model_runner import ModelRunner
from .metrics_engine import MetricsEngine
from .baseline_runner import BaselineRunner
from .plotting_engine import PlottingEngine
from .gpu_utils import get_available_gpus, calculate_gpu_assignment

log = logging.getLogger(__name__)


class BenchmarkRunner:
    """Main orchestration class for running benchmarks.
    
    Coordinates the complete benchmarking pipeline including data loading,
    model execution, metrics calculation, and visualization generation.
    
    Attributes:
        config: Hydra configuration object.
        data_manager: DataManager instance for data handling.
        model_runner: ModelRunner instance for model execution.
        
    Example:
        >>> runner = BenchmarkRunner(config)
        >>> results = runner.run_benchmark()
    """
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize BenchmarkRunner with configuration.
        
        Args:
            config: Hydra configuration containing dataset, model, and output settings.
        """
        self.config = config
        self.data_manager: Optional[DataManager] = None
        self.model_runner = ModelRunner()
        
    def run_benchmark(self) -> Dict[str, Any]:
        """Execute the complete benchmark pipeline."""
        log.info(f"Starting benchmark: {self.config.experiment.name}")
        
        # ALWAYS use k-fold logic (even if just 1 fold)
        return self._run_kfold_benchmark()

    
    def _run_kfold_benchmark(self) -> Dict[str, Any]:
        """Run benchmark across specified folds."""
        # Determine which folds to run - default to ALL folds
        if hasattr(self.config, 'fold_indices') and self.config.fold_indices is not None:
            fold_indices = self.config.fold_indices
        elif hasattr(self.config.dataset, 'folds'):
            # Default to ALL folds (consistent with training behavior)
            fold_indices = list(range(len(self.config.dataset.folds)))
        else:
            # Backward compatibility - no folds defined
            raise ValueError("No folds defined in dataset config")
        
        log.info(f"Running benchmark on fold(s): {fold_indices}")
        
        # Load dataset manager
        dataset_config = OmegaConf.to_object(self.config.dataset)
        
        self.data_manager = DataManager(dataset_config)
        _ = self.data_manager.load_dataset()
        
        # Gather predictions from specified folds
        all_predictions = self._gather_fold_predictions(fold_indices)
        
        # Always use aggregated_folds for consistency, even with single fold
        split_name = 'aggregated_folds'
        
        output_dir = self._get_output_dir()
        
        # Calculate metrics on concatenated predictions
        log.info("Calculating metrics on aggregated predictions...")
        results = self._calculate_all_metrics(all_predictions, split_name, output_dir)
        
        # Generate plots if configured
        if self.config.output['generate_plots']:
            plotting_engine = PlottingEngine(
                self.data_manager,
                results,
                all_predictions,
                split_name,
                output_dir,
                self.config
            )
            
            # Generate all plots for k-fold results (always aggregated_folds now)
            log.info("Generating plots for k-fold benchmark results")
            plotting_engine.generate_all_plots()
        
        log.info("K-fold benchmark completed successfully")
        return results
    
    def _gather_fold_predictions(self, fold_indices: List[int]) -> Dict[str, sc.AnnData]:
        """Gather predictions with optional parallelism."""
        
        should_use_parallel = self._should_use_parallel_inference(fold_indices)
        
        if should_use_parallel:
            log.info("🚀 Using parallel fold inference")
            return self._gather_fold_predictions_parallel(fold_indices)
        else:
            log.info("🔄 Using sequential fold inference")
            return self._gather_fold_predictions_sequential(fold_indices)
    
    def _should_use_parallel_inference(self, fold_indices: List[int]) -> bool:
        """Determine whether to use parallel inference."""
        
        # Use same execution config as training
        parallel_enabled = getattr(self.config.execution, 'parallel_folds', True)
        
        if not parallel_enabled:
            log.info("Parallel inference disabled in execution config")
            return False
        
        if len(fold_indices) <= 1:
            log.info("Single fold inference - using sequential")
            return False
        
        available_gpus = get_available_gpus()
        if not available_gpus:
            log.info("No GPUs available - falling back to sequential inference")
            return False
        
        return True
    
    def _gather_fold_predictions_parallel(self, fold_indices: List[int]) -> Dict[str, sc.AnnData]:
        """Gather predictions from folds in parallel."""
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
        
        log.info(f"Processing {len(fold_indices)} folds in parallel using {max_workers} workers")
        
        # Parallel execution
        fold_predictions = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fold inference jobs
            future_to_fold = {}
            for fold_idx in fold_indices:
                gpu_id = gpu_assignment[fold_idx]
                future = executor.submit(self._process_single_fold_with_gpu, fold_idx, gpu_id)
                future_to_fold[future] = fold_idx
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_fold):
                fold_idx = future_to_fold[future]
                try:
                    predictions = future.result()  # No timeout
                    fold_predictions[fold_idx] = predictions
                    log.info(f"✅ Fold {fold_idx} inference completed")
                except Exception as e:
                    log.error(f"❌ Fold {fold_idx} inference failed: {e}")
                    raise RuntimeError(f"Inference failed for fold {fold_idx}: {e}")
        
        # Concatenate predictions (existing logic)
        return self._concatenate_fold_predictions(fold_predictions, fold_indices)
    
    def _process_single_fold_with_gpu(self, fold_idx: int, gpu_id: int) -> Dict[str, sc.AnnData]:
        """Process single fold predictions with specific GPU."""
        fold_config = self.config.dataset.folds[fold_idx]
        fold_split = fold_config.split
        
        log.info(f"Processing fold {fold_idx} on GPU {gpu_id} (split: {fold_split})")
        
        return self._process_single_fold(fold_idx, gpu_id)
    
    def _process_single_fold(self, fold_idx: int, gpu_id: Optional[int] = None) -> Dict[str, sc.AnnData]:
        """Process single fold predictions (shared logic for both parallel and sequential)."""
        fold_config = self.config.dataset.folds[fold_idx]
        fold_split = fold_config.split
        
        # Get fold output directory
        output_dir = self._get_output_dir()
        fold_output_dir = output_dir / f'fold_{fold_split}'
        fold_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load fold-specific baselines
        fold_baselines = self._load_fold_baselines(fold_config)
        
        # Load universal baselines for this fold's test conditions
        baseline_runner = BaselineRunner(self.data_manager)
        universal_baselines = {}
        
        # Ground truth is REQUIRED
        if not self.config.dataset.get('ground_truth_baseline_key'):
            raise ValueError("ground_truth_baseline_key is required in dataset config")
        universal_baselines['ground_truth'] = baseline_runner.load_baseline(
            self.config.dataset.ground_truth_baseline_key,
            'ground_truth',
            fold_split
        )

        universal_baselines['control'] = baseline_runner.load_baseline(
            self.config.dataset.control_baseline_key,
            'control',
            fold_split
        )
        
        # Technical duplicate is REQUIRED
        if not self.config.dataset.get('technical_duplicate_baseline_key'):
            raise ValueError("technical_duplicate_baseline_key is required in dataset config")
        universal_baselines['technical_duplicate'] = baseline_runner.load_baseline(
            self.config.dataset.technical_duplicate_baseline_key,
            'technical_duplicate',
            fold_split
        )
        
        # Sparse mean baseline is optional
        if self.config.dataset.get('sparse_mean_baseline_key'):
            universal_baselines['sparse_mean'] = baseline_runner.load_baseline(
                self.config.dataset.sparse_mean_baseline_key,
                'sparse_mean',
                fold_split
            )

        if self.config.dataset.get('interpolated_duplicate_baseline_key'):
            universal_baselines['interpolated_duplicate'] = baseline_runner.load_baseline(
                self.config.dataset.interpolated_duplicate_baseline_key,
                'interpolated_duplicate',
                fold_split
            )
        
        # Additive baseline is optional
        if self.config.dataset.get('additive_baseline_key'):
            universal_baselines['additive'] = baseline_runner.load_baseline(
                self.config.dataset.additive_baseline_key,
                'additive',
                fold_split
            )
        
        fold_predictions = {}
        
        # Run model predictions for this fold
        # Check multi-model case FIRST (to avoid accessing deleted cfg.model)
        if hasattr(self.config, 'models'):
            # Multi-model case - NEW feature
            from omegaconf import open_dict
            from ..utils.hash_utils import get_model_path_for_config
            
            for model_config in self.config.models:
                if model_config.type == 'baselines_only':
                    continue
                
                # For multi-model: Calculate path directly without modifying self.config (thread-safe)
                from omegaconf import open_dict
                from ..utils.hash_utils import get_model_path_for_config
                
                # Clean the model config: remove model_path if CLI added it (it shouldn't be in hash)
                clean_model_config = OmegaConf.to_object(model_config)
                if 'model_path' in clean_model_config:
                    del clean_model_config['model_path']
                clean_model_config = OmegaConf.create(clean_model_config)
                
                # Create fold-specific dataset config
                fold_dataset_config = OmegaConf.create(OmegaConf.to_object(self.config.dataset))
                with open_dict(fold_dataset_config):
                    fold_dataset_config.split = fold_split
                
                # Create individual training config for this model
                individual_training_config = OmegaConf.create({
                    'output_dir': f'models/{model_config.name}_{self.config.dataset.name}/',
                    'save_intermediate': True
                })
                
                # Calculate model path (same logic as training)
                model_path = get_model_path_for_config(
                    fold_dataset_config,
                    clean_model_config,
                    individual_training_config
                )
                log.info(f"  Calculated model path for fold {fold_split}: {model_path}")
                
                # Create model config dict with path
                model_config_dict = OmegaConf.to_object(clean_model_config)
                model_config_dict['model_path'] = str(model_path)
                
                # Run predictions
                predictions_path = self.model_runner.run_model(
                    model_config_dict,
                    self.data_manager,
                    fold_split,
                    fold_output_dir,
                    gpu_id=gpu_id
                )
                
                # Load predictions
                model_predictions = sc.read_h5ad(predictions_path)
                
                # Add covariate column if missing (model predictions may not have it)
                if 'covariate' not in model_predictions.obs.columns:
                    log.warning("Covariate column not found in predictions, using ground truth covariate!!")
                    model_predictions.obs['covariate'] = universal_baselines['ground_truth'].obs['covariate'][0]
                
                # Fix "none" covariate values with real covariate from ground truth
                # TODO: Eventually we should fix this hack -- it's only for scgpt
                if 'covariate' in model_predictions.obs.columns:
                    none_mask = model_predictions.obs['covariate'] == 'none'
                    if none_mask.any():
                        log.warning(f"Found {none_mask.sum()} 'none' covariate values, replacing with ground truth covariate")
                        real_covariate = universal_baselines['ground_truth'].obs['covariate'].iloc[0]
                        
                        # Handle categorical covariate column
                        if hasattr(model_predictions.obs['covariate'], 'cat'):
                            # Add the new category if it doesn't exist
                            if real_covariate not in model_predictions.obs['covariate'].cat.categories:
                                model_predictions.obs['covariate'] = model_predictions.obs['covariate'].cat.add_categories([real_covariate])
                        
                        model_predictions.obs.loc[none_mask, 'covariate'] = real_covariate
                        
                        # Also fix pair_key if it exists and contains "none_"
                        if 'pair_key' in model_predictions.obs.columns:
                            none_pair_mask = model_predictions.obs['pair_key'].str.startswith('none_')
                            if none_pair_mask.any():
                                # Handle categorical pair_key column
                                new_pair_keys = (
                                    model_predictions.obs.loc[none_pair_mask, 'pair_key']
                                    .str.replace('none_', f'{real_covariate}_', regex=False)
                                )
                                
                                if hasattr(model_predictions.obs['pair_key'], 'cat'):
                                    # Add new categories if they don't exist
                                    new_categories = set(new_pair_keys) - set(model_predictions.obs['pair_key'].cat.categories)
                                    if new_categories:
                                        model_predictions.obs['pair_key'] = model_predictions.obs['pair_key'].cat.add_categories(list(new_categories))
                                
                                model_predictions.obs.loc[none_pair_mask, 'pair_key'] = new_pair_keys
                
                # Add delta calculations using fold-specific baselines
                model_predictions = self._add_delta_calculations(
                    model_predictions, fold_baselines, universal_baselines
                )
      
                model_display_name = self.get_model_display_name(
                    OmegaConf.to_object(model_config)
                )
                fold_predictions[model_display_name] = model_predictions
        elif hasattr(self.config, 'model') and self.config.model is not None and self.config.model.type != 'baselines_only':
            # Single model case - UNCHANGED from original code
            fold_model_config = self._create_fold_model_config(fold_split)
            
            # Run predictions
            predictions_path = self.model_runner.run_model(
                OmegaConf.to_object(fold_model_config.model),
                self.data_manager,
                fold_split,
                fold_output_dir,
                gpu_id=gpu_id
            )
            
            # Load predictions
            model_predictions = sc.read_h5ad(predictions_path)
            
            # Add covariate column if missing (model predictions may not have it)
            if 'covariate' not in model_predictions.obs.columns:
                log.warning("Covariate column not found in predictions, using ground truth covariate!!")
                model_predictions.obs['covariate'] = universal_baselines['ground_truth'].obs['covariate'][0]
            
            # Fix "none" covariate values with real covariate from ground truth
            # TODO: Eventually we should fix this hack -- it's only for scgpt
            if 'covariate' in model_predictions.obs.columns:
                none_mask = model_predictions.obs['covariate'] == 'none'
                if none_mask.any():
                    log.warning(f"Found {none_mask.sum()} 'none' covariate values, replacing with ground truth covariate")
                    real_covariate = universal_baselines['ground_truth'].obs['covariate'].iloc[0]
                    
                    # Handle categorical covariate column
                    if hasattr(model_predictions.obs['covariate'], 'cat'):
                        # Add the new category if it doesn't exist
                        if real_covariate not in model_predictions.obs['covariate'].cat.categories:
                            model_predictions.obs['covariate'] = model_predictions.obs['covariate'].cat.add_categories([real_covariate])
                    
                    model_predictions.obs.loc[none_mask, 'covariate'] = real_covariate
                    
                    # Also fix pair_key if it exists and contains "none_"
                    if 'pair_key' in model_predictions.obs.columns:
                        none_pair_mask = model_predictions.obs['pair_key'].str.startswith('none_')
                        if none_pair_mask.any():
                            # Handle categorical pair_key column
                            new_pair_keys = (
                                model_predictions.obs.loc[none_pair_mask, 'pair_key']
                                .str.replace('none_', f'{real_covariate}_', regex=False)
                            )
                            
                            if hasattr(model_predictions.obs['pair_key'], 'cat'):
                                # Add new categories if they don't exist
                                new_categories = set(new_pair_keys) - set(model_predictions.obs['pair_key'].cat.categories)
                                if new_categories:
                                    model_predictions.obs['pair_key'] = model_predictions.obs['pair_key'].cat.add_categories(list(new_categories))
                            
                            model_predictions.obs.loc[none_pair_mask, 'pair_key'] = new_pair_keys
            
            # Add delta calculations using fold-specific baselines
            model_predictions = self._add_delta_calculations(
                model_predictions, fold_baselines, universal_baselines
            )
  
            model_display_name = self.get_model_display_name(
                OmegaConf.to_object(self.config.model)
            )
            fold_predictions[model_display_name] = model_predictions
        
        # Determine what predictions to use for baseline filtering
        if fold_predictions:
            # Use first model's predictions for filtering baselines
            model_predictions = list(fold_predictions.values())[0]
        else:
            # No model predictions - need to use ground truth for filtering
            model_predictions = universal_baselines['ground_truth']
        
        # Process universal baselines - filter to match model's actual predictions
        for baseline_name, baseline_pred in tqdm(universal_baselines.items(), desc="Processing universal baselines"):
            # Filter baseline to match exact covariate-condition pairs that the model predicted
            # (some pairs may be skipped due to missing embeddings)
            model_pairs = model_predictions.obs[['covariate', 'condition']].drop_duplicates()
            
            # Create mask for matching covariate-condition pairs
            baseline_mask = pd.Series(False, index=baseline_pred.obs.index)
            for _, row in model_pairs.iterrows():
                pair_mask = (baseline_pred.obs['covariate'] == row['covariate']) & \
                           (baseline_pred.obs['condition'] == row['condition'])
                baseline_mask |= pair_mask
            
            filtered_baseline = baseline_pred[baseline_mask].copy()
            
            # Apply fold-specific delta calculations
            baseline_with_deltas = self._add_delta_calculations(
                filtered_baseline, fold_baselines, universal_baselines
            )
            
            fold_predictions[baseline_name] = baseline_with_deltas
        
        # Process fold-specific baselines - filter to match model predictions
        for baseline_name, baseline_pred in tqdm(fold_baselines.items(), desc="Processing fold-specific baselines"):
            # Filter baseline to match exact covariate-condition pairs that the model predicted
            model_pairs = model_predictions.obs[['covariate', 'condition']].drop_duplicates()
            
            # Create mask for matching covariate-condition pairs
            baseline_mask = pd.Series(False, index=baseline_pred.obs.index)
            for _, row in model_pairs.iterrows():
                pair_mask = (baseline_pred.obs['covariate'] == row['covariate']) & \
                           (baseline_pred.obs['condition'] == row['condition'])
                baseline_mask |= pair_mask
            
            filtered_baseline = baseline_pred[baseline_mask].copy()
            
            # Apply fold-specific delta calculations
            baseline_with_deltas = self._add_delta_calculations(
                filtered_baseline, fold_baselines, universal_baselines
            )
            fold_predictions[baseline_name] = baseline_with_deltas
        
        return fold_predictions
    
    def _gather_fold_predictions_sequential(self, fold_indices: List[int]) -> Dict[str, sc.AnnData]:
        """Sequential fold predictions (existing implementation)."""
        all_fold_predictions = {}  # Will store lists of predictions per model/baseline
        
        # Process each fold
        for fold_idx in fold_indices:
            fold_config = self.config.dataset.folds[fold_idx]
            fold_split = fold_config.split
            
            log.info(f"Processing fold {fold_idx} sequentially: {fold_split}")
            
            # Get predictions for this fold
            fold_predictions = self._process_single_fold(fold_idx)
            
            # Accumulate predictions
            for model_name, predictions in fold_predictions.items():
                if model_name not in all_fold_predictions:
                    all_fold_predictions[model_name] = []
                all_fold_predictions[model_name].append(predictions)
        
        # Concatenate predictions if multiple folds
        if len(fold_indices) > 1:
            return self._concatenate_fold_predictions_from_lists(all_fold_predictions)
        else:
            # Single fold - just unwrap the lists
            return {name: preds[0] for name, preds in all_fold_predictions.items()}
    
    def _concatenate_fold_predictions(self, fold_predictions: Dict[int, Dict[str, sc.AnnData]], 
                                     fold_indices: List[int]) -> Dict[str, sc.AnnData]:
        """Concatenate predictions from multiple folds (parallel result format)."""
        # Convert to list format
        all_fold_predictions = {}
        for fold_idx in fold_indices:
            for model_name, predictions in fold_predictions[fold_idx].items():
                if model_name not in all_fold_predictions:
                    all_fold_predictions[model_name] = []
                all_fold_predictions[model_name].append(predictions)
        
        return self._concatenate_fold_predictions_from_lists(all_fold_predictions)
    
    def _concatenate_fold_predictions_from_lists(self, all_fold_predictions: Dict[str, List[sc.AnnData]]) -> Dict[str, sc.AnnData]:
        """Concatenate predictions from list format."""
        log.info("Concatenating predictions from all folds...")
        concatenated_predictions = {}
        for model_name, fold_predictions_list in all_fold_predictions.items():
            # Concatenate along axis 0 (observations), preserving obsm fields
            concatenated = sc.concat(fold_predictions_list, axis=0, merge='same')
            concatenated_predictions[model_name] = concatenated
            log.info(f"  {model_name}: {concatenated.shape[0]} total predictions")
        return concatenated_predictions
    

    def _load_fold_baselines(self, fold_config: Dict) -> Dict[str, sc.AnnData]:
        """Load baselines specific to a fold using the main DataManager."""
        baselines = {}
        baseline_runner = BaselineRunner(self.data_manager)
        
        baselines['dataset_mean'] = baseline_runner.load_baseline(
            fold_config.dataset_mean_baseline_key,
            'dataset_mean',
            fold_config.split
        )
        
        if fold_config.get('linear_baseline_key'):
            baselines['linear'] = baseline_runner.load_baseline(
                fold_config.linear_baseline_key,
                'linear',
                fold_config.split
            )
        
        return baselines
    
    def _add_delta_calculations(self, predictions: sc.AnnData,
                               fold_baselines: Dict[str, sc.AnnData],
                               universal_baselines: Dict[str, sc.AnnData]) -> sc.AnnData:
        """Add delta calculations to predictions using fold-specific baselines."""
        
        # Add delta from control if available
        if 'control' in fold_baselines or 'control' in universal_baselines:
            control_pred = fold_baselines['control'] if 'control' in fold_baselines else universal_baselines['control']
            delta_ctrl = np.zeros_like(predictions.X)
            # Find intersection of control and predictions var_names
            common_var_names = set(control_pred.var_names) & set(predictions.var_names)
            if not common_var_names:
                raise ValueError("No common variable names found between control and predictions")
            
            # Filter control and predictions to only include common var_names
            control_pred = control_pred[:, list(common_var_names)]
            predictions = predictions[:, list(common_var_names)]
            
            for i, cov in enumerate(tqdm(predictions.obs['covariate'], desc="Adding delta from control")):
                # Find matching control - MUST exist
                mask = (control_pred.obs['covariate'] == cov)
                if not mask.any():
                    # TODO: Why would this happen? I mean it's fine because all the values are identical, but still...
                    warnings.warn(f"No control baseline found for covariate={cov}")
                delta_ctrl[i] = predictions.X[i] - control_pred[mask].X[0]
            
            predictions.obsm['delta_ctrl'] = delta_ctrl
        else:
            raise ValueError("No control baseline found")
        
        # Add delta from dataset mean if available
        if 'dataset_mean' in fold_baselines:
            dataset_mean_pred = fold_baselines['dataset_mean']
            delta_mean = np.zeros_like(predictions.X)
            # Find intersection of dataset mean and predictions var_names
            common_var_names = set(dataset_mean_pred.var_names) & set(predictions.var_names)
            if not common_var_names:
                raise ValueError("No common variable names found between dataset mean and predictions")
            
            # Filter dataset mean and predictions to only include common var_names
            dataset_mean_pred = dataset_mean_pred[:, list(common_var_names)]
            predictions = predictions[:, list(common_var_names)]
            
            for i, cov in enumerate(tqdm(predictions.obs['covariate'], desc="Adding delta from dataset mean")):
                # Find matching dataset mean - MUST exist
                mask = (dataset_mean_pred.obs['covariate'] == cov)
                if not mask.any():
                    warnings.warn(f"No dataset mean baseline found for covariate={cov}")
                delta_mean[i] = predictions.X[i] - dataset_mean_pred[mask].X[0]
            
            predictions.obsm['delta_mean'] = delta_mean
        else:
            raise ValueError("No dataset mean baseline found")
        
        return predictions
    

    def _create_fold_model_config(self, fold_split: str) -> DictConfig:
        """Create model config for specific fold."""
        from omegaconf import open_dict
        from ..utils.hash_utils import get_model_path_for_config
        
        fold_config = OmegaConf.create(OmegaConf.to_object(self.config))
        with open_dict(fold_config):
            # Update the dataset split
            fold_config.dataset.split = fold_split
            
            # Update training output_dir to match the actual dataset name
            fold_config.training.output_dir = f'models/{fold_config.model.name}_{fold_config.dataset.name}/'
            
            # Calculate and add the model path for this fold's trained model
            model_path = get_model_path_for_config(
                fold_config.dataset,
                fold_config.model,
                fold_config.training
            )
            fold_config.model.model_path = str(model_path)
            log.info(f"  Calculated model path for fold {fold_split}: {model_path}")
        
        return fold_config
    
    def _get_generic_baseline_name(self, fold_specific_name: str) -> str:
        """Map fold-specific baseline name to generic name.
        
        Only applies to truly fold-specific baselines (ctrl, split_mean, linear).
        Universal baselines (additive, technical_duplicate) don't get mapped.
        
        E.g., 'split_fold_0_ctrl_baseline' -> 'ctrl_baseline'
             'split_fold_1_split_mean_baseline' -> 'split_mean_baseline'
        """
        # Handle technical duplicate renaming (universal baseline)
        if 'technical_duplicate_second_half' in fold_specific_name:
            return 'technical duplicate'
        
        # Remove split_fold_N_ prefix using regex (for fold-specific baselines)
        generic_name = re.sub(r'^split_fold_\d+_', '', fold_specific_name)
        
        # Also handle older _foldN suffix pattern if it exists
        generic_name = re.sub(r'_fold\d+$', '', generic_name)
        
        return generic_name
    
    def _get_output_dir(self) -> Path:
        """Get output directory based on experiment name."""
        from datetime import datetime
        
        # Create output directory based on experiment name (set by CLI for multi-model)
        experiment_name = self.config.experiment.name
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = Path(f"outputs/{experiment_name}/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _calculate_all_metrics(self, all_predictions: Dict[str, sc.AnnData], split_name: str, output_dir: Path) -> Dict[str, Any]:
        """Calculate metrics for all models."""
        # Check if nir analysis should be run (default False)
        run_nir = getattr(self.config, 'run_nir_analysis', False)
        metrics_engine = MetricsEngine(self.data_manager, run_nir=run_nir)
        
        # Handle aggregated_folds special case
        if split_name == 'aggregated_folds':
            # For aggregated folds, count unique covariate-condition pairs from predictions
            sample_pred = list(all_predictions.values())[0]
            unique_pairs = sample_pred.obs[['covariate', 'condition']].drop_duplicates()
            n_test_pairs = len(unique_pairs)
        else:
            n_test_pairs = len(self.data_manager.get_covariate_condition_pairs(split_name, 'test'))
        
        results = {
            'config': OmegaConf.to_object(self.config),
            'split_used': split_name,
            'models': {},
            'metadata': {
                'n_models_run': len(all_predictions),
                'n_genes': self.data_manager.adata.n_vars,
                'n_test_cov_pert_pairs': n_test_pairs
            }
        }
        
        # Extract ground truth from predictions
        if 'ground_truth' not in all_predictions:
            raise ValueError("ground_truth baseline is required but not found in predictions")
        
        ground_truth_adata = all_predictions['ground_truth']
        # Convert AnnData predictions to DataFrame format for metrics engine
        ground_truth_df, ground_truth_deltas = self._extract_dataframes_and_deltas(ground_truth_adata)
        
        # Create nir cache directory if running nir analysis
        # Use a persistent location (not timestamped) so cache is reused across runs
        nir_cache_dir = None
        if run_nir:
            dataset_name = self.config.dataset.name
            nir_cache_dir = Path(f"outputs/.nir_cache/{dataset_name}")
            nir_cache_dir.mkdir(exist_ok=True, parents=True)
            log.info(f"nir analysis enabled - using cache directory: {nir_cache_dir}")
        
        for model_name, predictions_adata in all_predictions.items():
            if model_name == 'ground_truth':
                continue  # Skip ground truth in model processing
                
            log.info(f"Calculating metrics for: {model_name}")
            
            # Convert AnnData to DataFrame format
            predictions_df, predictions_deltas = self._extract_dataframes_and_deltas(predictions_adata)
            
            # Check for cached nir results if enabled
            cached_nir_scores = None
            if run_nir and nir_cache_dir:
                cached_nir_scores = self._load_cached_nir_scores(
                    model_name, predictions_df, ground_truth_df, nir_cache_dir
                )
            
            # Calculate metrics using new format
            model_metrics = metrics_engine.calculate_all_metrics(
                predictions_df,
                predictions_deltas, 
                ground_truth_df,
                ground_truth_deltas,
                cached_nir_scores=cached_nir_scores
            )
            
            # Cache nir scores for future runs if we calculated them
            if run_nir and nir_cache_dir and cached_nir_scores is None:
                self._save_nir_scores_to_cache(
                    model_name, predictions_df, ground_truth_df, 
                    model_metrics.get('nir', {}), nir_cache_dir
                )
            
            model_summary = self._calculate_summary_stats(model_metrics)
            results['models'][model_name] = {
                'metrics': model_metrics,
                'summary_stats': model_summary
            }
        
        # Save results
        self._save_results(results, output_dir)
        
        # Print multi-model summary
        self._print_multi_model_summary(results)
        
        return results
    
    def _extract_dataframes_and_deltas(self, adata: sc.AnnData) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Extract expression DataFrames and delta DataFrames from AnnData.
        
        Args:
            adata: AnnData object with already mean-aggregated expression data and delta calculations in obsm
            
        Returns:
            Tuple of (expressions_df, deltas_dict) where:
            - expressions_df: DataFrame with cov_pert_key as index
            - deltas_dict: {'deltactrl': DataFrame, 'deltamean': DataFrame} with cov_pert_key as index
        """
        # Create covariate-condition key column
        adata.obs['cov_pert_key'] = adata.obs['covariate'].astype(str) + '_' + adata.obs['condition'].astype(str)
        
        # Convert entire AnnData to DataFrame in one go
        expressions_df = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs['cov_pert_key'])
        
        deltas = {}
        
        deltas['deltactrl'] = pd.DataFrame(adata.obsm['delta_ctrl'], columns=adata.var_names, index=adata.obs['cov_pert_key'])        
        deltas['deltamean'] = pd.DataFrame(adata.obsm['delta_mean'], columns=adata.var_names, index=adata.obs['cov_pert_key'])

        # Remove any keys containing "ctrl" or "control" from the dataframes
        expressions_df = expressions_df[~expressions_df.index.str.contains('ctrl')]
        expressions_df = expressions_df[~expressions_df.index.str.contains('control')]
        deltas['deltactrl'] = deltas['deltactrl'][~deltas['deltactrl'].index.str.contains('ctrl')]
        deltas['deltamean'] = deltas['deltamean'][~deltas['deltamean'].index.str.contains('ctrl')]
        deltas['deltactrl'] = deltas['deltactrl'][~deltas['deltactrl'].index.str.contains('control')]
        deltas['deltamean'] = deltas['deltamean'][~deltas['deltamean'].index.str.contains('control')]

        
        return expressions_df, deltas
    
    def _print_multi_model_summary(self, results: Dict[str, Any]):
        """Print a nice summary table for all models."""
        if not results['models']:
            print("No models were run.")
            return
        
        # Filter out ground_truth and rename control to control_mean for display
        display_models = {}
        for model_name, model_data in results['models'].items():
            if model_name == 'ground_truth':
                continue  # Skip ground truth - it's the reference, not a model to compare
            display_name = 'control_mean' if model_name == 'control' else model_name
            display_models[display_name] = model_data
        
        if not display_models:
            print("No models to display (ground_truth is hidden).")
            return
        
        # Calculate dynamic column width for model names
        max_model_name_len = max(len(name) for name in display_models.keys())
        model_col_width = max(max_model_name_len + 2, len("Model") + 2)  # At least as wide as "Model" header
        
        # Calculate total table width for ALL metrics (added 3 nir columns)
        col_widths = [model_col_width, 8, 8, 10, 12, 10, 12, 10, 12, 10, 12, 10, 10, 9, 12, 12]  # All column widths
        total_width = sum(col_widths) + len(col_widths) - 1  # +spaces between columns
        
        print("\n" + "="*(total_width))
        print(f"BENCHMARK RESULTS SUMMARY (split: {results['split_used']})")
        print("="*(total_width+2))
        
        # Header with ALL metrics
        print(f"{'Model':<{model_col_width}} {'MSE':<8} {'WMSE':<8} {'rΔ Ctrl':<10} {'rΔ Ctrl DEG':<12} {'rΔ Pert':<10} {'rΔ Pert DEG':<12} {'R²Δ Ctrl':<10} {'R²Δ Ctrl DEG':<12} {'R²Δ Pert':<10} {'R²Δ Pert DEG':<12} {'WR²Δ Ctrl':<10} {'WR²Δ Pert':<10} {'Cent Acc':<9}")
        print("-" * total_width)
        
        # Model rows
        model_scores = {}
        for model_name, model_results in display_models.items():
            stats = model_results['summary_stats']
            
            # Get mean values for display - ALL metrics
            mse = stats.get('mse_mean', float('nan'))
            wmse = stats.get('wmse_mean', float('nan'))
            pearson_deltactrl = stats.get('pearson_deltactrl_mean', float('nan'))
            pearson_deltactrl_degs = stats.get('pearson_deltactrl_degs_mean', float('nan'))
            pearson_deltapert = stats.get('pearson_deltapert_mean', float('nan'))
            pearson_deltapert_degs = stats.get('pearson_deltapert_degs_mean', float('nan'))
            r2_deltactrl = stats.get('r2_deltactrl_mean', float('nan'))
            r2_deltactrl_degs = stats.get('r2_deltactrl_degs_mean', float('nan'))
            r2_deltapert = stats.get('r2_deltapert_mean', float('nan'))
            r2_deltapert_degs = stats.get('r2_deltapert_degs_mean', float('nan'))
            weighted_r2_deltactrl = stats.get('weighted_r2_deltactrl_mean', float('nan'))
            weighted_r2_deltapert = stats.get('weighted_r2_deltapert_mean', float('nan'))
            nir = stats.get('nir_mean', float('nan'))
            
            # Store for ranking (using control-based DEGs metric)
            model_scores[model_name] = {
                'pearson_deltactrl_degs': pearson_deltactrl_degs,
                'pearson_deltapert_degs': pearson_deltapert_degs,
                'r2_deltactrl_degs': r2_deltactrl_degs,
                'r2_deltapert_degs': r2_deltapert_degs,
                'weighted_r2_deltactrl': weighted_r2_deltactrl,
                'weighted_r2_deltapert': weighted_r2_deltapert,
                'mse': mse,
                'wmse': wmse,
                'pearson_deltactrl': pearson_deltactrl,
                'pearson_deltapert': pearson_deltapert,
                'r2_deltactrl': r2_deltactrl,
                'r2_deltapert': r2_deltapert,
                'nir': nir,
            }
            
            print(f"{model_name:<{model_col_width}} {mse:<8.4f} {wmse:<8.4f} {pearson_deltactrl:<10.4f} {pearson_deltactrl_degs:<12.4f} {pearson_deltapert:<10.4f} {pearson_deltapert_degs:<12.4f} {r2_deltactrl:<10.4f} {r2_deltactrl_degs:<12.4f} {r2_deltapert:<10.4f} {r2_deltapert_degs:<12.4f} {weighted_r2_deltactrl:<10.4f} {weighted_r2_deltapert:<10.4f} {nir:<9.4f}")
        
        # Find best model
        if model_scores:
            # Get the best model by weighted R² Δ
            best_model_name = None
            best_weighted_r2_deltapert = float('-inf')
            
            for model_name, model_scores in model_scores.items():
                if 'technical duplicate' in model_name:
                    continue
                current_weighted_r2_deltapert = model_scores['weighted_r2_deltapert']
                if current_weighted_r2_deltapert > best_weighted_r2_deltapert:
                    best_weighted_r2_deltapert = current_weighted_r2_deltapert
                    best_model_name = model_name
            
            if best_model_name:
                print(f"\nBest performing model 🎖️: {best_model_name} (Weighted R² Δ Pert mean: {best_weighted_r2_deltapert:.4f})")
            else:
                print("\nNo valid model scores found")
        
        print("="*total_width)
    
    def _calculate_summary_stats(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate summary statistics across covariate-perturbation pairs."""
        summary = {}
        total_count = len(metrics['mse'])

        for metric_name, cov_pert_scores in metrics.items():
            if not cov_pert_scores:
                continue

            # Deal with cases where scores are all nan or 0 (0 can happen probably due to tolerance issues for pearson delta DEGs)
            if len(cov_pert_scores) == 0 or np.sum(list(cov_pert_scores.values())) == 0:
                summary[f"{metric_name}_mean"] = np.nan
                summary[f"{metric_name}_median"] = np.nan
                summary[f"{metric_name}_std"] = np.nan
                summary[f"{metric_name}_nan_prop"] = 1
                continue
                
            scores = list(cov_pert_scores.values())
            scores_non_nan = [s for s in scores if not np.isnan(s)]
            
            if scores_non_nan:
                summary[f"{metric_name}_mean"] = np.mean(scores_non_nan)
                summary[f"{metric_name}_median"] = np.median(scores_non_nan) 
                summary[f"{metric_name}_std"] = np.std(scores_non_nan)
                summary[f"{metric_name}_nan_prop"] = 1 - len(scores_non_nan) / total_count
            else:
                # All scores are NaN
                summary[f"{metric_name}_mean"] = np.nan
                summary[f"{metric_name}_median"] = np.nan
                summary[f"{metric_name}_std"] = np.nan
                summary[f"{metric_name}_nan_prop"] = 1.0
        
        return summary
    
    def _create_detailed_metrics_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create a long-format table with perturbation-level metrics for all models.
        
        Args:
            results: Dictionary containing all model results with metrics.
            
        Returns:
            DataFrame with columns: model, perturbation, metric, value
        """
        detailed_rows = []
        
        for model_name, model_data in results.get('models', {}).items():
            # Skip ground_truth in detailed metrics - it's the reference, not a model to compare
            if model_name == 'ground_truth':
                continue
            # Rename control to control_mean for consistency
            display_name = 'control_mean' if model_name == 'control' else model_name
            model_metrics = model_data.get('metrics', {})
            
            for metric_name, perturbation_scores in model_metrics.items():
                for perturbation, value in perturbation_scores.items():
                    detailed_rows.append({
                        'model': display_name,
                        'perturbation': perturbation,
                        'metric': metric_name,
                        'value': value
                    })
        
        return pd.DataFrame(detailed_rows)
    
    def get_model_display_name(self, model_config: Dict) -> str:
        """Get display name for benchmarking outputs."""
        return model_config.get('display_name', model_config.get('name'))
    
    def _load_cached_nir_scores(
        self, model_name: str, predictions_df: pd.DataFrame, 
        ground_truth_df: pd.DataFrame, cache_dir: Path
    ) -> Optional[Dict[str, float]]:
        """Load cached nir scores if they exist and are valid."""
        import hashlib
        
        # Create a stable hash based on index (perturbation keys) only
        # This is stable because the same test set = same perturbation keys
        pred_keys = sorted(predictions_df.index.tolist())
        gt_keys = sorted(ground_truth_df.index.tolist())
        
        pred_hash = hashlib.md5(json.dumps(pred_keys, sort_keys=True).encode()).hexdigest()[:12]
        gt_hash = hashlib.md5(json.dumps(gt_keys, sort_keys=True).encode()).hexdigest()[:12]
        cache_key = f"{model_name}_{pred_hash}_{gt_hash}"
        cache_file = cache_dir / f"{cache_key}_nir.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_scores = json.load(f)
                log.info(f"  ✓ Loaded cached nir scores for {model_name} from {cache_file.name}")
                return cached_scores
            except Exception as e:
                log.warning(f"  Failed to load cached nir scores: {e}")
                return None
        
        log.info(f"  No cached nir scores found for {model_name}, will calculate")
        return None
    
    def _save_nir_scores_to_cache(
        self, model_name: str, predictions_df: pd.DataFrame, 
        ground_truth_df: pd.DataFrame, nir_scores: Dict[str, float], cache_dir: Path
    ) -> None:
        """Save nir scores to cache for future runs."""
        import hashlib
        
        # Create the same hash as in load
        pred_keys = sorted(predictions_df.index.tolist())
        gt_keys = sorted(ground_truth_df.index.tolist())
        
        pred_hash = hashlib.md5(json.dumps(pred_keys, sort_keys=True).encode()).hexdigest()[:12]
        gt_hash = hashlib.md5(json.dumps(gt_keys, sort_keys=True).encode()).hexdigest()[:12]
        cache_key = f"{model_name}_{pred_hash}_{gt_hash}"
        cache_file = cache_dir / f"{cache_key}_nir.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(nir_scores, f, indent=2)
            log.info(f"  ✓ Saved nir scores to cache: {cache_file.name}")
        except Exception as e:
            log.warning(f"  Failed to save nir scores to cache: {e}")

            
    
    def _save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save benchmark results to files."""
        
        # Save JSON results (for easy inspection)
        json_path = output_dir / 'results.json'
        json_results = results.copy()
        
        # Convert numpy types to native Python types for JSON serialization
        for model_name, model_data in json_results.get('models', {}).items():
            if 'summary_stats' in model_data:
                model_data['summary_stats'] = {k: float(v) if isinstance(v, np.number) else v 
                                             for k, v in model_data['summary_stats'].items()}
            if 'metrics' in model_data:
                model_data['metrics'] = {
                    metric_name: {k: float(v) if isinstance(v, np.number) else v 
                                 for k, v in metric_dict.items()}
                    for metric_name, metric_dict in model_data['metrics'].items()
                }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        log.info(f"Results saved to {json_path}")
        
        # Save pickle results (preserves all data types)
        pickle_path = output_dir / 'results.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        log.info(f"Results saved to {pickle_path}")
        
        # Save summary stats as CSV for easy analysis  
        if results.get('models'):
            summary_rows = []
            for model_name, model_data in results['models'].items():
                # Skip ground_truth in CSV output - it's the reference, not a model to compare
                if model_name == 'ground_truth':
                    continue
                # Rename control to control_mean for consistency
                display_name = 'control_mean' if model_name == 'control' else model_name
                row = {'model': display_name}
                row.update(model_data.get('summary_stats', {}))
                summary_rows.append(row)
            
            summary_df = pd.DataFrame(summary_rows)
            csv_path = output_dir / 'summary_stats.csv'
            summary_df.to_csv(csv_path, index=False)
            log.info(f"Summary stats saved to {csv_path}")
            
            # Save detailed perturbation-level metrics
            detailed_df = self._create_detailed_metrics_table(results)
            detailed_csv_path = output_dir / 'detailed_metrics.csv'
            detailed_df.to_csv(detailed_csv_path, index=False)
            log.info(f"Detailed perturbation-level metrics saved to {detailed_csv_path}")
        
        # Print handled by _print_multi_model_summary 