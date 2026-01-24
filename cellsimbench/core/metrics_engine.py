"""
Metrics calculation engine for CellSimBench framework.

Provides comprehensive metrics computation for evaluating perturbation
response predictions against ground truth.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import scanpy as sc
import warnings
from tqdm import tqdm

# Import metrics functions from data_manager
from .data_manager import mse, wmse, pearson, r2_score_on_deltas, DataManager


class MetricsEngine:

    def __init__(self, data_manager: DataManager, run_nir: bool = False) -> None:
        """Initialize MetricsEngine with a DataManager instance.
        
        Args:
            data_manager: DataManager for accessing ground truth data and DEG weights.
            run_nir: Whether to run nir calculation (default False).
        """
        self.data_manager = data_manager
        self.run_nir = run_nir
        
    def calculate_all_metrics(
        self,
        predictions: pd.DataFrame,
        predictions_deltas: Dict[str, pd.DataFrame],
        ground_truth: pd.DataFrame,
        ground_truth_deltas: Dict[str, pd.DataFrame],
        cached_nir_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:

        # Ensure predictions and ground truth have the same var_names
        # Use sorted list to ensure deterministic, reproducible ordering
        common_var_names = sorted(set(predictions.columns) & set(ground_truth.columns))

        if not common_var_names:
            raise ValueError("Predictions and ground truth have different var_names")
        predictions = predictions[common_var_names]
        ground_truth = ground_truth[common_var_names]
        predictions_deltas = {key: df[common_var_names] for key, df in predictions_deltas.items()}
        ground_truth_deltas = {key: df[common_var_names] for key, df in ground_truth_deltas.items()}


        # Calculate nir scores (needs full dataset) - only if enabled
        if cached_nir_scores is not None:
            # Use cached scores
            nir_scores = cached_nir_scores
        elif self.run_nir:
            # Calculate fresh scores
            nir_scores = self._calculate_nir_scores(
                predictions, ground_truth
            )
        else:
            # Skip nir analysis, provide default scores
            nir_scores = {key: 0.0 for key in predictions.index}

        # Get all covariate-condition pairs from DataFrame index
        cov_condition_pairs = [(key.split('_')[0], '_'.join(key.split('_')[1:])) 
                              for key in predictions.index]
        
        # Calculate metrics for each covariate-condition pair
        condition_metrics = {}

        for covariate_value, condition in tqdm(cov_condition_pairs):
            cov_pert_key = f"{covariate_value}_{condition}"

            pred_expression = predictions.loc[cov_pert_key].values

            if cov_pert_key not in ground_truth.index:
                print(f"Covariate-condition pair {cov_pert_key} not found in ground truth")
                continue
            truth_expression = ground_truth.loc[cov_pert_key].values
            
            # Get pre-computed deltas
            pred_deltas_ctrl = predictions_deltas['deltactrl'].loc[cov_pert_key].values
            truth_deltas_ctrl = ground_truth_deltas['deltactrl'].loc[cov_pert_key].values
            pred_deltas_mean = predictions_deltas['deltamean'].loc[cov_pert_key].values
            truth_deltas_mean = ground_truth_deltas['deltamean'].loc[cov_pert_key].values


            # Get DEG weights and mask using covariate and perturbation
            weights = self.data_manager.get_deg_weights(covariate_value, condition, gene_order=common_var_names)
            deg_mask = self.data_manager.get_deg_mask(covariate_value, condition, gene_order=common_var_names)
            condition_metrics[cov_pert_key] = {
                'mse': self._calculate_mse(pred_expression, truth_expression),
                'wmse': self._calculate_wmse(pred_expression, truth_expression, weights),
                
                # Delta metrics with control baseline - use pre-supplied deltas
                'pearson_deltactrl': self._calculate_pearson_delta_direct(pred_deltas_ctrl, truth_deltas_ctrl),
                'pearson_deltactrl_degs': self._calculate_pearson_delta_direct(
                    pred_deltas_ctrl[deg_mask], truth_deltas_ctrl[deg_mask]
                ) if deg_mask.sum() > 2 else np.nan,
                'r2_deltactrl': self._calculate_r2_delta_direct(pred_deltas_ctrl, truth_deltas_ctrl),
                'r2_deltactrl_degs': self._calculate_r2_delta_direct(
                    pred_deltas_ctrl[deg_mask], truth_deltas_ctrl[deg_mask]
                ) if deg_mask.sum() > 2 else np.nan,
                'weighted_r2_deltactrl': self._calculate_weighted_r2_delta_direct(
                    pred_deltas_ctrl, truth_deltas_ctrl, weights
                ),
                
                # Delta metrics with dataset mean baseline - use pre-supplied deltas
                'pearson_deltapert': self._calculate_pearson_delta_direct(pred_deltas_mean, truth_deltas_mean),
                'pearson_deltapert_degs': self._calculate_pearson_delta_direct(
                    pred_deltas_mean[deg_mask], truth_deltas_mean[deg_mask]
                ) if deg_mask.sum() > 2 else np.nan,
                'r2_deltapert': self._calculate_r2_delta_direct(pred_deltas_mean, truth_deltas_mean),
                'r2_deltapert_degs': self._calculate_r2_delta_direct(
                    pred_deltas_mean[deg_mask], truth_deltas_mean[deg_mask]
                ) if deg_mask.sum() > 2 else np.nan,
                'weighted_r2_deltapert': self._calculate_weighted_r2_delta_direct(
                    pred_deltas_mean, truth_deltas_mean, weights
                ),
                
                # nir metrics
                'nir': nir_scores[cov_pert_key] if cov_pert_key in nir_scores else np.nan,
                # 'nir_deltactrl': nir_deltactrl_scores[cov_pert_key],
                # 'nir_deltamean': nir_deltamean_scores[cov_pert_key],
            }

            
        # Reorganize to metric -> cov_pert_key -> score format
        organized_metrics = {}
        for metric in ['mse', 'wmse', 'pearson_deltactrl', 'pearson_deltactrl_degs',
                      'r2_deltactrl', 'r2_deltactrl_degs', 'weighted_r2_deltactrl',
                      'pearson_deltapert', 'pearson_deltapert_degs', 'r2_deltapert',
                      'r2_deltapert_degs', 'weighted_r2_deltapert',
                      'nir']:
            organized_metrics[metric] = {
                cov_pert_key: condition_metrics[cov_pert_key][metric] 
                for cov_pert_key in condition_metrics.keys()
            }
        
        return organized_metrics
    
    def _calculate_mse(self, pred: np.ndarray, truth: np.ndarray) -> float:
        """Calculate MSE following plotting.py logic.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            
        Returns:
            Mean squared error.
        """
        return mse(pred, truth)
    
    def _calculate_wmse(self, pred: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted MSE following plotting.py logic.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            weights: DEG-based weights for each gene.
            
        Returns:
            Weighted mean squared error.
        """
        return wmse(pred, truth, weights)
    
    def _calculate_pearson_delta(self, pred: np.ndarray, truth: np.ndarray, 
                               control: np.ndarray) -> float:
        """Calculate Pearson correlation of deltas.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            control: Control/baseline expression values.
            
        Returns:
            Pearson correlation coefficient of deltas from control.
        """
        delta_pred = pred - control
        delta_truth = truth - control
        try:
            corr, _ = pearsonr(delta_pred, delta_truth)
            return corr
        except:
            return np.nan
    
    def _calculate_pearson_delta_degs(self, pred: np.ndarray, truth: np.ndarray,
                                    control: np.ndarray, deg_mask: np.ndarray) -> float:
        """Calculate Pearson correlation of deltas for DEGs only.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            control: Control/baseline expression values.
            deg_mask: Boolean mask indicating DEG positions.
            
        Returns:
            Pearson correlation coefficient for DEGs only.
        """
        delta_pred = pred[deg_mask] - control[deg_mask]
        delta_truth = truth[deg_mask] - control[deg_mask]
        try:
            corr, _ = pearsonr(delta_pred, delta_truth)
            return corr
        except:
            return np.nan
    
    def _calculate_r2_delta(self, pred: np.ndarray, truth: np.ndarray,
                          control: np.ndarray) -> float:
        """Calculate R² on deltas.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            control: Control/baseline expression values.
            
        Returns:
            R² score on delta values.
        """
        delta_pred = pred - control
        delta_truth = truth - control
        return r2_score_on_deltas(delta_truth, delta_pred)
    
    def _calculate_r2_delta_degs(self, pred: np.ndarray, truth: np.ndarray,
                               control: np.ndarray, deg_mask: np.ndarray) -> float:
        """Calculate R² on deltas for DEGs only.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            control: Control/baseline expression values.
            deg_mask: Boolean mask indicating DEG positions.
            
        Returns:
            R² score on delta values for DEGs only.
        """
        delta_pred = pred[deg_mask] - control[deg_mask]
        delta_truth = truth[deg_mask] - control[deg_mask]
        return r2_score_on_deltas(delta_truth, delta_pred)
    
    def _calculate_weighted_r2_delta(self, pred: np.ndarray, truth: np.ndarray,
                                   control: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted R² on deltas.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            control: Control/baseline expression values.
            weights: DEG-based weights for each gene.
            
        Returns:
            Weighted R² score on delta values.
        """
        delta_pred = pred - control
        delta_truth = truth - control
        return r2_score_on_deltas(delta_truth, delta_pred, weights)
    
    def _calculate_pearson_delta_direct(self, pred_deltas: np.ndarray, truth_deltas: np.ndarray) -> float:
        """Calculate Pearson correlation on pre-computed deltas.
        
        Args:
            pred_deltas: Pre-computed predicted delta values.
            truth_deltas: Pre-computed ground truth delta values.
            
        Returns:
            Pearson correlation coefficient of deltas.
        """
        corr, _ = pearsonr(pred_deltas, truth_deltas)
        return corr
    
    def _calculate_r2_delta_direct(self, pred_deltas: np.ndarray, truth_deltas: np.ndarray) -> float:
        """Calculate R² on pre-computed deltas.
        
        Args:
            pred_deltas: Pre-computed predicted delta values.
            truth_deltas: Pre-computed ground truth delta values.
            
        Returns:
            R² score on pre-computed delta values.
        """
        return r2_score_on_deltas(truth_deltas, pred_deltas)
    
    def _calculate_weighted_r2_delta_direct(self, pred_deltas: np.ndarray, truth_deltas: np.ndarray, 
                                          weights: np.ndarray) -> float:
        """Calculate weighted R² on pre-computed deltas.
        
        Args:
            pred_deltas: Pre-computed predicted delta values.
            truth_deltas: Pre-computed ground truth delta values.
            weights: DEG-based weights for each gene.
            
        Returns:
            Weighted R² score on pre-computed delta values.
        """
        return r2_score_on_deltas(truth_deltas, pred_deltas, weights)
    
    def _calculate_nir_scores(
        self, 
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate nir for all perturbations within their covariate groups.
        
        For each perturbation, measures the fraction of times its predicted profile
        is closer to its correct ground truth than to other perturbations' ground truths
        WITHIN THE SAME COVARIATE GROUP.
        
        Args:
            predictions: DataFrame with predicted expression profiles (cov_pert_key as index)
            ground_truth: DataFrame with ground truth expression profiles (cov_pert_key as index)
            
        Returns:
            Dict mapping cov_pert_key to nir score (0-1)
        """
        from scipy.spatial.distance import cdist
        
        nir_scores = {}
        
        # Group perturbations by covariate
        covariate_groups = {}
        for pert_key in predictions.index:
            covariate = pert_key.split('_')[0]
            if covariate not in covariate_groups:
                covariate_groups[covariate] = []
            covariate_groups[covariate].append(pert_key)
        
        # Calculate nir within each covariate group
        for covariate, pert_keys in covariate_groups.items():
            
            # Filter to only perturbations present in both predictions and ground truth
            valid_pert_keys = [pk for pk in pert_keys if pk in ground_truth.index]
            missing_pert_keys = [pk for pk in pert_keys if pk not in ground_truth.index]
            
            if missing_pert_keys:
                print(f"Warning: {len(missing_pert_keys)} perturbations not in ground truth for covariate {covariate}, skipping those")
            
            if len(valid_pert_keys) < 2:
                # Need at least 2 perturbations to calculate nir
                print(f"Skipping covariate {covariate}: only {len(valid_pert_keys)} valid perturbations (need ≥2)")
                continue
            
            # Get predictions and ground truths for valid perturbations only
            predictions_cov = predictions.loc[valid_pert_keys]
            ground_truth_cov = ground_truth.loc[valid_pert_keys]
            pert_keys = valid_pert_keys  # Use only valid keys for the rest of the calculation
            
            # Compute pairwise distance matrix for this covariate group
            distance_matrix = cdist(
                predictions_cov.values, 
                ground_truth_cov.values, 
                metric='euclidean'
            )
            
            # Calculate nir for each perturbation in this covariate
            for i, pert_key in tqdm(enumerate(pert_keys), desc="Calculating nir for covariate " + covariate):
                # Distance from this prediction to its correct ground truth
                correct_distance = distance_matrix[i, i]
                
                # Compare to all OTHER ground truths within same covariate
                comparisons = []
                for j in range(len(pert_keys)):
                    if i != j:  # Skip self-comparison
                        # Is prediction closer to correct GT than to this other GT?
                        comparisons.append(1 if correct_distance < distance_matrix[i, j] else 0)
                
                # Average across all comparisons
                nir_scores[pert_key] = np.mean(comparisons) if comparisons else 0.0
        return nir_scores 