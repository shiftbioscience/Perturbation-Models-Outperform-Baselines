"""
DEG quantile analysis helper for perturbation strength stratification.

Provides tools for analyzing model performance stratified by perturbation strength
as measured by DEG count.
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from .data_manager import DataManager

log = logging.getLogger(__name__)


class DEGQuantileAnalyzer:
    """Helper class for assigning perturbations to quantiles based on DEG count.
    
    Used to stratify perturbations by strength (number of DEGs) for performance
    analysis across different perturbation magnitudes.
    
    Attributes:
        data_manager: DataManager instance for accessing DEG information.
        
    Example:
        >>> analyzer = DEGQuantileAnalyzer(data_manager)
        >>> quantiles = analyzer.assign_perturbations_to_quantiles(perts, 'donor1', 10)
    """
    
    def __init__(self, data_manager: DataManager) -> None:
        """Initialize DEGQuantileAnalyzer with DataManager.
        
        Args:
            data_manager: DataManager for accessing DEG masks and counts.
        """
        self.data_manager = data_manager
    
    def assign_perturbations_to_quantiles(self, perturbations: List[str], 
                                        covariate: str, n_quantiles: int = 10) -> Dict[str, int]:
        """Assign perturbations to quantiles based on DEG count.
        
        Args:
            perturbations: List of perturbation names for this covariate.
            covariate: Covariate value to use for DEG calculation.
            n_quantiles: Number of quantiles to create.
            
        Returns:
            Dictionary mapping perturbation names to quantile indices (0-based).
        """
        # Calculate DEG counts for each perturbation
        deg_counts: Dict[str, int] = {}
        
        for perturbation in perturbations:
            try:
                # Get DEG mask for this covariate-perturbation combination
                deg_mask = self.data_manager.get_deg_mask(covariate, perturbation)
                deg_counts[perturbation] = int(deg_mask.sum())
                
            except Exception as e:
                log.warning(f"Could not get DEG count for {covariate}_{perturbation}: {e}")
                # Assign to lowest quantile if DEG info is missing
                deg_counts[perturbation] = 0
        
        if not deg_counts:
            log.warning(f"No DEG counts calculated for covariate {covariate}")
            return {}
        
        # Sort perturbations by DEG count
        sorted_perturbations = sorted(deg_counts.keys(), key=lambda x: deg_counts[x])
        
        # Assign to quantiles
        quantile_assignments = {}
        n_perts = len(sorted_perturbations)
        
        for i, perturbation in enumerate(sorted_perturbations):
            # Calculate quantile index (0-based)
            quantile_idx = min(int(i * n_quantiles / n_perts), n_quantiles - 1)
            quantile_assignments[perturbation] = quantile_idx
        
        # Log quantile distribution
        self._log_quantile_distribution(quantile_assignments, deg_counts, covariate, n_quantiles)
        
        return quantile_assignments
    
    def _log_quantile_distribution(self, quantile_assignments: Dict[str, int], 
                                 deg_counts: Dict[str, int], covariate: str, n_quantiles: int) -> None:
        """Log the distribution of perturbations across quantiles.
        
        Args:
            quantile_assignments: Dictionary mapping perturbations to quantiles.
            deg_counts: Dictionary mapping perturbations to DEG counts.
            covariate: Covariate value for logging.
            n_quantiles: Number of quantiles.
        """
        log.info(f"DEG quantile distribution for {covariate}:")
        for q in range(n_quantiles):
            perts_in_quantile = [p for p, qidx in quantile_assignments.items() if qidx == q]
            
            if perts_in_quantile:
                deg_range = [deg_counts[p] for p in perts_in_quantile]
                min_degs = min(deg_range)
                max_degs = max(deg_range)
                
                log.info(f"  Q{q+1}: {len(perts_in_quantile)} perturbations, "
                        f"DEG range: {min_degs}-{max_degs}")
            else:
                log.info(f"  Q{q+1}: 0 perturbations")
    
    def get_quantile_statistics(self, quantile_assignments: Dict[str, int], 
                              deg_counts: Dict[str, int]) -> Dict[int, Dict[str, float]]:
        """Calculate statistics for each quantile.
        
        Args:
            quantile_assignments: Dictionary mapping perturbations to quantile indices.
            deg_counts: Dictionary mapping perturbations to DEG counts.
            
        Returns:
            Dictionary mapping quantile indices to statistics including
            mean_degs, median_degs, n_perts, min_degs, and max_degs.
        """
        quantile_stats: Dict[int, Dict[str, float]] = {}
        
        # Group perturbations by quantile
        quantiles = {}
        for pert, q_idx in quantile_assignments.items():
            if q_idx not in quantiles:
                quantiles[q_idx] = []
            quantiles[q_idx].append(pert)
        
        # Calculate stats for each quantile
        for q_idx, perts in quantiles.items():
            deg_values = [deg_counts[p] for p in perts if p in deg_counts]
            
            if deg_values:
                quantile_stats[q_idx] = {
                    'mean_degs': np.mean(deg_values),
                    'median_degs': np.median(deg_values),
                    'n_perts': len(perts),
                    'min_degs': min(deg_values),
                    'max_degs': max(deg_values)
                }
            else:
                quantile_stats[q_idx] = {
                    'mean_degs': 0,
                    'median_degs': 0,
                    'n_perts': 0,
                    'min_degs': 0,
                    'max_degs': 0
                }
        
        return quantile_stats 