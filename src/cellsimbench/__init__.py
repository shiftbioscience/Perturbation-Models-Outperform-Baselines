"""
CellSimBench: A flexible benchmarking framework for perturbation response prediction models.
"""

__version__ = "0.1.0"
__author__ = "CellSimBench Team"

# Core imports
from cellsimbench.core.benchmark import BenchmarkRunner
from cellsimbench.core.data_manager import DataManager
from cellsimbench.core.metrics_engine import MetricsEngine

__all__ = [
    "BenchmarkRunner",
    "DataManager", 
    "MetricsEngine",
] 