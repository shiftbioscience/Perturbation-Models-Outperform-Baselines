"""
Built-in models for CellSimBench.

Note: Most baseline functionality is handled directly by BaselineRunner.
This module is kept for any future non-baseline built-in models.
"""

from typing import Dict, List
import numpy as np
import scanpy as sc
from .base import BuiltinModel


# Registry of built-in models (currently empty - baselines handled by BaselineRunner)
BUILTIN_MODELS = {}


def get_builtin_model(model_name: str):
    """Get a built-in model by name."""
    if model_name not in BUILTIN_MODELS:
        raise ValueError(f"Unknown built-in model: {model_name}. Available: {list(BUILTIN_MODELS.keys())}")
    return BUILTIN_MODELS[model_name] 