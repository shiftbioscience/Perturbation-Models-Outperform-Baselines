"""
Utility functions and classes for CellSimBench.

Provides helper utilities for JSON serialization and other common operations.
"""

import json
from pathlib import Path
from typing import Any

class PathEncoder(json.JSONEncoder):
    """JSON encoder that handles Path objects.
    
    Extends the default JSON encoder to serialize pathlib.Path objects
    as strings, allowing configuration objects containing paths to be
    saved to JSON format.
    
    Example:
        >>> config = {'model_path': Path('/path/to/model')}
        >>> json.dumps(config, cls=PathEncoder)
    """
    
    def default(self, obj: Any) -> Any:
        """Override default method to handle Path objects.
        
        Args:
            obj: Object to serialize.
            
        Returns:
            String representation if obj is a Path, otherwise delegates
            to parent class.
        """
        if isinstance(obj, Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)
    