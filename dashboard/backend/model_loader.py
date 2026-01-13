"""
Model Deserialization Service
Handles loading and caching of trained models from the CI/CD pipeline
"""

import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import numpy as np

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Service for loading and managing ML models.
    Supports XGBoost, LightGBM, PyTorch, and River models.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model loader.
        
        Args:
            models_dir: Directory where models are stored
        """
        self.models_dir = Path(models_dir)
        self._model_cache = {}
        self._cache_timestamps = {}
        
    def get_available_models(self) -> List[str]:
        """Get list of available model files."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory {self.models_dir} does not exist")
            return []
        
        model_files = []
        for ext in ['*.pkl', '*.pth', '*.joblib']:
            model_files.extend([f.name for f in self.models_dir.glob(ext)])
        
        return sorted(model_files)
    
    def load_model(self, model_name: str, force_reload: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load a model from disk with caching support.
        
        Args:
            model_name: Name of the model file
            force_reload: If True, bypass cache and reload from disk
            
        Returns:
            Dictionary containing model and metadata, or None if loading fails
        """
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        # Check cache
        if not force_reload and model_name in self._model_cache:
            cached_time = self._cache_timestamps.get(model_name)
            file_mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
            
            if cached_time and cached_time >= file_mtime:
                logger.info(f"Using cached model: {model_name}")
                return self._model_cache[model_name]
        
        # Load model based on file extension
        try:
            if model_name.endswith('.pth'):
                model_data = self._load_pytorch_model(model_path)
            elif model_name.endswith('.pkl'):
                model_data = self._load_pickle_model(model_path)
            elif model_name.endswith('.joblib'):
                model_data = self._load_joblib_model(model_path)
            else:
                logger.error(f"Unsupported model format: {model_name}")
                return None
            
            # Update cache
            self._model_cache[model_name] = model_data
            self._cache_timestamps[model_name] = datetime.now()
            
            logger.info(f"Successfully loaded model: {model_name}")
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def _load_pickle_model(self, model_path: Path) -> Dict[str, Any]:
        """Load a pickle model file."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Standardize format
        if isinstance(model_data, dict):
            return model_data
        else:
            # Wrap single model in dict
            return {
                'model': model_data,
                'type': 'pickle',
                'path': str(model_path)
            }
    
    def _load_joblib_model(self, model_path: Path) -> Dict[str, Any]:
        """Load a joblib model file."""
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict):
            return model_data
        else:
            return {
                'model': model_data,
                'type': 'joblib',
                'path': str(model_path)
            }
    
    def _load_pytorch_model(self, model_path: Path) -> Dict[str, Any]:
        """Load a PyTorch model file."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install torch to load .pth models.")
        
        # Try to load state dict
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            return {
                'state_dict': state_dict,
                'type': 'pytorch',
                'path': str(model_path)
            }
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            raise
    
    def load_normalization_params(self, params_file: str = "normalization_params.pkl") -> Optional[Dict]:
        """Load normalization parameters for QML models."""
        params_path = self.models_dir / params_file
        
        if not params_path.exists():
            logger.warning(f"Normalization params not found: {params_path}")
            return None
        
        try:
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
            logger.info(f"Loaded normalization params from {params_file}")
            return params
        except Exception as e:
            logger.error(f"Error loading normalization params: {e}")
            return None
    
    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        Get metadata about a model file.
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Dictionary with metadata (size, modified time, etc.)
        """
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            return {}
        
        stat = model_path.stat()
        return {
            'name': model_name,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'path': str(model_path)
        }
    
    def clear_cache(self):
        """Clear the model cache."""
        self._model_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Model cache cleared")
    
    def get_category_models(self) -> Dict[str, str]:
        """
        Get mapping of vehicle categories to their model files.
        
        Returns:
            Dictionary mapping category names to model file names
        """
        category_models = {}
        
        for model_file in self.get_available_models():
            if model_file.startswith('advanced_model_'):
                # Extract category from filename
                category = model_file.replace('advanced_model_', '').replace('.pkl', '')
                category = category.replace('_', '-')
                category_models[category] = model_file
            elif model_file.startswith('specialized_'):
                if '3w' in model_file.lower():
                    category_models['3-Wheelers'] = model_file
                elif 'bus' in model_file.lower():
                    category_models['Bus'] = model_file
        
        return category_models
