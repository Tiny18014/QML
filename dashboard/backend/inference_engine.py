"""
Live Inference Engine
Handles real-time predictions using loaded models
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import sys

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Add scripts to path for feature engineering
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR / "scripts"))

from model_loader import ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Engine for running live predictions with trained models.
    Supports batch and single predictions.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the inference engine.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.model_loader = ModelLoader(models_dir)
        self._loaded_models = {}
        
        # Try to import feature engineering functions
        try:
            from advanced_model_trainer import create_advanced_features, prepare_data_for_training
            self.create_advanced_features = create_advanced_features
            self.prepare_data_for_training = prepare_data_for_training
        except ImportError:
            logger.warning("Could not import feature engineering functions")
            self.create_advanced_features = None
            self.prepare_data_for_training = None
    
    def predict(self, 
                df: pd.DataFrame, 
                category: str,
                use_cache: bool = True) -> pd.DataFrame:
        """
        Generate predictions for input data.
        
        Args:
            df: Input dataframe with features
            category: Vehicle category to predict for
            use_cache: Whether to use cached models
            
        Returns:
            DataFrame with predictions
        """
        try:
            # Load appropriate model
            model_data = self._get_model_for_category(category, use_cache)
            
            if not model_data:
                logger.error(f"No model available for category: {category}")
                return pd.DataFrame()
            
            # Prepare features
            df_prepared = self._prepare_features(df, category)
            
            if df_prepared.empty:
                logger.error("Feature preparation failed")
                return pd.DataFrame()
            
            # Run prediction based on model type
            predictions = self._run_prediction(model_data, df_prepared)
            
            # Add predictions to dataframe
            df_result = df.copy()
            df_result['Predicted_Sales'] = np.maximum(0, predictions).astype(int)
            
            return df_result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _get_model_for_category(self, category: str, use_cache: bool) -> Optional[Dict]:
        """Load the appropriate model for a vehicle category."""
        # Check cache first
        if use_cache and category in self._loaded_models:
            return self._loaded_models[category]
        
        # Determine model filename
        category_models = self.model_loader.get_category_models()
        
        if category in category_models:
            model_file = category_models[category]
        else:
            # Try to construct filename
            cat_filename = category.replace(" ", "_").replace("/", "_")
            model_file = f"advanced_model_{cat_filename}.pkl"
        
        # Load model
        model_data = self.model_loader.load_model(model_file)
        
        if model_data and use_cache:
            self._loaded_models[category] = model_data
        
        return model_data
    
    def _prepare_features(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Prepare features for prediction."""
        if self.create_advanced_features is None:
            logger.warning("Feature engineering not available, using raw features")
            return df
        
        try:
            # Create advanced features
            df_featured = self.create_advanced_features(df.copy())
            
            # Filter by category
            if 'Vehicle_Category' in df_featured.columns:
                df_featured = df_featured[df_featured['Vehicle_Category'] == category]
            
            return df_featured
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return pd.DataFrame()
    
    def _run_prediction(self, model_data: Dict, df: pd.DataFrame) -> np.ndarray:
        """Run prediction with the loaded model."""
        model = model_data.get('primary_model') or model_data.get('model')
        
        if model is None:
            raise ValueError("No model found in model_data")
        
        # Get feature names
        feature_names = model_data.get('feature_names', [])
        
        if feature_names:
            # Prepare feature matrix
            if self.prepare_data_for_training:
                X_scaled, _, _, _ = self.prepare_data_for_training(
                    df, 
                    feature_subset=feature_names
                )
            else:
                # Manual feature selection
                available_features = [f for f in feature_names if f in df.columns]
                X = df[available_features].values
                
                # Scale if scaler available
                scaler = model_data.get('scaler')
                if scaler:
                    X_scaled = scaler.transform(X)
                else:
                    X_scaled = X
        else:
            # Use all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            X_scaled = df[numeric_cols].values
        
        # Run prediction
        predictions = model.predict(X_scaled)
        
        return predictions
    
    def batch_predict(self, 
                     df: pd.DataFrame,
                     categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run predictions for multiple categories.
        
        Args:
            df: Input dataframe
            categories: List of categories to predict, or None for all
            
        Returns:
            DataFrame with predictions for all categories
        """
        if categories is None:
            if 'Vehicle_Category' in df.columns:
                categories = df['Vehicle_Category'].unique().tolist()
            else:
                logger.error("No categories specified and none found in data")
                return pd.DataFrame()
        
        all_predictions = []
        
        for category in categories:
            logger.info(f"Running predictions for {category}")
            
            # Filter data for category
            df_cat = df[df['Vehicle_Category'] == category].copy()
            
            if df_cat.empty:
                continue
            
            # Predict
            df_pred = self.predict(df_cat, category)
            
            if not df_pred.empty:
                all_predictions.append(df_pred)
        
        if all_predictions:
            return pd.concat(all_predictions, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_model_info(self, category: str) -> Dict[str, Any]:
        """
        Get information about the model for a category.
        
        Args:
            category: Vehicle category
            
        Returns:
            Dictionary with model information
        """
        model_data = self._get_model_for_category(category, use_cache=True)
        
        if not model_data:
            return {}
        
        info = {
            'category': category,
            'type': model_data.get('type', 'unknown'),
            'path': model_data.get('path', ''),
        }
        
        # Add feature info if available
        if 'feature_names' in model_data:
            info['num_features'] = len(model_data['feature_names'])
            info['features'] = model_data['feature_names']
        
        return info
    
    def clear_cache(self):
        """Clear cached models."""
        self._loaded_models.clear()
        self.model_loader.clear_cache()
        logger.info("Inference engine cache cleared")
