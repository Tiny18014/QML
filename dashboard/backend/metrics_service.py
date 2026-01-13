"""
Metrics Service
Handles loading, caching, and processing of model performance metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsService:
    """
    Service for managing model performance metrics.
    Supports loading from CSV, JSON, and computing real-time metrics.
    """
    
    def __init__(self, output_dir: str = "output", models_dir: str = "models"):
        """
        Initialize the metrics service.
        
        Args:
            output_dir: Directory containing metrics files
            models_dir: Directory containing model files
        """
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        self._metrics_cache = {}
        
    def load_model_metrics(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load model evaluation metrics from CSV file.
        
        Args:
            force_reload: Force reload from disk
            
        Returns:
            DataFrame with model metrics
        """
        cache_key = 'model_evaluation'
        
        if not force_reload and cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]
        
        # Try multiple possible filenames
        possible_files = [
            'model_evaluation_summary_final.csv',
            'model_evaluation_summary.csv',
            'model_metrics.csv'
        ]
        
        for filename in possible_files:
            file_path = Path('.') / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    self._metrics_cache[cache_key] = df
                    logger.info(f"Loaded metrics from {filename}")
                    return df
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        # Try output directory
        for filename in possible_files:
            file_path = self.output_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    self._metrics_cache[cache_key] = df
                    logger.info(f"Loaded metrics from {file_path}")
                    return df
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.warning("No model metrics file found")
        return pd.DataFrame()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of model performance metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        df = self.load_model_metrics()
        
        if df.empty:
            return {}
        
        summary = {
            'total_models': len(df),
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract key metrics if columns exist
        metric_columns = ['MAE', 'RMSE', 'R2', 'MAPE']
        
        for col in metric_columns:
            if col in df.columns:
                summary[f'avg_{col.lower()}'] = df[col].mean()
                summary[f'best_{col.lower()}'] = df[col].min() if col != 'R2' else df[col].max()
        
        return summary
    
    def get_category_metrics(self, category: str) -> Dict[str, Any]:
        """
        Get metrics for a specific vehicle category.
        
        Args:
            category: Vehicle category name
            
        Returns:
            Dictionary with category metrics
        """
        df = self.load_model_metrics()
        
        if df.empty or 'Category' not in df.columns:
            return {}
        
        # Filter by category
        df_cat = df[df['Category'].str.contains(category, case=False, na=False)]
        
        if df_cat.empty:
            return {}
        
        metrics = {
            'category': category,
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract metrics
        metric_columns = ['MAE', 'RMSE', 'R2', 'MAPE']
        for col in metric_columns:
            if col in df_cat.columns:
                metrics[col.lower()] = df_cat[col].iloc[0]
        
        return metrics
    
    def compute_prediction_metrics(self, 
                                   actual: np.ndarray, 
                                   predicted: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics for predictions.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with computed metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Ensure arrays are valid
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {}
        
        metrics = {
            'mae': float(mean_absolute_error(actual, predicted)),
            'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
            'r2': float(r2_score(actual, predicted)),
        }
        
        # MAPE
        mape_mask = actual != 0
        if mape_mask.sum() > 0:
            mape = np.mean(np.abs((actual[mape_mask] - predicted[mape_mask]) / actual[mape_mask])) * 100
            metrics['mape'] = float(mape)
        
        return metrics
    
    def get_prediction_history(self, limit: int = 100) -> pd.DataFrame:
        """
        Get historical prediction data if available.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with prediction history
        """
        # Look for prediction files
        pred_files = list(self.output_dir.glob("*predictions*.csv"))
        
        if not pred_files:
            return pd.DataFrame()
        
        # Load most recent
        pred_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        try:
            df = pd.read_csv(pred_files[0])
            
            if limit and len(df) > limit:
                df = df.tail(limit)
            
            return df
        except Exception as e:
            logger.error(f"Error loading prediction history: {e}")
            return pd.DataFrame()
    
    def export_metrics(self, filepath: str, format: str = 'json') -> bool:
        """
        Export metrics to file.
        
        Args:
            filepath: Output file path
            format: Export format ('json' or 'csv')
            
        Returns:
            True if successful
        """
        summary = self.get_metrics_summary()
        
        if not summary:
            return False
        
        try:
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(summary, f, indent=2)
            elif format == 'csv':
                df = self.load_model_metrics()
                df.to_csv(filepath, index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return False
    
    def get_live_metrics(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute live metrics from a predictions dataframe.
        
        Args:
            predictions_df: DataFrame with 'actual' and 'predicted' columns
            
        Returns:
            Dictionary with live metrics
        """
        if predictions_df.empty:
            return {}
        
        metrics = {
            'total_predictions': len(predictions_df),
            'timestamp': datetime.now().isoformat()
        }
        
        # Check for actual vs predicted columns
        actual_col = None
        predicted_col = None
        
        for col in predictions_df.columns:
            if 'actual' in col.lower():
                actual_col = col
            if 'predicted' in col.lower() or 'pred' in col.lower():
                predicted_col = col
        
        if actual_col and predicted_col:
            metrics.update(self.compute_prediction_metrics(
                predictions_df[actual_col].values,
                predictions_df[predicted_col].values
            ))
        
        # Add summary statistics
        if predicted_col:
            metrics['total_predicted_sales'] = int(predictions_df[predicted_col].sum())
            metrics['avg_predicted_sales'] = float(predictions_df[predicted_col].mean())
        
        return metrics
    
    def clear_cache(self):
        """Clear metrics cache."""
        self._metrics_cache.clear()
        logger.info("Metrics cache cleared")
