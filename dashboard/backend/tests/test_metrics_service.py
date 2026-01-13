"""
Unit tests for Metrics Service
"""

import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add dashboard backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics_service import MetricsService


class TestMetricsService(unittest.TestCase):
    """Test cases for MetricsService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories
        self.temp_output = tempfile.mkdtemp()
        self.temp_models = tempfile.mkdtemp()
        self.metrics_service = MetricsService(self.temp_output, self.temp_models)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_output, ignore_errors=True)
        shutil.rmtree(self.temp_models, ignore_errors=True)
    
    def test_initialization(self):
        """Test MetricsService initialization."""
        self.assertIsNotNone(self.metrics_service)
        self.assertEqual(str(self.metrics_service.output_dir), self.temp_output)
    
    def test_compute_prediction_metrics(self):
        """Test computing metrics from predictions."""
        actual = np.array([100, 200, 150, 180, 220])
        predicted = np.array([110, 190, 160, 175, 210])
        
        metrics = self.metrics_service.compute_prediction_metrics(actual, predicted)
        
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('mape', metrics)
        
        # Check values are reasonable
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['rmse'], 0)
        self.assertLessEqual(metrics['r2'], 1.0)
    
    def test_compute_metrics_with_nan(self):
        """Test metric computation handles NaN values."""
        actual = np.array([100, np.nan, 150, 180, np.nan])
        predicted = np.array([110, 190, np.nan, 175, 210])
        
        metrics = self.metrics_service.compute_prediction_metrics(actual, predicted)
        
        # Should still compute metrics after removing NaN
        self.assertIn('mae', metrics)
        self.assertGreater(len(metrics), 0)
    
    def test_get_metrics_summary_empty(self):
        """Test getting summary with no metrics file."""
        summary = self.metrics_service.get_metrics_summary()
        
        # Should return empty dict
        self.assertIsInstance(summary, dict)
    
    def test_get_live_metrics(self):
        """Test computing live metrics from dataframe."""
        df = pd.DataFrame({
            'actual_sales': [100, 200, 150, 180],
            'predicted_sales': [110, 190, 160, 175],
            'date': pd.date_range('2024-01-01', periods=4)
        })
        
        metrics = self.metrics_service.get_live_metrics(df)
        
        self.assertIn('total_predictions', metrics)
        self.assertEqual(metrics['total_predictions'], 4)
        self.assertIn('timestamp', metrics)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Populate cache
        self.metrics_service._metrics_cache['test'] = 'data'
        
        # Clear cache
        self.metrics_service.clear_cache()
        
        # Verify cache is empty
        self.assertEqual(len(self.metrics_service._metrics_cache), 0)


if __name__ == '__main__':
    unittest.main()
