"""
Unit tests for Model Loader
"""

import unittest
import tempfile
import pickle
from pathlib import Path
import sys

# Add dashboard backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_loader import ModelLoader


class TestModelLoader(unittest.TestCase):
    """Test cases for ModelLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test models
        self.temp_dir = tempfile.mkdtemp()
        self.model_loader = ModelLoader(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ModelLoader initialization."""
        self.assertIsNotNone(self.model_loader)
        self.assertEqual(str(self.model_loader.models_dir), self.temp_dir)
        self.assertIsInstance(self.model_loader._model_cache, dict)
    
    def test_get_available_models_empty(self):
        """Test getting models from empty directory."""
        models = self.model_loader.get_available_models()
        self.assertEqual(len(models), 0)
    
    def test_load_pickle_model(self):
        """Test loading a pickle model."""
        # Create a dummy model
        model_data = {'model': 'test_model', 'version': '1.0'}
        model_path = Path(self.temp_dir) / 'test_model.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Load the model
        loaded = self.model_loader.load_model('test_model.pkl')
        
        self.assertIsNotNone(loaded)
        self.assertIn('model', loaded)
    
    def test_model_caching(self):
        """Test that models are cached correctly."""
        # Create a dummy model
        model_data = {'model': 'cached_model'}
        model_path = Path(self.temp_dir) / 'cached.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Load twice
        loaded1 = self.model_loader.load_model('cached.pkl')
        loaded2 = self.model_loader.load_model('cached.pkl')
        
        # Should use cache
        self.assertIn('cached.pkl', self.model_loader._model_cache)
    
    def test_get_model_metadata(self):
        """Test getting model metadata."""
        # Create a dummy model
        model_path = Path(self.temp_dir) / 'metadata_test.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)
        
        metadata = self.model_loader.get_model_metadata('metadata_test.pkl')
        
        self.assertIn('name', metadata)
        self.assertIn('size_mb', metadata)
        self.assertEqual(metadata['name'], 'metadata_test.pkl')
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Create and load a model
        model_path = Path(self.temp_dir) / 'cache_test.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)
        
        self.model_loader.load_model('cache_test.pkl')
        
        # Verify cache has data
        self.assertGreater(len(self.model_loader._model_cache), 0)
        
        # Clear cache
        self.model_loader.clear_cache()
        
        # Verify cache is empty
        self.assertEqual(len(self.model_loader._model_cache), 0)


if __name__ == '__main__':
    unittest.main()
