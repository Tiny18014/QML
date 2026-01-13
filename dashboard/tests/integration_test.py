"""
Integration test to verify dashboard components work together
"""

import sys
from pathlib import Path

# Block torch to avoid installation issues
sys.modules['torch'] = None

# Add paths
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "dashboard" / "backend"))

def test_model_loader():
    """Test model loader can be imported and initialized."""
    from model_loader import ModelLoader
    
    ml = ModelLoader(str(ROOT_DIR / "models"))
    models = ml.get_available_models()
    
    print(f"✅ ModelLoader: Found {len(models)} models")
    if models:
        print(f"   Sample models: {models[:3]}")
    
    # Test category models
    cat_models = ml.get_category_models()
    print(f"✅ Category models: {len(cat_models)} categories")
    
    return True

def test_metrics_service():
    """Test metrics service."""
    from metrics_service import MetricsService
    
    ms = MetricsService(
        str(ROOT_DIR / "output"),
        str(ROOT_DIR / "models")
    )
    
    summary = ms.get_metrics_summary()
    print(f"✅ MetricsService: Summary has {len(summary)} keys")
    
    # Test metric computation
    import numpy as np
    actual = np.array([100, 200, 150])
    predicted = np.array([110, 190, 160])
    
    metrics = ms.compute_prediction_metrics(actual, predicted)
    print(f"✅ Metric computation: {list(metrics.keys())}")
    
    return True

def test_data_refresh():
    """Test data refresh utilities."""
    sys.path.insert(0, str(ROOT_DIR / "dashboard" / "frontend" / "utils"))
    
    from data_refresh import DataRefreshManager
    
    drm = DataRefreshManager(
        str(ROOT_DIR / "models"),
        str(ROOT_DIR / "output"),
        str(ROOT_DIR / "data"),
        str(ROOT_DIR / "predictions")
    )
    
    updates = drm.check_for_updates()
    print(f"✅ DataRefreshManager: {len(updates)} update keys")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Dashboard Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Model Loader", test_model_loader),
        ("Metrics Service", test_metrics_service),
        ("Data Refresh", test_data_refresh),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * 60)
        try:
            if test_func():
                passed += 1
                print(f"✅ {name} PASSED")
            else:
                failed += 1
                print(f"❌ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
