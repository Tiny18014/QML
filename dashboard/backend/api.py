"""
API Endpoints for Dashboard Backend
Optional Flask API for serving forecasting data
"""

from flask import Flask, jsonify, request
from pathlib import Path
import sys
import logging

# Add parent directories to path
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR / "scripts"))
sys.path.append(str(ROOT_DIR / "dashboard" / "backend"))

from model_loader import ModelLoader
from inference_engine import InferenceEngine
from metrics_service import MetricsService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize services
models_dir = ROOT_DIR / "models"
output_dir = ROOT_DIR / "output"

model_loader = ModelLoader(str(models_dir))
inference_engine = InferenceEngine(str(models_dir))
metrics_service = MetricsService(str(output_dir), str(models_dir))


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'EV Forecasting Dashboard API',
        'version': '1.0.0'
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    try:
        models = model_loader.get_available_models()
        category_models = model_loader.get_category_models()
        
        return jsonify({
            'success': True,
            'models': models,
            'category_models': category_models,
            'total': len(models)
        })
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/models/<model_name>', methods=['GET'])
def get_model_info(model_name):
    """Get information about a specific model."""
    try:
        metadata = model_loader.get_model_metadata(model_name)
        
        if not metadata:
            return jsonify({
                'success': False,
                'error': 'Model not found'
            }), 404
        
        return jsonify({
            'success': True,
            'metadata': metadata
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics."""
    try:
        summary = metrics_service.get_metrics_summary()
        
        return jsonify({
            'success': True,
            'metrics': summary
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/metrics/<category>', methods=['GET'])
def get_category_metrics(category):
    """Get metrics for a specific category."""
    try:
        metrics = metrics_service.get_category_metrics(category)
        
        if not metrics:
            return jsonify({
                'success': False,
                'error': 'Category metrics not found'
            }), 404
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        logger.error(f"Error getting category metrics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Run prediction endpoint.
    Expects JSON with 'data' (list of records) and 'category' fields.
    """
    try:
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({
                'success': False,
                'error': 'Missing data field'
            }), 400
        
        import pandas as pd
        
        # Convert data to DataFrame
        df = pd.DataFrame(request_data['data'])
        category = request_data.get('category')
        
        if not category:
            return jsonify({
                'success': False,
                'error': 'Missing category field'
            }), 400
        
        # Run prediction
        predictions_df = inference_engine.predict(df, category)
        
        if predictions_df.empty:
            return jsonify({
                'success': False,
                'error': 'Prediction failed'
            }), 500
        
        # Convert to JSON
        result = predictions_df.to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'predictions': result,
            'count': len(result)
        })
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all caches."""
    try:
        model_loader.clear_cache()
        inference_engine.clear_cache()
        metrics_service.clear_cache()
        
        return jsonify({
            'success': True,
            'message': 'All caches cleared'
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def run_api(host='0.0.0.0', port=5000, debug=False):
    """
    Run the Flask API server.
    
    Args:
        host: Host to bind to
        port: Port to run on
        debug: Enable debug mode
    """
    logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_api(debug=True)
