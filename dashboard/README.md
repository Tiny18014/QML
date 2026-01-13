# EV Demand Forecasting Dashboard

## Overview

The EV Demand Intelligence Hub is a comprehensive, real-time dashboard for visualizing and analyzing electric vehicle sales forecasts. It integrates machine learning models, live inference capabilities, and interactive visualizations to support strategic decision-making.

## Features

### 🎯 Core Capabilities

1. **Live Inference Engine**
   - Automatic loading of latest models from CI/CD pipeline
   - Support for multiple model types (XGBoost, LightGBM, PyTorch)
   - Model caching and version tracking
   - Real-time prediction generation

2. **Interactive Visualizations**
   - **Sentiment Gauges**: Real-time market sentiment indicators with color-coded zones
   - **Volatility Plots**: Time-series volatility analysis with interactive controls
   - **Brand-Specific Trends**: Multi-category comparison and forecast visualization
   - **Market Health Monitoring**: KPI cards, alerts, and drill-down analysis

3. **Auto-Refresh Mechanism**
   - Configurable refresh intervals
   - Automatic detection of new model artifacts
   - Manual refresh controls

4. **AI-Powered Insights**
   - Integration with Agno agent for strategic summaries
   - Correlation analysis between sentiment and demand
   - Automated alert generation

### 📊 Dashboard Components

#### Backend Services (`dashboard/backend/`)

- **`model_loader.py`**: Model deserialization and caching service
- **`inference_engine.py`**: Live inference with batch prediction support
- **`metrics_service.py`**: Performance metrics loading and computation
- **`api.py`**: Optional Flask API for programmatic access

#### Frontend Components (`dashboard/frontend/`)

- **`components/sentiment_gauge.py`**: Sentiment visualization with gauges and trends
- **`components/volatility_plot.py`**: Volatility analysis with heatmaps
- **`components/trend_lines.py`**: Multi-brand trend comparison
- **`components/market_health.py`**: Market health scoring and alerts
- **`utils/data_refresh.py`**: Auto-refresh utilities

## Installation

### Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tiny18014/QML.git
   cd QML
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model files**
   ```bash
   ls models/
   # Should see: advanced_model_*.pkl, specialized_*.pkl, etc.
   ```

## Usage

### Running the Enhanced Dashboard

#### Option 1: Enhanced Dashboard (Recommended)

```bash
streamlit run dashboard/frontend/app.py
```

This launches the full-featured dashboard with all visualization components.

#### Option 2: Legacy Dashboard

```bash
streamlit run scripts/streamlit_dashboard.py
```

This launches the original dashboard for backward compatibility.

### Running the API Server (Optional)

```bash
python dashboard/backend/api.py
```

API will be available at `http://localhost:5000`

### API Endpoints

- `GET /api/health` - Health check
- `GET /api/models` - List available models
- `GET /api/models/<model_name>` - Get model metadata
- `GET /api/metrics` - Get performance metrics
- `POST /api/predict` - Run predictions
- `POST /api/cache/clear` - Clear caches

## Configuration

Edit `dashboard/config/dashboard_config.yaml` to customize:

- Data directories
- Model settings
- Visualization preferences
- Refresh intervals
- Alert thresholds

## Architecture

### Data Flow

```
CI/CD Pipeline (GitHub Actions)
    ↓
Model Artifacts (models/*.pkl)
    ↓
Model Loader (Backend)
    ↓
Inference Engine
    ↓
Visualization Components (Frontend)
    ↓
Interactive Dashboard
```

### Component Interaction

```
User Interface (Streamlit)
    ↓
Data Refresh Manager → Model Loader → Inference Engine
    ↓                       ↓              ↓
Updates Available?    Load Models    Generate Predictions
    ↓                       ↓              ↓
Auto Refresh          Cache Models   Return Results
```

## Dashboard Sections

### 1. Overview Tab

- AI-powered strategic insights
- Quick visualizations
- On-demand forecasting tool
- Export functionality

### 2. Sentiment Analysis Tab

- Current sentiment gauge (0-100 scale)
- Historical sentiment trends
- Sentiment interpretation
- Key indicators

### 3. Volatility & Risk Tab

- Time-series volatility plots
- Volatility heatmaps by region/category
- High volatility event detection
- Configurable rolling windows

### 4. Trends & Forecasts Tab

- Multi-brand comparison charts
- Growth rate analysis
- Market share visualization
- Forecast vs actual comparison

### 5. Market Health Tab

- Overall health score (0-100)
- KPI cards (sales, coverage, diversity)
- Automated alerts
- Drill-down analysis

## Model Management

### Supported Model Types

1. **XGBoost/LightGBM Models** (`.pkl` files)
   - Category-specific models: `advanced_model_<category>.pkl`
   - Specialized models: `specialized_3w_monthly_model.pkl`, `specialized_bus_monthly_model.pkl`

2. **PyTorch Models** (`.pth` files)
   - Hybrid models: `ev_sales_hybrid_model_simple.pth`
   - Requires: `normalization_params.pkl`

3. **River Models** (Future support)
   - Hoeffding Tree ensembles
   - Incremental learning models

### Model Loading Priority

1. Cache (if available and up-to-date)
2. Disk (from `models/` directory)
3. Fallback to previous version

### Version Tracking

- Models are timestamped
- Automatic rollback on load failure
- Version metadata displayed in UI

## Performance Optimization

### Caching Strategy

- **Model Cache**: Persist loaded models in memory
- **Prediction Cache**: Cache recent predictions
- **Metrics Cache**: Cache computed metrics

### Lazy Loading

- Components load only when tab is activated
- Large datasets paginated
- Visualizations rendered on-demand

### Target Performance

- **Dashboard Load**: < 2 seconds (initial)
- **Prediction Generation**: < 5 seconds (batch)
- **Visualization Render**: < 1 second
- **Auto-Refresh**: Non-blocking, background

## Data Requirements

### Input Data Format

Predictions DataFrame should contain:
- `Date`: datetime
- `State`: string
- `Vehicle_Category`: string
- `Predicted_Sales`: integer
- `EV_Sales_Quantity`: integer (optional, for actual values)

### Sample Data

Located in:
- `data/EV_Dataset.csv`: Historical data
- `output/daily_predictions_2026.csv`: Latest predictions

## Troubleshooting

### Common Issues

1. **"No models found"**
   - Check `models/` directory exists
   - Verify `.pkl` files are present
   - Run CI/CD pipeline to generate models

2. **"Import error"**
   - Ensure all dependencies installed: `pip install -r requirements.txt`
   - Check Python version (3.8+)

3. **"No data available"**
   - Click "Load Latest Predictions" in sidebar
   - Check `output/` directory for prediction files
   - Verify data files are not corrupted

4. **"Agent unavailable"**
   - Check HF_TOKEN in Streamlit secrets
   - Verify API quota not exceeded
   - Fallback metrics will be shown

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Testing

### Unit Tests

```bash
# Test backend components
python -m pytest dashboard/backend/tests/

# Test frontend components  
python -m pytest dashboard/frontend/tests/
```

### Manual Testing

1. **Model Loading**
   ```bash
   python -c "from dashboard.backend.model_loader import ModelLoader; ml = ModelLoader(); print(ml.get_available_models())"
   ```

2. **Inference**
   ```bash
   python scripts/test_ondemand_forecast.py
   ```

3. **API**
   ```bash
   curl http://localhost:5000/api/health
   ```

## Security Considerations

1. **API Security**
   - No authentication in current version (local use only)
   - For production: Add API keys, rate limiting

2. **Data Privacy**
   - No PII in default datasets
   - Secrets managed via Streamlit secrets

3. **Model Integrity**
   - Models signed by CI/CD pipeline
   - Checksum verification (future)

## CI/CD Integration

Dashboard automatically detects new models from the CI/CD pipeline:

1. Pipeline trains models weekly
2. Models committed to `models/` directory
3. Dashboard detects file changes
4. Auto-refresh loads new models
5. New predictions generated

## Extending the Dashboard

### Adding New Visualizations

1. Create component file in `dashboard/frontend/components/`
2. Implement render function
3. Import in `dashboard/frontend/app.py`
4. Add to tabs in `render_main_content()`

### Adding New Metrics

1. Update `metrics_service.py` with new computation
2. Update `market_health.py` to display metric
3. Add to configuration YAML

### Custom Models

1. Save model in `models/` directory
2. Add loading logic in `model_loader.py`
3. Update `inference_engine.py` for predictions

## License

See repository LICENSE file.

## Support

For issues and questions:
- GitHub Issues: https://github.com/Tiny18014/QML/issues
- Documentation: See repository wiki

## Changelog

### Version 1.0.0 (Initial Release)

- ✅ Live inference engine with model loading
- ✅ Four visualization components (sentiment, volatility, trends, health)
- ✅ Auto-refresh mechanism
- ✅ Agno agent integration
- ✅ On-demand forecasting
- ✅ Comprehensive configuration
- ✅ API endpoints (optional)
- ✅ Performance optimizations
- ✅ Full documentation

## Acknowledgments

Built on the existing QML forecasting pipeline with enhancements for real-time visualization and decision support.
