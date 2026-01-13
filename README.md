# QML - EV Demand Forecasting System

A comprehensive machine learning system for Electric Vehicle (EV) demand forecasting with real-time visualization and decision support capabilities.

## 🚀 Quick Start

### Running the Dashboard

```bash
# Option 1: Quick Launcher
python dashboard/launcher.py

# Option 2: Direct Launch
streamlit run dashboard/frontend/app.py

# Option 3: Legacy Dashboard
streamlit run scripts/streamlit_dashboard.py
```

### Installation

```bash
# Clone repository
git clone https://github.com/Tiny18014/QML.git
cd QML

# Install dependencies
pip install -r requirements.txt
```

## 📊 Features

### Enhanced Dashboard (NEW)

The new comprehensive dashboard provides:

- **Live Inference Engine**: Real-time model loading and predictions
- **Real-time Sentiment Gauges**: Market sentiment visualization with color-coded zones
- **Volatility Analysis**: Interactive time-series plots with heatmaps
- **Brand-Specific Trends**: Multi-category comparison and growth analysis
- **Market Health Monitoring**: KPI cards, automated alerts, and drill-down capabilities
- **AI-Powered Insights**: Integration with Agno agent for strategic summaries
- **Auto-Refresh**: Automatic detection and loading of new model artifacts

**Documentation**: See [`dashboard/README.md`](dashboard/README.md) for complete documentation.

### Model Training Pipeline

- **Advanced Models**: XGBoost and LightGBM ensemble models
- **Specialized Models**: Category-specific (Bus, 3-Wheelers) monthly models
- **QML Models**: Quantum-hybrid neural network models
- **Incremental Learning**: Continuous model updates

### CI/CD Pipeline

Automated monthly training and deployment via GitHub Actions:
- Data preprocessing and merging
- Model training and evaluation
- Automated model deployment

## 📁 Project Structure

```
QML/
├── dashboard/              # 🆕 Enhanced Dashboard System
│   ├── backend/           # Model loading, inference, metrics, API
│   ├── frontend/          # Streamlit app and visualization components
│   ├── config/            # Configuration files
│   ├── tests/             # Unit and integration tests
│   ├── README.md          # Dashboard documentation
│   ├── DEPLOYMENT.md      # Deployment guide
│   └── launcher.py        # Quick start script
├── scripts/               # Training and prediction scripts
│   ├── streamlit_dashboard.py  # Legacy dashboard
│   ├── advanced_model_trainer.py
│   ├── qml_model_trainer.py
│   └── ...
├── models/                # Trained model artifacts (15+ models)
├── data/                  # Dataset files
├── output/                # Predictions and metrics
└── requirements.txt       # Python dependencies
```

## 🎯 Dashboard Components

### 1. Sentiment Analysis Tab
- **Current Sentiment Gauge**: 0-100 score with visual indicators
- **Historical Trends**: Sentiment evolution over time
- **Key Indicators**: Total sales, average daily, top performers

### 2. Volatility & Risk Tab
- **Time-Series Plots**: Rolling volatility visualization
- **Heatmaps**: Volatility by region/category
- **Event Detection**: High volatility alerts
- **Configurable Windows**: Adjustable analysis periods

### 3. Trends & Forecasts Tab
- **Multi-Brand Comparison**: Side-by-side trend analysis
- **Growth Rates**: Category-specific growth visualization
- **Market Share**: Distribution analysis
- **Forecast vs Actual**: Prediction accuracy comparison

### 4. Market Health Tab
- **Health Score**: Overall market health (0-100)
- **KPI Cards**: Sales, coverage, diversity metrics
- **Automated Alerts**: Significant change notifications
- **Drill-Down**: Detailed analysis by region/category

## 🔧 Technical Stack

- **Backend**: Python, Flask (optional API)
- **Frontend**: Streamlit, Plotly
- **ML Models**: XGBoost, LightGBM, PyTorch
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Configuration**: YAML

## 📈 Model Performance

The system includes 15+ trained models:
- **Category-Specific Models**: 2-Wheelers, 3-Wheelers, 4-Wheelers, Bus, Others
- **Specialized Monthly Models**: Bus, 3-Wheelers
- **Hybrid Models**: QML quantum-enhanced models

**Performance Metrics**: See `model_evaluation_summary_final.csv`

## 🔒 Security

- **CodeQL Verified**: 0 security alerts
- **No Hardcoded Secrets**: Uses environment variables
- **Production Ready**: Deployment guide with security best practices
- **Error Handling**: Comprehensive exception handling throughout

## 📚 Documentation

- **Dashboard Guide**: [`dashboard/README.md`](dashboard/README.md)
- **Deployment Guide**: [`dashboard/DEPLOYMENT.md`](dashboard/DEPLOYMENT.md)
- **Implementation Summary**: [`dashboard/IMPLEMENTATION_SUMMARY.md`](dashboard/IMPLEMENTATION_SUMMARY.md)
- **Configuration Reference**: [`dashboard/config/dashboard_config.yaml`](dashboard/config/dashboard_config.yaml)

## 🧪 Testing

```bash
# Run integration tests
python dashboard/tests/integration_test.py

# Run unit tests (requires additional dependencies)
python -m pytest dashboard/backend/tests/
```

## 🚀 Deployment

### Local Development
```bash
streamlit run dashboard/frontend/app.py
```

### Production Deployment
See [`dashboard/DEPLOYMENT.md`](dashboard/DEPLOYMENT.md) for complete production deployment guide including:
- Environment setup
- Security configuration
- Performance tuning
- Monitoring and maintenance

## 🔄 CI/CD Integration

The dashboard automatically integrates with the CI/CD pipeline:

1. **Weekly Pipeline**: Trains models on the 1st of each month
2. **Artifact Detection**: Dashboard detects new models automatically
3. **Auto-Refresh**: Updates predictions with latest models
4. **Manual Trigger**: Can be triggered via GitHub Actions

**Workflow**: `.github/workflows/ev_forecast.yml`

## 📊 Data Pipeline

1. **Data Ingestion**: Raw EV sales data from multiple sources
2. **Preprocessing**: Cleaning, merging, feature engineering
3. **Model Training**: Ensemble and specialized models
4. **Prediction Generation**: Daily forecasts for 2026
5. **Visualization**: Real-time dashboard updates

## 🤝 Contributing

For issues, feature requests, or contributions:
- GitHub Issues: https://github.com/Tiny18014/QML/issues
- Pull Requests: Follow existing code style and include tests

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

Built on advanced machine learning techniques for EV demand forecasting with real-time visualization capabilities for strategic decision-making.

---

**Quick Links**:
- [Dashboard Documentation](dashboard/README.md)
- [Deployment Guide](dashboard/DEPLOYMENT.md)
- [Configuration](dashboard/config/dashboard_config.yaml)
- [CI/CD Workflow](.github/workflows/ev_forecast.yml)

**Status**: ✅ Production Ready
