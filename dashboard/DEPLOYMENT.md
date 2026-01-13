# Deployment Guide for EV Demand Forecasting Dashboard

## Prerequisites

- Python 3.8 or higher
- Git
- Access to the repository
- Optional: Virtual environment tool (venv, conda)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Tiny18014/QML.git
cd QML
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Test model loader
python -c "import sys; sys.modules['torch'] = None; from dashboard.backend.model_loader import ModelLoader; print('✅ Installation successful')"
```

## Running the Dashboard

### Option 1: Enhanced Dashboard (Recommended)

```bash
streamlit run dashboard/frontend/app.py
```

Access at: `http://localhost:8501`

### Option 2: Legacy Dashboard

```bash
streamlit run scripts/streamlit_dashboard.py
```

### Option 3: API Server (Optional)

```bash
python dashboard/backend/api.py
```

API available at: `http://localhost:5000`

## Configuration

Edit `dashboard/config/dashboard_config.yaml` to customize:

```yaml
# Data directories
data:
  models_dir: "models"
  output_dir: "output"
  data_dir: "data"

# Refresh settings
refresh:
  auto_refresh: true
  default_interval_seconds: 300  # 5 minutes

# Visualization settings
visualization:
  theme: "plotly_white"
  default_height: 400
```

## Environment Variables

Create `.streamlit/secrets.toml` for API keys:

```toml
DF_AGENT = "your_huggingface_token_here"
```

## Verifying the Deployment

### 1. Check Models Available

```bash
ls models/*.pkl
# Should show: advanced_model_*.pkl, specialized_*.pkl, etc.
```

### 2. Check Data Files

```bash
ls data/EV_Dataset.csv
ls output/daily_predictions_2026.csv  # If available
```

### 3. Run Integration Test

```bash
python dashboard/tests/integration_test.py
```

Expected output:
```
✅ Model Loader PASSED
```

## Troubleshooting

### Issue: "No module named 'streamlit'"

**Solution:**
```bash
pip install streamlit plotly pandas numpy
```

### Issue: "No models found"

**Solution:**
1. Verify `models/` directory exists
2. Check that `.pkl` files are present
3. Run CI/CD pipeline to generate models:
   ```bash
   python scripts/advanced_model_trainer.py
   ```

### Issue: "Torch import error"

**Solution:**
Torch is optional. The dashboard will work without it for non-PyTorch models:
```bash
pip uninstall torch  # Remove problematic torch
# Or install CPU-only version:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "Agent unavailable"

**Solution:**
1. Check HF_TOKEN in `.streamlit/secrets.toml`
2. Verify API quota not exceeded
3. Dashboard will show fallback metrics if agent unavailable

## Production Deployment

### Security Considerations

1. **Never enable Flask debug mode in production**
   ```python
   # In api.py
   run_api(debug=False)  # Always False for production
   ```

2. **Use environment variables for secrets**
   - Never commit secrets to Git
   - Use `.streamlit/secrets.toml` (in `.gitignore`)

3. **API Authentication** (if using API)
   - Add authentication middleware
   - Use HTTPS
   - Implement rate limiting

### Scalability

1. **Use caching**
   - Models are cached automatically
   - Configure TTL in `dashboard_config.yaml`

2. **Optimize data loading**
   - Use lazy loading for large datasets
   - Paginate results

3. **Deploy with Gunicorn** (for API)
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 dashboard.backend.api:app
   ```

### Monitoring

1. **Check logs**
   ```bash
   tail -f output/incremental_training.log
   ```

2. **Monitor performance**
   - Dashboard load time should be < 2 seconds
   - Prediction generation should be < 5 seconds

3. **Set up alerts**
   - Model update failures
   - API errors
   - High memory usage

## CI/CD Integration

The dashboard automatically detects new models from the CI/CD pipeline:

1. **Weekly Pipeline** runs on 1st of month (`ev_forecast.yml`)
2. **Models committed** to `models/` directory
3. **Dashboard auto-refresh** detects changes
4. **New predictions** generated automatically

### Manual Pipeline Trigger

```bash
# Via GitHub Actions UI or:
gh workflow run ev_forecast.yml
```

## Updating the Dashboard

### Pull Latest Changes

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Clear Cache

```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Clear Streamlit cache
rm -rf ~/.streamlit/cache/
```

### Restart Services

```bash
# Kill existing Streamlit process
pkill -f streamlit

# Restart dashboard
streamlit run dashboard/frontend/app.py
```

## Performance Tuning

### 1. Adjust Refresh Interval

```yaml
# dashboard_config.yaml
refresh:
  default_interval_seconds: 600  # 10 minutes instead of 5
```

### 2. Limit Data Points

```yaml
# dashboard_config.yaml
performance:
  max_data_points: 5000  # Reduce from 10000
```

### 3. Disable Auto-Refresh

In the dashboard sidebar, uncheck "Auto Refresh"

## Support

For issues:
- GitHub Issues: https://github.com/Tiny18014/QML/issues
- Check logs in `output/` directory
- Review documentation: `dashboard/README.md`

## Backup and Recovery

### Backup Models

```bash
cp -r models/ models_backup_$(date +%Y%m%d)/
```

### Restore Models

```bash
cp -r models_backup_YYYYMMDD/* models/
```

### Export Data

Use the dashboard export functionality or:
```bash
# Export predictions
python -c "import pandas as pd; df = pd.read_csv('output/daily_predictions_2026.csv'); df.to_excel('predictions_export.xlsx', index=False)"
```
