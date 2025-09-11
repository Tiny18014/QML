@echo off
echo 🚗 EV Model Incremental Training
echo ================================
echo.
echo This will update your advanced model with new data
echo without retraining the entire model from scratch.
echo.
echo Make sure you have:
echo 1. ✅ Advanced model already trained (advanced_ev_model.pkl)
echo 2. 📁 New data file (CSV or Excel) ready
echo.
pause
echo.
python scripts/run_pipeline.py --mode incremental
echo.
pause
