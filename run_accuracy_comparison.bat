@echo off
echo 🎯 EV Demand Forecasting Accuracy Comparison
echo ============================================
echo.
echo This will compare accuracy between:
echo 1. EV_Dataset.csv (historical data)
echo 2. Data Push Pipeline (Excel files)
echo.
echo Press any key to continue...
pause >nul

echo.
echo 🚀 Running Accuracy Comparison...
echo ============================================

python scripts/run_pipeline.py --mode accuracy

echo.
echo Accuracy comparison completed!
echo Check the output files for detailed results.
echo.
pause
