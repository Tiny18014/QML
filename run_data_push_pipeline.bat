@echo off
echo 🚀 Starting EV Data Push Pipeline...
echo.

REM Change to the project directory
cd /d "%~dp0"

REM Run the data push pipeline
python scripts/run_pipeline.py --mode datapush

echo.
echo ✅ Pipeline execution completed!
echo 📁 Check the output folder for results
echo 📋 Check output/push_pipeline.log for detailed logs
pause
