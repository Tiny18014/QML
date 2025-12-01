print("🚀 STARTING TEST SCRIPT 🚀")
import sys
from pathlib import Path
import pandas as pd

# Add scripts directory to path so we can import dashboard_utils
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

try:
    from dashboard_utils import generate_on_demand_forecast
except ImportError as e:
    print(f"Error importing dashboard_utils: {e}")
    sys.exit(1)

def test_forecast(category, state, days):
    print(f"\n--- Testing Forecast for {category} in {state} for {days} days ---")
    try:
        df, fig = generate_on_demand_forecast(category, state, days)
        if df is not None and not df.empty:
            print("✅ Forecast generated successfully.")
            print(f"Shape: {df.shape}")
            print("First 5 rows:")
            print(df.head())
            print(f"Total Forecasted Sales: {df['Forecasted_Sales'].sum()}")
        else:
            print(f"❌ Forecast returned None or empty. Message: {fig}")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test Case 1: Daily Model (2-Wheelers)
    test_forecast("2-Wheelers", "Maharashtra", 30)

    # Test Case 2: Monthly Model (Bus)
    test_forecast("Bus", "Delhi", 90)
