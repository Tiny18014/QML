# scripts/generate_forecast.py - FINAL VERSION
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import sys

# Define paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
SCRIPTS_DIR = ROOT_DIR / "scripts"

# Ensure the trainer script can be imported for its functions
sys.path.insert(0, str(SCRIPTS_DIR))
from advanced_model_trainer import create_advanced_features, prepare_data_for_training

def generate_forecast(category: str, state: str, days_to_forecast: int):
    """
    Generates a sales forecast for a specific category and state for a number of future days.
    """
    print(f"\n🚀 Generating {days_to_forecast}-day forecast for '{category}' in '{state}'")

    # 1. Load the correct specialized model
    category_filename = category.replace(" ", "_").replace("/", "_")
    model_path = MODELS_DIR / f"advanced_model_{category_filename}.pkl"
    if not model_path.exists():
        print(f"❌ Error: Model for category '{category}' not found.")
        return

    print("Loading model and historical data...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['primary_model']
    feature_names = model_data['feature_names']
    
    historical_df = pd.read_csv(DATA_PATH, parse_dates=['Date'])

    # 2. Create a future dataframe with all future dates
    last_date = historical_df['Date'].max()
    future_dates = pd.to_datetime(pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_forecast))
    
    future_df = pd.DataFrame({
        'Date': future_dates, 'State': state, 'Vehicle_Category': category
    })
    
    # 3. Create features for the future dates
    # Combine historical and future data so features like 'time_index' are calculated correctly
    combined_df = pd.concat([historical_df, future_df], ignore_index=True)
    
    # Generate features for the combined dataset
    df_with_features = create_advanced_features(combined_df)
    
    # Isolate just the future rows, which now have the correct features
    future_features = df_with_features.iloc[-days_to_forecast:].copy()

    # 4. Prepare the future data and make predictions in one batch
    X_pred_scaled, _, _, _ = prepare_data_for_training(future_features, feature_subset=feature_names)
    
    print("Making predictions...")
    predictions = model.predict(X_pred_scaled)
    future_df['Forecasted_Sales'] = predictions

    # 5. Display and plot the results
    print("\n📊========= FORECAST RESULTS =========📊")
    # Ensure sales are non-negative
    future_df['Forecasted_Sales'] = future_df['Forecasted_Sales'].round(2).clip(lower=0)
    print(future_df[['Date', 'Forecasted_Sales']].to_string(index=False))
    print("========================================")

    plt.figure(figsize=(12, 6))
    plt.plot(future_df['Date'], future_df['Forecasted_Sales'], marker='o', linestyle='-')
    plt.title(f"{days_to_forecast}-Day EV Sales Forecast for {category} in {state}")
    plt.xlabel("Date")
    plt.ylabel("Forecasted Sales Quantity")
    plt.grid(True)
    
    forecast_plot_path = ROOT_DIR / "output" / f"forecast_{state}_{category_filename}.png"
    plt.savefig(forecast_plot_path)
    print(f"\n✅ Forecast plot saved to {forecast_plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate EV Sales Forecast")
    parser.add_argument("--category", type=str, required=True, help="Vehicle category (e.g., '4-Wheelers')")
    parser.add_argument("--state", type=str, required=True, help="State (e.g., 'Karnataka')")
    parser.add_argument("--days", type=int, default=30, help="Number of days to forecast")
    args = parser.parse_args()
    
    generate_forecast(args.category, args.state, args.days)