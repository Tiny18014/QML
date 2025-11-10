import pandas as pd
from pathlib import Path
import joblib
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import logging

# --- Configuration ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- File Paths ---
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / 'data' / 'EV_Dataset.csv'
MODELS_PATH = BASE_PATH / 'models'
OUTPUT_PATH = BASE_PATH / 'output'

def generate_on_demand_forecast(category: str, state: str, days_to_forecast: int):
    """
    Generates an on-demand forecast for a specific category and state.

    Args:
        category (str): The vehicle category to forecast.
        state (str): The state to forecast.
        days_to_forecast (int): The number of days into the future to forecast.

    Returns:
        tuple: A pandas DataFrame with the forecast and a Plotly figure.
    """
    try:
        model_name = f"advanced_model_{category.replace(' ', '_')}.pkl"
        scaler_name = f"feature_scaler_{category.replace(' ', '_')}.pkl"
        
        model_path = MODELS_PATH / model_name
        scaler_path = MODELS_PATH / scaler_name

        if not model_path.exists() or not scaler_path.exists():
            logging.error(f"Model or scaler not found for category '{category}'")
            return None, None

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Generate future dates
        last_date = datetime.now()
        date_range = pd.date_range(start=last_date, periods=days_to_forecast + 1)[1:]
        
        future_df = pd.DataFrame({
            'Date': date_range,
            'State': state,
            'Vehicle_Category': category
        })
        
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Year'] = future_df['Date'].dt.year
        
        # One-hot encode categorical features
        future_df_encoded = pd.get_dummies(future_df, columns=['State', 'Vehicle_Category'], drop_first=True)
        
        # Align columns with training data
        feature_names = joblib.load(MODELS_PATH / 'feature_names.pkl')
        
        # Add missing columns
        for col in feature_names:
            if col not in future_df_encoded.columns:
                future_df_encoded[col] = 0
        
        # Ensure order is the same
        future_df_encoded = future_df_encoded[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(future_df_encoded)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        future_df['Predicted_Sales'] = predictions.round().astype(int)
        
        # Create plot
        fig = px.line(future_df, x='Date', y='Predicted_Sales', title=f'Forecast for {category} in {state}')
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Predicted Sales",
            template="plotly_white"
        )
        
        logging.info(f"Successfully generated forecast for {category} in {state}.")
        
        return future_df[['Date', 'Predicted_Sales']], fig

    except Exception as e:
        logging.error(f"Error in generate_on_demand_forecast: {e}", exc_info=True)
        return None, None

if __name__ == '__main__':
    # Example of how to run the script directly
    # This part will only execute when you run `python generate_forecast.py`
    
    # --- Parameters for the forecast ---
    TARGET_CATEGORY = "4-Wheelers"
    TARGET_STATE = "Delhi"
    DAYS = 90
    # ------------------------------------

    print(f"--- Generating On-Demand Forecast ---")
    print(f"Target: {TARGET_CATEGORY} in {TARGET_STATE}")
    print(f"Days: {DAYS}")
    
    forecast_data, figure = generate_on_demand_forecast(TARGET_CATEGORY, TARGET_STATE, DAYS)
    
    if forecast_data is not None and figure is not None:
        print("\n--- Forecast Results ---")
        print(forecast_data.head())
        
        # Save the forecast data to a CSV
        output_filename = OUTPUT_PATH / f"on_demand_forecast_{TARGET_STATE}_{TARGET_CATEGORY}.csv"
        forecast_data.to_csv(output_filename, index=False)
        print(f"\nForecast data saved to: {output_filename}")
        
        # Show the plot
        figure.show()
    else:
        print("\n--- Forecast Generation Failed ---")
        print("Please check the logs for more details.")
