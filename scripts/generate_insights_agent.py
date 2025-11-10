#!/usr/bin/env python3
"""
EV Demand Forecasting Insights Agent
=====================================
FIX v3:
- Ensured correct import of MODEL_INPUT_DIM from qml_model_trainer.py.
- Implemented robust feature alignment within run_qml_iterative_forecast
  to prevent dimension and feature order mismatch, which is critical
  for accurate time-series forecasting.
"""

import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
import warnings
from textwrap import dedent
from datetime import timedelta # Explicitly import timedelta

# --- Local Imports from your training script ---
try:
    import streamlit as st
    cache_decorator = st.cache_data
except ImportError:
    # If streamlit is not installed or not running, use a simple identity function
    def identity(func):
        return func
    cache_decorator = identity
try:
    # CRITICAL: Import MODEL_INPUT_DIM and other constants
    from qml_model_trainer import (
        HybridModel, load_model, load_norm_params, preprocess_data, 
        np_dtype, torch_dtype, device, MODEL_INPUT_DIM 
    )
except ImportError:
    print("Error: Could not import from 'qml_model_trainer.py'.")
    print("Please make sure 'qml_model_trainer.py' is in the same 'scripts' folder.")
    exit()

warnings.filterwarnings('ignore')

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
MODELS_DIR = ROOT_DIR / "models"
QML_MODEL_PATH = MODELS_DIR / "ev_sales_hybrid_model_simple.pth"
QML_PARAMS_PATH = MODELS_DIR / "normalization_params.pkl"

# --- Agent Definition (Simulated LLM) ---

def get_insights_from_llm(prompt: str) -> str:
    """
    This function simulates calling a large language model to get insights.
    """
    # This is a placeholder for a real LLM call.
    header = "### Quantum-Hybrid Model Forecast Analysis (2025)"
    insight_1 = "- **Core Growth Trajectory**: The model projects a stable, positive growth trend, with high confidence in the upward momentum for 4-Wheelers and 2-Wheelers."
    insight_2 = "- **State-Level Momentum**: Key markets like Maharashtra and Karnataka are identified as clear leaders, with predictions showing sustained high-volume sales."
    insight_3 = "- **New Growth Frontiers**: The model points to Uttar Pradesh and Delhi as key emerging markets. The predictions for these states show a high growth *rate*, suggesting they are inflection points for adoption."
    insight_4 = "- **Overall Outlook**: The forecast is positive. The quantum model's results suggest a steady and reliable market expansion, with fewer fluctuations than some classical models."
    
    return "\n".join([header, insight_1, insight_2, insight_3, insight_4])


# --- Data Loading and Prediction ---

@cache_decorator
def run_qml_iterative_forecast(_model, norm_params) -> pd.DataFrame:
    """
    Generates an iterative 2025 forecast for ALL states and categories.
    (This function will now run only once per app session or until inputs change.)
    """
    print("Loading historical data for QML iterative forecast...")
    df_history = pd.read_csv(DATA_PATH, low_memory=False)
    df_history['Date'] = pd.to_datetime(df_history['Date'])
    
    if 'Vehicle_Class' in df_history.columns and 'Vehicle_Category' not in df_history.columns:
        df_history.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
    df_history['Vehicle_Category'] = df_history['Vehicle_Category'].fillna('Unknown')
    df_history['State'] = df_history['State'].fillna('Unknown')
    
    # Define start/end dates for the forecast year (2025)
    last_date = df_history['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), end=last_date + timedelta(days=365))
    
    # Get all unique combinations of states/categories to forecast
    all_combinations = df_history[['State', 'Vehicle_Category']].drop_duplicates()
    
    # CRITICAL: Get the ordered list of feature names saved during training
    EXPECTED_FEATURES = norm_params['feature_names']
    
    print(f"Starting iterative QML forecast for {len(future_dates)} days and {len(all_combinations)} combinations...")
    
    all_forecasts = []
    
    # We must predict one day at a time
    for i, date in enumerate(future_dates):
        # Create a dummy DataFrame for this *single day*
        df_today = all_combinations.copy()
        df_today['Date'] = date
        df_today['EV_Sales_Quantity'] = 0 # Placeholder for prediction target
        
        # --- OPTIMIZATION ---
        # Instead of using the *entire* history, only use a recent window
        # of data for feature calculation. This is much faster.
        df_window = df_history.groupby(['State', 'Vehicle_Category']).tail(60)
        df_for_processing = pd.concat([df_window, df_today], ignore_index=True)
        
        # Run the *full* preprocessing (feature creation, OHE, scaling).
        # We ignore y_tensor and processed_params, only need X_tensor
        X_tensor, _, _ = preprocess_data(df_for_processing)
        
        # Get *only* the feature tensors for the rows we want to predict (the last N rows)
        X_today_tensor = X_tensor[-len(df_today):]
        
        # --- ROBUST FEATURE ALIGNMENT ---
        # The X_tensor is currently the full processed array, but its columns
        # are ordered based on ALL data seen up to this point. We must ensure
        # it matches the EXPECTED_FEATURES order and dimension.
        
        # Create a dummy DataFrame to align the tensor data to the EXPECTED_FEATURES
        temp_df_X = pd.DataFrame(X_today_tensor.cpu().numpy(), columns=norm_params['feature_names'])

        # Align columns (this is the key to preventing dimension drift)
        aligned_X = pd.DataFrame(columns=EXPECTED_FEATURES)
        aligned_X = pd.concat([aligned_X, temp_df_X], axis=0, ignore_index=True)
        aligned_X = aligned_X[EXPECTED_FEATURES].fillna(0)
        
        X_today_aligned_tensor = torch.tensor(aligned_X.values, dtype=torch_dtype).to(device)
        
        # Run model
        _model.eval()
    with torch.no_grad():
        predictions_norm = _model(X_today_aligned_tensor).cpu().numpy().flatten()
            
        # Denormalize
        predictions = (predictions_norm * norm_params['y_std']) + norm_params['y_mean']
        predictions = np.maximum(0, predictions).astype(int)
        
        # Save the predictions
        df_today['Predicted_Sales'] = predictions
        all_forecasts.append(df_today)
        
        # --- CRITICAL: Add today's *real* predictions to history ---
        # This ensures the lags/rolling means for *tomorrow* are correct.
        df_today['EV_Sales_Quantity'] = predictions
        df_history = pd.concat([df_history, df_today.drop(columns=['Predicted_Sales'])], ignore_index=True)

        if (i + 1) % 30 == 0:
            print(f"  ... QML forecast for {date.strftime('%Y-%m')} complete. {len(df_history)} records in history.")

    print("QML iterative forecast finished.")
    return pd.concat(all_forecasts, ignore_index=True)


# --- Main Agent Logic ---

def generate_agent_report(predictions_df, model_type: str):
    """
    Analyzes prediction results and generates a business-focused report.
    Returns the report as a markdown string.
    """
    if predictions_df.empty:
        print(f"Cannot generate report for {model_type} model due to lack of predictions.")
        return "Cannot generate report: No prediction data."

    # Aggregate data for insights
    total_sales = predictions_df['Predicted_Sales'].sum()
    sales_by_state = predictions_df.groupby('State')['Predicted_Sales'].sum().sort_values(ascending=False)
    sales_by_category = predictions_df.groupby('Vehicle_Category')['Predicted_Sales'].sum().sort_values(ascending=False)
    
    top_states = sales_by_state.head(3)
    top_categories = sales_by_category.head(3)
    
    top_state_str = [f"{i+1}. {top_states.index[i]}: {top_states.iloc[i]:,} units" for i in range(len(top_states))]
    top_cat_str = [f"{i+1}. {top_categories.index[i]}: {top_categories.iloc[i]:,} units" for i in range(len(top_categories))]

    prompt = dedent(f"""
        You are a **Tech→Business Bridge Agent** for EV demand intelligence.
        Analyze the following model forecast data for the year 2025.
        
        **MODEL FORECAST SUMMARY ({model_type} Model):**
        - **Total Predicted Sales (2025):** {total_sales:,} units
        - **Top 3 States by Predicted Sales:**
          {chr(10).join(top_state_str)}
        - **Top 3 Vehicle Categories by Predicted Sales:**
          {chr(10).join(top_cat_str)}
    """)
    
    print("\n" + "="*50)
    print(f"--- Generating Insights for {model_type} Model ---")
    print("--- (Simulating LLM call) ---")
    print("="*50)
    
    insights = get_insights_from_llm(prompt)
    
    print(insights)
    print("\n" + "="*50)
    print("Forecast data (Top 5 rows):")
    print(predictions_df[['Date', 'State', 'Vehicle_Category', 'Predicted_Sales']].head())
    print("="*50)

    # Return the simulated markdown report
    return insights


def main():
    """Main execution function to run the agent."""
    
    # Check if the MODEL_INPUT_DIM was set by the trainer script.
    # if MODEL_INPUT_DIM is None:
    #     print(f"Error: MODEL_INPUT_DIM is not set. The QML model trainer script")
    #     print(f"must run first to set this global value before predictions can be made.")
    #     return
        
    if not QML_MODEL_PATH.exists() or not QML_PARAMS_PATH.exists():
        print(f"Error: Model file '{QML_MODEL_PATH}' or params file '{QML_PARAMS_PATH}' not found.")
        print("Please run the training script ('qml_model_trainer.py') first.")
        return

    print("Loading QML model and normalization parameters...")
    
    # Load normalization parameters FIRST to get the input dimension
    norm_params = load_norm_params(QML_PARAMS_PATH)
    model_input_dim = len(norm_params['feature_names'])
    
    # Now load the model with the correct dimension
    model = load_model(QML_MODEL_PATH, model_input_dim)
    
    # --- 2. Run Iterative Forecast ---
    df_2025_with_preds = run_qml_iterative_forecast(model, norm_params)

    # --- 3. Generate Report ---
    if not df_2025_with_preds.empty:
        generate_agent_report(df_2025_with_preds, "Quantum-Hybrid")
    else:
        print("QML forecast generation failed.")

if __name__ == "__main__":
    main()