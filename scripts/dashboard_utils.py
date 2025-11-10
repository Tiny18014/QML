"""
Dashboard Utilities
===================
REFACTOR v4: Implements a custom DashboardAgent class to encapsulate the LLM persona, 
mirroring the structure of the friend's 'agno' agent framework but using the 
standard OpenAI SDK for the 'openai/gpt-oss-20b' model via the Hugging Face Router.
"""

import pandas as pd
import numpy as np
import torch
import joblib
import pickle
from pathlib import Path
import warnings
from textwrap import dedent
import plotly.express as px
import streamlit as st
import sys
import os 
from openai import OpenAI # New import for the gpt-oss-20b compatible API

# --- Path and Model Setup ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
MODELS_DIR = ROOT_DIR / "models"
CLASSICAL_MODEL_PREFIX = "advanced_model_"
QML_MODEL_PATH = MODELS_DIR / "ev_sales_hybrid_model_simple.pth"
QML_PARAMS_PATH = MODELS_DIR / "normalization_params.pkl"

# Ensure other scripts (like qml_model_trainer) can be imported
sys.path.append(str(ROOT_DIR / "scripts"))

# CRITICAL IMPORTS from training script
try:
    from advanced_model_trainer import create_advanced_features, prepare_data_for_training
    from qml_model_trainer import (
        HybridModel, load_model as load_qml_model, load_norm_params as load_qml_params,
        preprocess_data, torch_dtype, device, MODEL_INPUT_DIM
    )
except ImportError as e:
    # Define placeholder constant for the script to run without crashing the dashboard
    MODEL_INPUT_DIM = 50 
    
warnings.filterwarnings('ignore')

# --- AGENT CONFIGURATION & INITIALIZATION ---

# Token used by your friend's setup - NOW USING DF_AGENT
HF_TOKEN = st.secrets.get("DF_AGENT")

# Dedicated router URL for gpt-oss models on Hugging Face infrastructure
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
LLM_MODEL_ID = "openai/gpt-oss-20b"
LLM_CLIENT = None

if HF_TOKEN:
    try:
        # Initialize the OpenAI client pointing to the HF router
        LLM_CLIENT = OpenAI(
            base_url=HF_ROUTER_BASE_URL,
            api_key=HF_TOKEN, # The HF token is used as the API key here
            timeout=30.0 # Set a reasonable timeout
        )
        print(f"OpenAI Client initialized for Hugging Face Router with {LLM_MODEL_ID}.")
    except Exception as e:
        print(f"Error initializing OpenAI client for HF Router: {e}")
        LLM_CLIENT = None


class DashboardAgent:
    """
    Replicates the Agent structure for the Strategic Insights report generation.
    """
    def __init__(self):
        # Description acts as the primary System Instruction
        self.description = dedent(
            """\
            You are a **Tech→Business Bridge Agent** for EV demand intelligence. 
            Your purpose is to translate the provided numerical forecast data into **executive-style insights**. 
            
            You analyze the sales forecasts from the provided model and interpret what they imply about 
            **market demand**, **high-growth regions**, and **strategic direction**.
            You do not use statistical jargon — your tone is **analytical, confident, and human-like**, 
            as though briefing senior management.
            """
        )
        
        # Instructions define the specific output format/constraints
        self.instructions = dedent(
            """\
            PROCESS:
            - Analyze the key metrics: Total Sales, Top 5 States, and Top 5 Categories.
            - Focus on identifying where sales growth is concentrated and what the market opportunity is.
            
            OUTPUT STYLE:
            Output it as a **markdown** report with:
            - A headline which is just the model type and forecast year. Ex: ### Classical Model Forecast Analysis (2025)
            - 3–4 concise analytical bullets: high-level trends, key risks/opportunities, and strategic implications. 
            - KEEP IT SHORT, DO NOT WRITE LONG BULLETS.
            - Tone: Executive, inferential, realistic, and commercially relevant.
            """
        )

    def invoke(self, model_type: str, data_summary: str):
        """Builds the full prompt and calls the LLM."""
        
        # The full prompt combines the system prompt (description) and the user prompt (data + instructions)
        system_prompt = self.description
        user_prompt = dedent(f"""
            {self.instructions}
            
            INPUT DATA (Model: {model_type}):
            {data_summary}
            
            Now, generate the report following the OUTPUT STYLE.
            """)
            
        return LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL_ID, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.25,
            max_tokens=512, # Keep output concise
        )

# Instantiate the agent globally
report_agent = DashboardAgent()


# --- Core Functions for Strategic Insights ---

@st.cache_data
def get_2025_data():
    """Loads the dataset and filters/creates data for the year 2025."""
    print("Loading and filtering data for 2025...")
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], low_memory=False)
    df_2025 = df[df['Date'].dt.year == 2025].copy()
    
    if df_2025.empty:
        print("No data available for 2025. Creating dummy data from 2024.")
        df_2024 = df[df['Date'].dt.year == 2024].copy()
        if df_2024.empty:
            raise ValueError("No data for 2024 found to create dummy 2025 data.")
        df_2025 = df_2024.copy()
        df_2025['Date'] = df_2025['Date'] + pd.DateOffset(years=1)
        df_2025['EV_Sales_Quantity'] = 0
        
    if 'Vehicle_Category' not in df_2025.columns and 'Vehicle_Class' in df_2025.columns:
        df_2025.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
    df_2025['Vehicle_Category'] = df_2025['Vehicle_Category'].fillna('Unknown')

    return df_2025

@st.cache_data
def run_classical_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates predictions using the per-category advanced classical models.
    """
    print("\nRunning predictions with advanced classical models...")
    all_category_preds = []
    
    # Ensure Vehicle_Category is ready for grouping
    df['Vehicle_Category'] = df['Vehicle_Category'].fillna('Unknown')
    categories = df['Vehicle_Category'].unique()
    
    # 1. Create all advanced features for the full 2025 data once
    # This prepares the time-series and cross-category features correctly.
    df_featured_full = create_advanced_features(df.copy())

    for category in categories:
        df_category = df_featured_full[df_featured_full['Vehicle_Category'] == category].copy()
        if df_category.empty: 
            continue
            
        category_filename = category.replace(" ", "_").replace("/", "_")
        model_path = MODELS_DIR / f"{CLASSICAL_MODEL_PREFIX}{category_filename}.pkl"
        
        if not model_path.exists():
            print(f"Warning: Model file not found for '{category}' at {model_path}. Skipping.")
            continue
            
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            model = model_data['primary_model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            
            # 2. Prepare features for the model using the pre-fitted scaler
            # The prepare_data_for_training function is used here ONLY to transform the data
            X_scaled, _, _, _ = prepare_data_for_training(df_category.copy(), feature_subset=feature_names) 

            # CRITICAL CHECK: Ensure the current transformed feature count matches the trained model
            if X_scaled.shape[1] != len(feature_names):
                print(f"Warning: Feature count mismatch for '{category}'. Expected {len(feature_names)}, got {X_scaled.shape[1]}. Skipping.")
                continue

            # 3. Make predictions
            predictions = model.predict(X_scaled)
            
            df_category['Predicted_Sales'] = np.maximum(0, predictions).astype(int)
            all_category_preds.append(df_category)
        
        except Exception as e:
            print(f"Error predicting for category '{category}': {e}")
            continue
            
    if not all_category_preds: 
        return pd.DataFrame()
        
    # 4. Aggregate results
    final_df = pd.concat(all_category_preds, ignore_index=True)
    return final_df[['Date', 'State', 'Vehicle_Category', 'Predicted_Sales']]

@st.cache_data
def run_qml_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Generates predictions using the hybrid QML model."""
    print("\nRunning predictions with QML model...")
    if not QML_MODEL_PATH.exists() or not QML_PARAMS_PATH.exists():
        print("QML model or params not found. Skipping.")
        return pd.DataFrame()
    
    df_qml = df.copy()
    
    # --- Load Parameters Once and Check Critical Values ---
    norm_params = load_qml_params(QML_PARAMS_PATH) # Load 1
    input_dim = norm_params.get('input_dim')
    EXPECTED_FEATURE_NAMES = norm_params.get('feature_names')
    
    if input_dim is None or EXPECTED_FEATURE_NAMES is None:
        print("Error: Model dimension/feature names not found in params file. Check trainer script.")
        return pd.DataFrame()

    # --- 1. PREPROCESS DATA (Feature Creation) ---
    X_tensor, _, processed_params = preprocess_data(df_qml.copy()) 

    # --- 2. ALIGN FEATURES (Robust Alignment) ---
    # Convert X_tensor back to a DataFrame using the feature names from the current run
    current_features = pd.DataFrame(X_tensor.cpu().numpy(), columns=processed_params['feature_names'])
    
    # Use reindex to align current features with the expected features
    aligned_X_df = current_features.reindex(columns=EXPECTED_FEATURE_NAMES).fillna(0)
    
    # Final check before prediction
    if aligned_X_df.shape[1] != input_dim:
        print(f"Warning: QML feature mismatch AFTER alignment. Expected {input_dim}, got {aligned_X_df.shape[1]}. Skipping.")
        return pd.DataFrame()

    # Convert aligned DataFrame back to Tensor (ready for model)
    X_tensor = torch.tensor(aligned_X_df.values, dtype=torch_dtype).to(device)
    X_tensor = X_tensor.to("cpu") # Ensure data is on CPU for Pennylane lightning.qubit

    # --- 3. RUN PREDICTION ---
    model = load_qml_model(QML_MODEL_PATH, input_dim)
    
    model.eval()
    with torch.no_grad():
        predictions_norm = model(X_tensor).cpu().numpy().flatten()
        
    # Denormalize
    y_mean, y_std = norm_params['y_mean'], norm_params['y_std']
    predictions = (predictions_norm * y_std) + y_mean
    
    df_qml['Predicted_Sales'] = np.maximum(0, predictions).astype(int)
    
    return df_qml[['Date', 'State', 'Vehicle_Category', 'Predicted_Sales']]

def generate_agent_report(predictions_df: pd.DataFrame, model_type: str):
    """Analyzes prediction results and generates a business-focused report using the custom Agent."""
    global report_agent

    if predictions_df.empty:
        st.warning(f"Cannot generate report for {model_type} model: No prediction data.")
        return
    
    if not LLM_CLIENT:
        # Changed message to reflect DF_AGENT
        st.error("LLM Agent is not initialized. Check if the 'DF_AGENT' secret is set correctly.")
        _display_fallback_insights(predictions_df, model_type)
        return

    # 1. Prepare Data Summary for the Agent
    total_sales = predictions_df['Predicted_Sales'].sum()
    sales_by_state = predictions_df.groupby('State')['Predicted_Sales'].sum().sort_values(ascending=False)
    sales_by_category = predictions_df.groupby('Vehicle_Category')['Predicted_Sales'].sum().sort_values(ascending=False)
    
    data_summary = dedent(f"""
        Total Forecasted Sales (Units): {total_sales:,}
        Top 5 States by Sales: {sales_by_state.head(5).to_dict()}
        Top 5 Vehicle Categories by Sales: {sales_by_category.head(5).to_dict()}
    """)
    
    # 2. Invoke the Agent
    try:
        with st.spinner(f"Generating expert report using {LLM_MODEL_ID} (Agent Mode)..."):
            response = report_agent.invoke(model_type, data_summary)
        
        st.markdown(f"### {model_type} Model Forecast Analysis (2025) 🤖")
        # Extract content from the OpenAI SDK response object
        st.markdown(response.choices[0].message.content)

    except Exception as e:
        st.error(f"LLM API Error (HF Router): Failed to generate report. {e}")
        _display_fallback_insights(predictions_df, model_type)


def _display_fallback_insights(predictions_df: pd.DataFrame, model_type: str):
    """Simple fallback function to keep the UI from breaking."""
    
    if predictions_df.empty:
        return

    total_sales = predictions_df['Predicted_Sales'].sum()
    sales_by_state = predictions_df.groupby('State')['Predicted_Sales'].sum().sort_values(ascending=False).head(5)
    sales_by_category = predictions_df.groupby('Vehicle_Category')['Predicted_Sales'].sum().sort_values(ascending=False).head(5)

    st.markdown(f"### Generic {model_type} Forecast Analysis (2025) - Fallback")
    st.markdown(f"* **Total Projected Sales:** The model forecasts **{total_sales:,}** units.")
    st.markdown(f"* **Key Markets:** The top three markets are **{', '.join(list(sales_by_state.index)[:3])}**.")
    st.markdown(f"* **Dominant Segments:** Demand is highest in the **{', '.join(list(sales_by_category.index)[:2])}** categories.")


# --- On-Demand Forecasting Function (Remains the same) ---

@st.cache_data
def generate_on_demand_forecast(category: str, state: str, days_to_forecast: int):
    """Generates and plots a forecast for a specific state and category."""
    
    category_filename = category.replace(" ", "_").replace("/", "_")
    classical_model_path = MODELS_DIR / f"advanced_model_{category_filename}.pkl"
    
    if not classical_model_path.exists():
        return None, f"Error: Model for category '{category}' not found at {classical_model_path}"

    with open(classical_model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    model = model_data['primary_model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # 1. Prepare Historical Data
    historical_df = pd.read_csv(DATA_PATH, parse_dates=['Date'], low_memory=False)
    
    # --- FIX 1: Standardize column names immediately on historical data ---
    if 'Vehicle_Class' in historical_df.columns:
        historical_df.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
    
    # --- FIX 2: Correct filtering logic (Removed redundant/incorrect Vehicle_Class check) ---
    historical_df = historical_df[
        (historical_df['State'] == state) & 
        (historical_df['Vehicle_Category'] == category)
    ].copy()
    
    # Check if we have enough historical data to forecast
    if historical_df.empty:
        return None, f"Error: No historical data found for '{category}' in '{state}'. Please choose a different combination."

    last_date = historical_df['Date'].max()
    future_dates = pd.to_datetime(pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_forecast))
    
    # 2. Create Future Data Frame
    future_df = pd.DataFrame({'Date': future_dates, 'State': state, 'Vehicle_Category': category, 'EV_Sales_Quantity': 0})
    
    # 3. Combine Data for Feature Generation (CRITICAL for lags/rolling)
    combined_df = pd.concat([historical_df, future_df], ignore_index=True)
    
    # 4. Generate Features using the trainer's logic
    # NOTE: create_advanced_features and prepare_data_for_training are imported from trainer scripts
    df_with_features = create_advanced_features(combined_df)
    
    # Isolate the future features
    future_features = df_with_features.iloc[-days_to_forecast:].copy()

    # 5. Prepare and Scale Features (using the pre-fitted scaler from the loaded model)
    X_pred_scaled, _, _, _ = prepare_data_for_training(future_features.copy(), feature_subset=feature_names)
    
    # 6. Make Predictions
    predictions = model.predict(X_pred_scaled)
    future_df['Forecasted_Sales'] = np.maximum(0, predictions).round(0).astype(int)

    # 7. Create Plot
    fig = px.line(future_df, x='Date', y='Forecasted_Sales', title=f"{days_to_forecast}-Day Forecast for {category} in {state}")
    fig.update_traces(mode='lines+markers')
    fig.update_layout(xaxis_title="Date", yaxis_title="Forecasted Sales")
    
    return future_df[['Date', 'Forecasted_Sales']], fig