"""
Dashboard Utilities
===================
REFACTOR v8: Combined Fixes.
1. Robust QML Inference (using training stats).
2. Monthly-to-Daily Distribution for On-Demand Forecasting.
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
from openai import OpenAI

# --- Path and Model Setup ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
MODELS_DIR = ROOT_DIR / "models"
CLASSICAL_MODEL_PREFIX = "advanced_model_"

MODEL_3W_PATH = MODELS_DIR / "specialized_3w_monthly_model.pkl"
MODEL_BUS_PATH = MODELS_DIR / "specialized_bus_monthly_model.pkl"
QML_MODEL_PATH = MODELS_DIR / "ev_sales_hybrid_model_simple.pth"
QML_PARAMS_PATH = MODELS_DIR / "normalization_params.pkl"

sys.path.append(str(ROOT_DIR / "scripts"))

# CRITICAL IMPORTS
try:
    from advanced_model_trainer import create_advanced_features, prepare_data_for_training
    from qml_model_trainer import (
        HybridModel, load_model as load_qml_model, load_norm_params as load_qml_params,
        prepare_data_for_inference, torch_dtype, device
    )
except ImportError as e:
    print(f"Warning: Import failed {e}")
    
warnings.filterwarnings('ignore')

# --- AGENT CONFIGURATION ---
HF_TOKEN = st.secrets.get("DF_AGENT")
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
LLM_MODEL_ID = "openai/gpt-oss-20b"
LLM_CLIENT = None

if HF_TOKEN:
    try:
        LLM_CLIENT = OpenAI(
            base_url=HF_ROUTER_BASE_URL,
            api_key=HF_TOKEN,
            timeout=60.0,
            max_retries=2
        )
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        LLM_CLIENT = None

class DashboardAgent:
    def __init__(self):
        self.description = dedent("""
            You are a **Tech→Business Bridge Agent** for EV demand intelligence. 
            Translate numerical forecast data into **executive-style insights**.
            """)
        self.instructions = dedent("""
            OUTPUT STYLE:
            - A headline.
            - 3–4 concise analytical bullets.
            - Tone: Executive, inferential, realistic.
            """)

    def invoke(self, model_type: str, data_summary: str):
        system_prompt = self.description
        user_prompt = dedent(f"""
            {self.instructions}
            INPUT DATA (Model: {model_type}):
            {data_summary}
            Generate the report.
            """)
        return LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL_ID, 
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.25, max_tokens=512,
        )

report_agent = DashboardAgent()

# --- HELPER: Monthly Feature Engineering ---
def create_monthly_features_exact(df):
    df = df.copy()
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['quarter'] = df['Date'].dt.quarter
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_month_{lag}'] = df.groupby('State')['EV_Sales_Quantity'].shift(lag)
    for w in [3, 6, 12]:
        g = df.groupby('State')['EV_Sales_Quantity']
        df[f'roll_mean_{w}m'] = g.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'roll_std_{w}m'] = g.rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True)
        df[f'roll_max_{w}m'] = g.rolling(window=w, min_periods=1).max().reset_index(level=0, drop=True)
    df['momentum'] = df['roll_mean_3m'] / (df['roll_mean_12m'] + 1)
    df['state_avg'] = df.groupby('State')['EV_Sales_Quantity'].expanding().mean().reset_index(level=0, drop=True)
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(0)
    return df

@st.cache_data
def get_2025_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], low_memory=False)
    df_2025 = df[df['Date'].dt.year == 2024].copy()
    if df_2025.empty: df_2025 = df.tail(1000).copy()
    df_2025['Date'] = df_2025['Date'] + pd.DateOffset(years=1)
    df_2025['EV_Sales_Quantity'] = 0 
    if 'Vehicle_Class' in df_2025.columns:
        df_2025.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
    df_2025['Vehicle_Category'] = df_2025['Vehicle_Category'].fillna('Unknown')
    return df_2025

@st.cache_data
def run_classical_predictions(df_target: pd.DataFrame) -> pd.DataFrame:
    print("\nRunning Unified Classical Forecast...")
    all_preds = []
    df_history = pd.read_csv(DATA_PATH, parse_dates=['Date'], low_memory=False)
    if 'Vehicle_Class' in df_history.columns:
        df_history.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
    
    categories = df_target['Vehicle_Category'].unique()
    standard_cats = [c for c in categories if c not in ['3-Wheelers', 'Bus']]
    if standard_cats:
        df_featured_standard = create_advanced_features(df_target.copy())
    
    for category in categories:
        try:
            if category in ['3-Wheelers', 'Bus']:
                print(f"🚀 Running Specialized Monthly Model for {category}...")
                model_path = MODEL_3W_PATH if category == '3-Wheelers' else MODEL_BUS_PATH
                if not model_path.exists(): continue
                
                model = joblib.load(model_path)
                hist_cat = df_history[df_history['Vehicle_Category'] == category].copy()
                hist_cat['YearMonth'] = hist_cat['Date'].dt.to_period('M')
                monthly_hist = hist_cat.groupby(['State', 'YearMonth']).agg({'EV_Sales_Quantity': 'sum', 'Date': 'first'}).reset_index()
                monthly_hist['Date'] = monthly_hist['YearMonth'].dt.to_timestamp()
                monthly_hist = create_monthly_features_exact(monthly_hist)
                
                future_dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='MS')
                future_rows = []
                states = monthly_hist['State'].unique()
                
                for state in states:
                    state_data = monthly_hist[monthly_hist['State'] == state].sort_values('Date')
                    if state_data.empty: continue
                    last_row = state_data.iloc[-1]
                    base_feats = {
                        'state_avg': last_row['state_avg'], 'momentum': last_row['momentum'],
                        'state_encoded': last_row.get('state_encoded', 0)
                    }
                    for w in [3, 6, 12]:
                        for s in ['mean', 'std', 'max']:
                            base_feats[f'roll_{s}_{w}m'] = last_row.get(f'roll_{s}_{w}m', 0)

                    for date in future_dates:
                        lag_1 = last_row['EV_Sales_Quantity']
                        row = {
                            'Date': date, 'State': state, 'Vehicle_Category': category,
                            'month': date.month, 'year': date.year, 'quarter': date.quarter,
                            'lag_month_1': lag_1, 'lag_month_2': lag_1, 'lag_month_3': lag_1, 
                            'lag_month_6': lag_1, 'lag_month_12': lag_1, **base_feats
                        }
                        future_rows.append(row)
                        
                df_future_monthly = pd.DataFrame(future_rows)
                state_means = monthly_hist.groupby('State')['EV_Sales_Quantity'].mean()
                df_future_monthly['state_encoded'] = df_future_monthly['State'].map(state_means).fillna(0)
                
                feature_cols = [
                    'month', 'year', 'quarter',
                    'lag_month_1', 'lag_month_2', 'lag_month_3', 'lag_month_6', 'lag_month_12',
                    'roll_mean_3m', 'roll_std_3m', 'roll_max_3m',
                    'roll_mean_6m', 'roll_std_6m', 'roll_max_6m',
                    'roll_mean_12m', 'roll_std_12m', 'roll_max_12m',
                    'momentum', 'state_avg', 'state_encoded'
                ]
                
                preds = model.predict(df_future_monthly[feature_cols])
                df_future_monthly['Monthly_Pred'] = np.maximum(preds, 0)
                
                df_distributed = []
                for _, row in df_future_monthly.iterrows():
                    days_in_month = pd.Period(row['Date'], freq='M').days_in_month
                    daily_val = row['Monthly_Pred'] / days_in_month
                    dates = pd.date_range(start=row['Date'], periods=days_in_month, freq='D')
                    temp = pd.DataFrame({'Date': dates, 'State': row['State'], 'Vehicle_Category': category, 'Predicted_Sales': int(daily_val)})
                    df_distributed.append(temp)
                
                if df_distributed: all_preds.append(pd.concat(df_distributed))

            else:
                cat_filename = category.replace(" ", "_").replace("/", "_")
                model_path = MODELS_DIR / f"{CLASSICAL_MODEL_PREFIX}{cat_filename}.pkl"
                if not model_path.exists(): continue
                
                df_cat = df_featured_standard[df_featured_standard['Vehicle_Category'] == category].copy()
                if df_cat.empty: continue
                
                with open(model_path, 'rb') as f: model_data = pickle.load(f)
                X_scaled, _, _, _ = prepare_data_for_training(df_cat, feature_subset=model_data['feature_names'])
                if X_scaled.shape[1] == len(model_data['feature_names']):
                    preds = model_data['primary_model'].predict(X_scaled)
                    df_cat['Predicted_Sales'] = np.maximum(0, preds).astype(int)
                    all_preds.append(df_cat[['Date', 'State', 'Vehicle_Category', 'Predicted_Sales']])

        except Exception as e:
            print(f"Error predicting {category}: {e}")
            continue

    if not all_preds: return pd.DataFrame()
    return pd.concat(all_preds, ignore_index=True)

@st.cache_data
def run_qml_predictions(df: pd.DataFrame) -> pd.DataFrame:
    print("\nRunning predictions with QML model...")
    if not QML_MODEL_PATH.exists() or not QML_PARAMS_PATH.exists():
        return pd.DataFrame()
    
    df_qml = df.copy()
    norm_params = load_qml_params(QML_PARAMS_PATH)
    input_dim = norm_params.get('input_dim')
    
    # --- ROBUST INFERENCE CALL ---
    X_tensor = prepare_data_for_inference(df_qml, norm_params)
    
    model = load_qml_model(QML_MODEL_PATH, input_dim)
    with torch.no_grad():
        predictions_norm = model(X_tensor).cpu().numpy().flatten()
    
    y_mean, y_std = norm_params['y_mean'], norm_params['y_std']
    predictions = (predictions_norm * y_std) + y_mean
    df_qml['Predicted_Sales'] = np.maximum(0, predictions).astype(int)
    
    return df_qml[['Date', 'State', 'Vehicle_Category', 'Predicted_Sales']]

def generate_agent_report(predictions_df, model_type):
    global report_agent
    if predictions_df.empty: return "No data."
    if not LLM_CLIENT: return _generate_fallback_insights_text(predictions_df, model_type, "Agent not active")
    
    total = predictions_df['Predicted_Sales'].sum()
    states = predictions_df.groupby('State')['Predicted_Sales'].sum().sort_values(ascending=False).head(5).to_dict()
    cats = predictions_df.groupby('Vehicle_Category')['Predicted_Sales'].sum().sort_values(ascending=False).head(5).to_dict()
    summary = f"Total: {total}\nTop States: {states}\nTop Cats: {cats}"
    
    try:
        res = report_agent.invoke(model_type, summary)
        return f"### {model_type} Insights\n{res.choices[0].message.content}"
    except Exception as e:
        return _generate_fallback_insights_text(predictions_df, model_type, str(e))

def _generate_fallback_insights_text(df, model, error):
    total = df['Predicted_Sales'].sum()
    
    # Clean up the error message for display
    if "402" in str(error) or "exceeded" in str(error):
        display_error = "AI Agent usage limit reached (Free Tier). Showing raw metrics."
    else:
        display_error = str(error)

    return dedent(f"""
        ### {model} Analysis (Metrics Only)
        
        **Status:** {display_error}
        
        - **Total Forecasted Sales:** {total:,} units
        - **Data Source:** {model} Predictive Model
        
        *Note: Upgrade API key to restore full AI executive summaries.*
    """)

@st.cache_data
def generate_on_demand_forecast(category: str, state: str, days_to_forecast: int):
    # === PATH A: SPECIALIZED (Bus/3W) ===
    if category in ['3-Wheelers', 'Bus']:
        model_path = MODEL_3W_PATH if category == '3-Wheelers' else MODEL_BUS_PATH
        if not model_path.exists(): return None, "Model not found."
        
        # Determine months needed
        months_needed = int(np.ceil(days_to_forecast / 28)) + 1
        
        model = joblib.load(model_path)
        df_hist = pd.read_csv(DATA_PATH, parse_dates=['Date'])
        df_hist = df_hist[(df_hist['Vehicle_Category'] == category) & (df_hist['State'] == state)].copy()
        if df_hist.empty: return None, f"No data for {category} in {state}"

        df_hist['YearMonth'] = df_hist['Date'].dt.to_period('M')
        monthly_hist = df_hist.groupby(['State', 'YearMonth']).agg({'EV_Sales_Quantity':'sum', 'Date':'first'}).reset_index()
        monthly_hist['Date'] = monthly_hist['YearMonth'].dt.to_timestamp()
        monthly_hist = create_monthly_features_exact(monthly_hist)
        last_row = monthly_hist.iloc[-1]
        
        future_rows = []
        last_date = last_row['Date']
        
        # Forward generation
        for i in range(1, months_needed + 1):
            next_date = last_date + pd.DateOffset(months=i)
            row = {
                'Date': next_date, 'State': state, 
                'month': next_date.month, 'year': next_date.year, 'quarter': next_date.quarter,
                'lag_month_1': last_row['EV_Sales_Quantity'],
                'lag_month_2': last_row['EV_Sales_Quantity'], 'lag_month_3': last_row['EV_Sales_Quantity'],
                'lag_month_6': last_row['EV_Sales_Quantity'], 'lag_month_12': last_row['EV_Sales_Quantity'],
                'roll_mean_3m': last_row.get('roll_mean_3m', 0), 'roll_std_3m': last_row.get('roll_std_3m', 0), 'roll_max_3m': last_row.get('roll_max_3m', 0),
                'roll_mean_6m': last_row.get('roll_mean_6m', 0), 'roll_std_6m': last_row.get('roll_std_6m', 0), 'roll_max_6m': last_row.get('roll_max_6m', 0),
                'roll_mean_12m': last_row.get('roll_mean_12m', 0), 'roll_std_12m': last_row.get('roll_std_12m', 0), 'roll_max_12m': last_row.get('roll_max_12m', 0),
                'momentum': last_row.get('momentum', 1), 'state_avg': last_row.get('state_avg', 0), 'state_encoded': last_row.get('state_encoded', 0)
            }
            future_rows.append(row)
            
        future_monthly_df = pd.DataFrame(future_rows)
        feature_cols = [
            'month', 'year', 'quarter', 'lag_month_1', 'lag_month_2', 'lag_month_3', 'lag_month_6', 'lag_month_12',
            'roll_mean_3m', 'roll_std_3m', 'roll_max_3m', 'roll_mean_6m', 'roll_std_6m', 'roll_max_6m',
            'roll_mean_12m', 'roll_std_12m', 'roll_max_12m', 'momentum', 'state_avg', 'state_encoded'
        ]
        preds = model.predict(future_monthly_df[feature_cols])
        future_monthly_df['Forecasted_Monthly_Sales'] = np.maximum(preds, 0).astype(int)
        
        # --- DISTRIBUTION LOGIC (Monthly -> Daily with variation) ---
        daily_rows = []
        for _, row in future_monthly_df.iterrows():
            month_date = row['Date']
            total_sales = row['Forecasted_Monthly_Sales']
            days_in_month = pd.Period(month_date, freq='M').days_in_month
            
            # Random noise centered around 1.0 to create variety
            noise = np.random.uniform(0.8, 1.2, days_in_month)
            weights = noise / noise.sum()
            
            # Distribute total sales
            daily_float = total_sales * weights
            daily_int = np.floor(daily_float).astype(int)
            
            # Add remainder to random days to match exact total
            remainder = total_sales - daily_int.sum()
            if remainder > 0:
                daily_int[np.random.choice(days_in_month, int(remainder), replace=False)] += 1
                
            dates = pd.date_range(start=month_date, periods=days_in_month, freq='D')
            for d, sales in zip(dates, daily_int):
                daily_rows.append({'Date': d, 'Forecasted_Sales': sales})
                
        daily_future_df = pd.DataFrame(daily_rows).head(days_to_forecast)
        
        # Now returns a line chart (daily data) instead of bar chart
        fig = px.line(daily_future_df, x='Date', y='Forecasted_Sales', 
                      title=f"Forecast: {category} in {state}")
        return daily_future_df[['Date', 'Forecasted_Sales']], fig

    else:
        # Standard Daily Model Logic
        cat_filename = category.replace(" ", "_").replace("/", "_")
        model_path = MODELS_DIR / f"{CLASSICAL_MODEL_PREFIX}{cat_filename}.pkl"
        if not model_path.exists(): return None, "Model not found."
        
        with open(model_path, 'rb') as f: data = pickle.load(f)
        
        df_hist = pd.read_csv(DATA_PATH, parse_dates=['Date'])
        df_hist = df_hist[(df_hist['Vehicle_Category'] == category) & (df_hist['State'] == state)]
        
        # FIX: Ensure we start forecasting from a relevant date
        # If history ends long ago, we jump to "Tomorrow" to give a useful forecast
        last_date_hist = df_hist['Date'].max()
        today = pd.Timestamp.now().normalize()
        
        if last_date_hist < today - pd.Timedelta(days=30):
            # Gap in data? Start forecast from Today
            start_date = today
        else:
            # Continuous data? Start from end of history
            start_date = last_date_hist + pd.Timedelta(days=1)
            
        future_dates = pd.date_range(start=start_date, periods=days_to_forecast)
        future_df = pd.DataFrame({'Date': future_dates, 'State': state, 'Vehicle_Category': category, 'EV_Sales_Quantity': 0})
        
        # Combine for feature engineering
        combined = pd.concat([df_hist, future_df], ignore_index=True)
        
        df_feat = create_advanced_features(combined)
        future_feat = df_feat.iloc[-days_to_forecast:].copy()
        
        X_scaled, _, _, _ = prepare_data_for_training(future_feat, feature_subset=data['feature_names'])
        preds = data['primary_model'].predict(X_scaled)
        
        future_df['Forecasted_Sales'] = np.maximum(preds, 0).astype(int)
        
        fig = px.line(future_df, x='Date', y='Forecasted_Sales', 
                      title=f"Daily Forecast: {category} in {state}")
        return future_df[['Date', 'Forecasted_Sales']], fig