"""
Dashboard Utilities
===================
REFACTOR v14: FINAL PRODUCTION FIX.
1. Explicit Lag Generation (verified via debug script).
2. Explicit Sorting & Grouping for Recursive Loop.
3. Manual State Encoding Injection.
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

sys.path.append(str(ROOT_DIR / "scripts"))

# CRITICAL IMPORTS
try:
    from advanced_model_trainer import create_advanced_features, prepare_data_for_training
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

# --- HELPER: Manual Feature Injection (The Fix) ---
def add_lag_features_manual(df):
    """Manually calculates lags to ensure model has correct history context."""
    df = df.copy()
    # Ensure strictly sorted by State then Date
    df = df.sort_values(['State', 'Vehicle_Category', 'Date'])
    
    grouped = df.groupby(['State', 'Vehicle_Category'])
    
    # Explicitly recreate the exact features the model expects
    df['lag_1_day'] = grouped['EV_Sales_Quantity'].shift(1)
    df['lag_7_days'] = grouped['EV_Sales_Quantity'].shift(7)
    df['rolling_mean_7_days'] = grouped['EV_Sales_Quantity'].shift(1).rolling(window=7, min_periods=1).mean()
    df['rolling_std_7_days'] = grouped['EV_Sales_Quantity'].shift(1).rolling(window=7, min_periods=1).std()
    df['rolling_mean_30_days'] = grouped['EV_Sales_Quantity'].shift(1).rolling(window=30, min_periods=1).mean()
    
    # Important: Fill NAs with 0 so we don't drop rows
    cols = ['lag_1_day', 'lag_7_days', 'rolling_mean_7_days', 'rolling_std_7_days', 'rolling_mean_30_days']
    df[cols] = df[cols].fillna(0)
    
    return df

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
    # QML Model deprecated in favor of LightGBM Hybrid
    return pd.DataFrame()

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
# --- HELPER: Get Historical Weights ---
def get_daily_weights(df_hist, state, category, month):
    """Calculate daily sales distribution from 2024 data."""
    try:
        target_year = 2024
        mask = (
            (df_hist['State'] == state) & 
            (df_hist['Vehicle_Category'] == category) & 
            (df_hist['Date'].dt.year == target_year) & 
            (df_hist['Date'].dt.month == month)
        )
        hist_data = df_hist[mask].sort_values('Date')
        
        if not hist_data.empty and hist_data['EV_Sales_Quantity'].sum() > 0:
            total = hist_data['EV_Sales_Quantity'].sum()
            return hist_data['EV_Sales_Quantity'].values / total
    except Exception:
        pass
    return None

# --- HELPER: Distribute Monthly to Daily ---
def distribute_monthly_sales(total_sales, year, month, weights=None):
    """Distributes monthly total into daily values."""
    start_date = pd.Timestamp(year=year, month=month, day=1)
    days_in_month = start_date.days_in_month
    dates = pd.date_range(start=start_date, periods=days_in_month, freq='D')
    
    if total_sales <= 0:
        return pd.DataFrame({'Date': dates, 'Forecasted_Sales': 0})

    if weights is not None and len(weights) == days_in_month:
        daily_sales = (total_sales * weights).astype(int)
    else:
        # Fallback: Random Noise
        noise = np.random.uniform(0.8, 1.2, size=days_in_month)
        weights = noise / noise.sum()
        daily_sales = (total_sales * weights).astype(int)
    
    # Fix Rounding
    diff = total_sales - daily_sales.sum()
    if diff > 0:
        indices = np.random.choice(days_in_month, int(diff), replace=False)
        daily_sales[indices] += 1
        
    return pd.DataFrame({'Date': dates, 'Forecasted_Sales': daily_sales})

@st.cache_data
def generate_on_demand_forecast(category: str, state: str, days: int):
    """
    Robust On-Demand Forecast Logic.
    Prioritizes the Advanced Monthly Hybrid Model for all categories.
    """
    dates = pd.date_range(start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=days)
    source = "Unknown"
    df_final = pd.DataFrame()

    # 1. Load History & Calc Baseline Stats
    try:
        df_hist = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    except Exception:
        df_hist = pd.DataFrame()

    avg_daily_sales = 10
    growth_factor = 1.02
    
    if not df_hist.empty:
        subset = df_hist[(df_hist['State']==state) & (df_hist['Vehicle_Category']==category)]
        if not subset.empty:
            valid_sales = subset[subset['EV_Sales_Quantity'] > 0]
            if not valid_sales.empty:
                avg_daily_sales = valid_sales.tail(30)['EV_Sales_Quantity'].mean()
            else:
                avg_daily_sales = subset.tail(30)['EV_Sales_Quantity'].mean()
                
            if np.isnan(avg_daily_sales) or avg_daily_sales < 1: 
                avg_daily_sales = 5

    # --- STRATEGY A: MONTHLY HYBRID MODEL (The Hero Model) ---
    hybrid_model_path = MODELS_DIR / "advanced_model_monthly_hybrid.pkl"

    if hybrid_model_path.exists():
        try:
            with open(hybrid_model_path, 'rb') as f:
                models_data = pickle.load(f)

            if category in models_data:
                source = f"Hybrid Monthly Model ({category})"
                model_data = models_data[category]
                model = model_data['model']
                scaler = model_data['scaler']
                feature_names = model_data['features']
                patterns = model_data['daily_patterns']
                
                # We need to simulate monthly predictions for the span of 'days'
                # 1. Determine months involved
                start_date = dates[0]
                end_date = dates[-1]
                future_months = pd.date_range(start=start_date.replace(day=1), end=end_date.replace(day=1), freq='MS')
                if len(future_months) == 0: future_months = [start_date.replace(day=1)]
                
                # 2. Prepare History for Lags (Monthly Aggregation)
                cat_hist = df_hist[df_hist['Vehicle_Category'] == category].copy()
                cat_hist['Year'] = cat_hist['Date'].dt.year
                cat_hist['Month'] = cat_hist['Date'].dt.month
                monthly_hist = cat_hist.groupby(['State', 'Vehicle_Category', 'Year', 'Month'])['EV_Sales_Quantity'].sum().reset_index()
                monthly_hist['Date'] = pd.to_datetime(monthly_hist[['Year', 'Month']].assign(Day=1))
                
                # Filter for this state
                state_history = monthly_hist[monthly_hist['State'] == state].sort_values('Date')
                
                # 3. Recursive Forecasting Loop
                current_history = state_history.copy()
                all_daily_preds = []
                
                from advanced_model_trainer import create_monthly_features
                
                # Ensure we have enough history dummies if empty
                if current_history.empty:
                    # Create dummy history based on avg
                    dummy_dates = pd.date_range(end=start_date, periods=13, freq='MS')
                    current_history = pd.DataFrame({
                        'Date': dummy_dates,
                        'State': state,
                        'Vehicle_Category': category,
                        'Year': dummy_dates.year,
                        'Month': dummy_dates.month,
                        'EV_Sales_Quantity': avg_daily_sales * 30
                    })

                for m_date in future_months:
                    # Create temp row
                    new_row = pd.DataFrame([{
                        'State': state,
                        'Vehicle_Category': category,
                        'Date': m_date,
                        'Year': m_date.year,
                        'Month': m_date.month,
                        'EV_Sales_Quantity': 0
                    }])
                    
                    temp_df = pd.concat([current_history, new_row], ignore_index=True)
                    temp_features = create_monthly_features(temp_df)
                    row_to_predict = temp_features.iloc[[-1]].copy()
                    
                    # Encode State (using saved states list if available, else simple code)
                    # For on-demand, we rely on the robust scaler handling distribution,
                    # but state encoding is tricky if state is new.
                    # We try to map to known states or use 0.
                    known_states = model_data.get('states', [])
                    if state in known_states:
                        row_to_predict['State_Code'] = known_states.index(state)
                    else:
                        row_to_predict['State_Code'] = 0 # Default/Unknown

                    # Predict
                    X_pred = row_to_predict[feature_names]
                    X_pred_scaled = scaler.transform(X_pred)
                    pred_total = model.predict(X_pred_scaled)[0]
                    pred_total = max(0, pred_total)
                    
                    # Update History
                    current_history = pd.concat([
                        current_history,
                        pd.DataFrame([{
                            'State': state, 'Vehicle_Category': category,
                            'Date': m_date, 'Year': m_date.year, 'Month': m_date.month,
                            'EV_Sales_Quantity': pred_total
                        }])
                    ], ignore_index=True)
                    
                    # Distribute to Daily
                    days_in_month = pd.Period(m_date, freq='M').days_in_month
                    month_weights = patterns.get(m_date.month, {})
                    
                    daily_vals = []
                    range_start = m_date
                    for d in range(1, days_in_month + 1):
                        w = month_weights.get(d, 1.0/days_in_month)
                        val = int(pred_total * w)
                        daily_vals.append({
                            'Date': pd.Timestamp(year=m_date.year, month=m_date.month, day=d),
                            'Forecasted_Sales': val
                        })
                    
                    all_daily_preds.append(pd.DataFrame(daily_vals))
                
                if all_daily_preds:
                    df_final = pd.concat(all_daily_preds)
                    # Filter for requested date range
                    df_final = df_final[(df_final['Date'] >= dates[0]) & (df_final['Date'] <= dates[-1])]

        except Exception as e:
            print(f"Hybrid model forecast failed: {e}")
            df_final = pd.DataFrame()

    # --- STRATEGY B: FALLBACK ---
    if df_final.empty:
        source = "Statistical Projection (History Based)"
        current_val = avg_daily_sales
        preds = []
        for i in range(days):
            current_val *= (1 + (growth_factor - 1) / 30)
            seasonality = 1.1 if dates[i].dayofweek >= 5 else 0.95
            noise = np.random.uniform(0.9, 1.1)
            preds.append(max(0, int(current_val * seasonality * noise)))
        df_final = pd.DataFrame({'Date': dates, 'Forecasted_Sales': preds})

    title = f"{days}-Day Forecast: {category} in {state}"
    fig = px.line(df_final, x='Date', y='Forecasted_Sales', 
                  title=f"<b>{title}</b><br><i>Source: {source}</i>", 
                  template="plotly_white")
    fig.update_traces(line=dict(color='green', width=3), mode='lines+markers')
    
    return df_final, fig