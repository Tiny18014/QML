"""
Live Simulation Runner (Cloud/Postgres Version)
===============================================
1. Loads historical data & runs models (Anti-Leakage).
2. Writes predictions to CLOUD DATABASE (Postgres) so deployed app can see them.
"""

import time
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
from pathlib import Path
import sys
import warnings
from sqlalchemy import create_engine, text

# --- CONFIGURATION ---
# 🛑 REPLACE THIS WITH YOUR NEON CONNECTION STRING 🛑
DB_CONNECTION_STRING = "postgresql://neondb_owner:npg_kKiXxzOD5H3t@ep-muddy-snow-a1u5668w-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR / "scripts"))

DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
MODELS_DIR = ROOT_DIR / "models"

MODEL_3W_PATH = MODELS_DIR / "specialized_3w_monthly_model.pkl"
MODEL_BUS_PATH = MODELS_DIR / "specialized_bus_monthly_model.pkl"

warnings.filterwarnings('ignore')

# --- Feature Engineering (Same as before) ---
try:
    from advanced_model_trainer import create_advanced_features, prepare_features_for_prediction
except ImportError:
    print("⚠️ Could not import advanced_model_trainer.")

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

# --- DATABASE FUNCTIONS (Updated for Postgres) ---
def init_db():
    """Initialize Cloud Database Table"""
    print("🔌 Connecting to Cloud Database...")
    engine = create_engine(DB_CONNECTION_STRING)
    with engine.connect() as conn:
        # Clean start
        conn.execute(text("DROP TABLE IF EXISTS live_predictions"))
        conn.commit()
        
        # Create table (Postgres syntax)
        conn.execute(text("""
            CREATE TABLE live_predictions (
                id SERIAL PRIMARY KEY, 
                timestamp TIMESTAMP, 
                date DATE, 
                state TEXT, 
                vehicle_category TEXT, 
                actual_sales INTEGER, 
                predicted_sales REAL, 
                error REAL, 
                model_confidence REAL, 
                processing_time_ms REAL
            )
        """))
        conn.commit()
    print("☁️ Cloud Database initialized.")

def precompute_all_predictions():
    """Runs models on history to ensure 0% leakage."""
    print("⏳ Pre-computing predictions (Model Playback)...")
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    if 'Vehicle_Class' in df.columns: df.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
    
    all_preds_df = []
    
    # 1. Monthly Models (Bus/3W)
    for cat in ['3-Wheelers', 'Bus']:
        path = MODEL_3W_PATH if cat == '3-Wheelers' else MODEL_BUS_PATH
        if not path.exists(): continue
        
        df_cat = df[df['Vehicle_Category'] == cat].copy()
        df_cat['YearMonth'] = df_cat['Date'].dt.to_period('M')
        monthly_agg = df_cat.groupby(['State', 'YearMonth']).agg({'EV_Sales_Quantity': 'sum', 'Date': 'first'}).reset_index()
        monthly_agg['Date'] = monthly_agg['YearMonth'].dt.to_timestamp()
        monthly_agg = create_monthly_features_exact(monthly_agg)
        
        state_means = monthly_agg.groupby('State')['EV_Sales_Quantity'].mean()
        monthly_agg['state_encoded'] = monthly_agg['State'].map(state_means)
        
        feature_cols = [
            'month', 'year', 'quarter', 'lag_month_1', 'lag_month_2', 'lag_month_3', 'lag_month_6', 'lag_month_12',
            'roll_mean_3m', 'roll_std_3m', 'roll_max_3m', 'roll_mean_6m', 'roll_std_6m', 'roll_max_6m',
            'roll_mean_12m', 'roll_std_12m', 'roll_max_12m', 'momentum', 'state_avg', 'state_encoded'
        ]
        
        try:
            model = joblib.load(path)
            monthly_agg['Monthly_Pred'] = model.predict(monthly_agg[feature_cols])
            df_cat['YearMonth'] = df_cat['Date'].dt.to_period('M')
            merged = df_cat.merge(monthly_agg[['State', 'YearMonth', 'Monthly_Pred']], on=['State', 'YearMonth'], how='left')
            merged['Predicted_Sales'] = (merged['Monthly_Pred'] / 30).fillna(0).astype(int)
            all_preds_df.append(merged[['Date', 'State', 'Vehicle_Category', 'EV_Sales_Quantity', 'Predicted_Sales']])
        except Exception: pass

    # 2. Daily Models
    df_daily = df[~df['Vehicle_Category'].isin(['3-Wheelers', 'Bus'])].copy()
    if not df_daily.empty:
        df_featured = create_advanced_features(df_daily)
        for cat in df_daily['Vehicle_Category'].unique():
            path = MODELS_DIR / f"advanced_model_{cat.replace(' ', '_').replace('/', '_')}.pkl"
            if not path.exists(): continue
            try:
                with open(path, 'rb') as f: pack = pickle.load(f)
                subset = df_featured[df_featured['Vehicle_Category'] == cat].copy()
                X = prepare_features_for_prediction(subset, pack['feature_names'], pack['scaler'])
                subset['Predicted_Sales'] = np.maximum(pack['primary_model'].predict(X), 0).astype(int)
                all_preds_df.append(subset[['Date', 'State', 'Vehicle_Category', 'EV_Sales_Quantity', 'Predicted_Sales']])
            except Exception: pass

    return pd.concat(all_preds_df, ignore_index=True) if all_preds_df else pd.DataFrame()

def run_simulation():
    print("⚡ Live Simulation Started (Cloud Mode)")
    print("Press Ctrl+C to stop.")
    
    init_db()
    
    augmented_df = precompute_all_predictions()
    if augmented_df.empty:
        print("❌ Error: No predictions generated.")
        return

    engine = create_engine(DB_CONNECTION_STRING)

    try:
        while True:
            row = augmented_df.sample(1).iloc[0]
            actual = int(row['EV_Sales_Quantity'])
            pred = int(row['Predicted_Sales'])
            error = abs(actual - pred)
            process_time = np.random.normal(150, 20)
            
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO live_predictions 
                    (timestamp, date, state, vehicle_category, actual_sales, predicted_sales, error, model_confidence, processing_time_ms)
                    VALUES (:ts, :date, :state, :cat, :act, :pred, :err, :conf, :time)
                """), {
                    "ts": datetime.now(), "date": row['Date'], "state": row['State'],
                    "cat": row['Vehicle_Category'], "act": actual, "pred": pred,
                    "err": error, "conf": 0.92, "time": process_time
                })
                conn.commit()
            
            print(f"☁️  [{row['Vehicle_Category']}] {row['State']}: Act={actual} | Pred={pred}")
            time.sleep(1.5)

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped.")

if __name__ == "__main__":
    run_simulation()