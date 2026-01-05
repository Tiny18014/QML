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

MODEL_HYBRID_PATH = MODELS_DIR / "advanced_model_monthly_hybrid.pkl"

warnings.filterwarnings('ignore')

# --- Feature Engineering ---
try:
    from advanced_model_trainer import create_monthly_features
except ImportError:
    print("⚠️ Could not import advanced_model_trainer.")

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
    """Runs Hybrid Monthly Model on history to simulate daily predictions."""
    print("⏳ Pre-computing predictions (Model Playback)...")

    if not MODEL_HYBRID_PATH.exists():
        print("❌ Hybrid Model not found. Run training first.")
        return pd.DataFrame()

    try:
        with open(MODEL_HYBRID_PATH, 'rb') as f:
            models_data = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    if 'Vehicle_Class' in df.columns:
        df.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)

    # 1. Aggregate to Monthly
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    monthly_agg = df.groupby(['State', 'Vehicle_Category', 'Year', 'Month'])['EV_Sales_Quantity'].sum().reset_index()
    monthly_agg['Date'] = pd.to_datetime(monthly_agg[['Year', 'Month']].assign(Day=1))

    # 2. Create Features
    monthly_featured = create_monthly_features(monthly_agg)
    
    all_daily_preds = []
    
    # 3. Predict & Distribute
    for cat, model_data in models_data.items():
        if cat not in monthly_featured['Vehicle_Category'].unique(): continue
        
        # Filter & Prep
        cat_df = monthly_featured[monthly_featured['Vehicle_Category'] == cat].copy()
        
        # State Encoding match
        train_states = model_data.get('states', [])
        cat_df['State_Code'] = pd.Categorical(cat_df['State'], categories=train_states).codes
        
        feature_names = model_data['features']
        scaler = model_data['scaler']
        model = model_data['model']
        patterns = model_data['daily_patterns']
        
        # Predict Monthly Total
        X = cat_df[feature_names]
        X_scaled = scaler.transform(X)
        cat_df['Monthly_Pred'] = np.maximum(model.predict(X_scaled), 0)

        # Distribute to Daily (Vectorized where possible, but iterative for patterns is safer)
        # We need to map these monthly preds back to the daily rows in original df

        # Create a lookup for Monthly Preds
        pred_map = cat_df.set_index(['State', 'Year', 'Month'])['Monthly_Pred'].to_dict()

        # Get daily rows for this category
        daily_cat = df[df['Vehicle_Category'] == cat].copy()

        # Function to apply weight
        def get_pred(row):
            key = (row['State'], row['Year'], row['Month'])
            if key in pred_map:
                monthly_total = pred_map[key]
                day = row['Date'].day
                # Get weight for this month/day
                month_pats = patterns.get(row['Month'], {})
                weight = month_pats.get(day, 1.0/30.0) # Default if missing
                return int(monthly_total * weight)
            return 0

        daily_cat['Predicted_Sales'] = daily_cat.apply(get_pred, axis=1)
        all_daily_preds.append(daily_cat[['Date', 'State', 'Vehicle_Category', 'EV_Sales_Quantity', 'Predicted_Sales']])

    if not all_daily_preds:
        return pd.DataFrame()

    return pd.concat(all_daily_preds, ignore_index=True)

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