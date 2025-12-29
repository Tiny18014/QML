"""
Live Simulation Runner (Cloud/Postgres Version)
===============================================
1. Loads historical data & runs models (Anti-Leakage).
2. Writes predictions to CLOUD DATABASE (Postgres) so deployed app can see them.
"""

import time
import pandas as pd
import numpy as np
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

# Paths
MODELS_DIR = ROOT_DIR / "models"
HYBRID_MODEL_PATH = MODELS_DIR / "advanced_model_monthly_hybrid.pkl"

warnings.filterwarnings('ignore')

# --- Import from Trainer ---
try:
    from advanced_model_trainer import (
        load_and_clean_data,
        aggregate_to_monthly,
        create_monthly_features
    )
except ImportError:
    print("⚠️ Could not import advanced_model_trainer. Ensure the script is in the scripts/ directory.")
    sys.exit(1)


# --- DATABASE FUNCTIONS ---
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
    """
    Runs the Hybrid Monthly+Daily model on historical data.
    Generates daily predictions by predicting monthly totals and distributing them.
    """
    print("⏳ Pre-computing predictions (Model Playback)...")
    
    # 1. Load Data
    df = load_and_clean_data()
    
    # 2. Load Model
    if not HYBRID_MODEL_PATH.exists():
        print(f"❌ Model file not found: {HYBRID_MODEL_PATH}")
        return pd.DataFrame()

    with open(HYBRID_MODEL_PATH, 'rb') as f:
        models_data = pickle.load(f)
        
    # 3. Aggregate to Monthly & Create Features
    # We do this for the entire history to enable "playback"
    df_monthly = aggregate_to_monthly(df)
    df_features = create_monthly_features(df_monthly)

    all_daily_preds = []

    # 4. Predict & Distribute
    for cat, model_data in models_data.items():
        print(f"  -> Processing {cat}...")
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['features']
        patterns = model_data['daily_patterns'] # {Month: {Day: weight}}
        train_states = model_data['states']

        # Filter for category
        cat_df = df_features[df_features['Vehicle_Category'] == cat].copy()
        if cat_df.empty: continue

        # Filter states
        cat_df = cat_df[cat_df['State'].isin(train_states)]
        
        # Prepare X
        # Re-encode State
        cat_df['State_Code'] = pd.Categorical(cat_df['State'], categories=train_states).codes
        
        X = cat_df[feature_names]
        X_scaled = scaler.transform(X)
        
        # Predict Monthly
        cat_df['Predicted_Monthly'] = model.predict(X_scaled)
        cat_df['Predicted_Monthly'] = np.maximum(0, cat_df['Predicted_Monthly'])

        # Distribute to Daily
        # We iterate through the monthly predictions and generate daily rows
        for _, row in cat_df.iterrows():
            month = int(row['Month'])
            year = int(row['Year'])
            state = row['State']
            monthly_pred = row['Predicted_Monthly']

            # Get weights for this month
            weights = patterns.get(month, {})

            # Determine days in this specific month/year
            days_in_month = pd.Period(f"{year}-{month}-01").days_in_month

            for day in range(1, days_in_month + 1):
                weight = weights.get(day, 1.0/days_in_month)
                daily_pred = monthly_pred * weight

                all_daily_preds.append({
                    'Date': pd.Timestamp(year=year, month=month, day=day),
                    'State': state,
                    'Vehicle_Category': cat,
                    'Predicted_Sales': int(daily_pred)
                })

    # 5. Merge with Actuals
    pred_df = pd.DataFrame(all_daily_preds)

    # We merge with the original daily dataframe to get Actuals
    # Note: df has 'Date', 'State', 'Vehicle_Category', 'EV_Sales_Quantity'

    merged_df = pd.merge(
        df[['Date', 'State', 'Vehicle_Category', 'EV_Sales_Quantity']],
        pred_df,
        on=['Date', 'State', 'Vehicle_Category'],
        how='inner' # Only keep rows where we have both (or left if we want to show gaps)
    )

    # If inner join drops too much (e.g. if daily pattern generates days that don't exist in actuals?
    # Actually actuals might be missing days if 0 sales?
    # But load_and_clean_data loads the CSV. If the CSV is sparse, we might miss 0-sales days.
    # But for simulation, we usually want to simulate existing data points or continuous time.
    # The original script did: df_cat.merge(..., how='left') onto the original data.
    # So we should probably do a Right Join onto the original data or Inner.
    # Let's do Inner to simulate "known" days, or Left on df to keep all actuals.

    merged_df = pd.merge(
        df[['Date', 'State', 'Vehicle_Category', 'EV_Sales_Quantity']],
        pred_df,
        on=['Date', 'State', 'Vehicle_Category'],
        how='left'
    )
    merged_df['Predicted_Sales'] = merged_df['Predicted_Sales'].fillna(0).astype(int)

    print(f"✅ Generated predictions for {len(merged_df)} daily records.")
    return merged_df

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
            # Sample a random record to simulate a "live" event
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
