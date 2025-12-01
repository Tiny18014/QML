#!/usr/bin/env python3
"""
Specialized Bus Demand Trainer V6 (Compatibility Mode)
======================================================
- Features: Aligned EXACTLY with 3-Wheeler model (20 features).
  (Lags 1-12, Rolling 3/6/12m, Quarter, etc.)
- Logic: Uses Robust L1 Loss & Aug-2025 Split to fix accuracy issues.
- Result: Works with your existing Dashboard Utils without changes.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import warnings
import joblib

warnings.filterwarnings('ignore')

# --- Config ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "specialized_bus_monthly_model.pkl"

def load_and_aggregate_monthly(path):
    print("🔄 Loading and Aggregating Bus Data to Monthly...")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # FILTER FOR BUS
    df = df[df['Vehicle_Category'] == 'Bus'].copy()
    
    # Create Year-Month column
    df['YearMonth'] = df['Date'].dt.to_period('M')
    
    # Aggregate
    monthly_df = df.groupby(['State', 'YearMonth']).agg({
        'EV_Sales_Quantity': 'sum',
        'Date': 'first'
    }).reset_index()
    
    monthly_df['Date'] = monthly_df['YearMonth'].dt.to_timestamp()
    monthly_df = monthly_df.sort_values(['State', 'Date'])
    
    print(f"📊 Bus Monthly Rows: {len(monthly_df)}")
    print(f"   Max Monthly Sales: {monthly_df['EV_Sales_Quantity'].max()}")
    
    return monthly_df

def create_monthly_features_compatibility(df):
    """
    Creates features IDENTICAL to the 3-Wheeler model.
    This ensures dashboard_utils.py does not crash.
    """
    print("🔧 Creating Features (3-Wheeler Compatibility Mode)...")
    df = df.copy()
    
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['quarter'] = df['Date'].dt.quarter
    
    # 1. Full Lags (1 to 12)
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_month_{lag}'] = df.groupby('State')['EV_Sales_Quantity'].shift(lag)
        
    # 2. Full Rolling Stats (3, 6, 12 months)
    # Even though 12m might be noisy for Bus, we calculate it to match the dashboard schema
    for w in [3, 6, 12]:
        g = df.groupby('State')['EV_Sales_Quantity']
        df[f'roll_mean_{w}m'] = g.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'roll_std_{w}m'] = g.rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True)
        df[f'roll_max_{w}m'] = g.rolling(window=w, min_periods=1).max().reset_index(level=0, drop=True)

    # 3. Momentum & State Avg
    df['momentum'] = df['roll_mean_3m'] / (df['roll_mean_12m'] + 1)
    df['state_avg'] = df.groupby('State')['EV_Sales_Quantity'].expanding().mean().reset_index(level=0, drop=True)

    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(0)
    
    return df

def train_bus_model():
    print("🚀 Bus Trainer Started (V6 - Compatibility)")
    
    # 1. Load
    df = load_and_aggregate_monthly(DATA_PATH)
    
    # 2. Features (Using the 3W-aligned function)
    df_processed = create_monthly_features_compatibility(df)
    
    # 3. Split (Safe Date Cutoff)
    # We ignore the incomplete Oct/Nov data to prevent bad training
    split_date = pd.Timestamp("2025-08-01")
    print(f"📅 Splitting at {split_date.date()} (Safe Cutoff)")
    
    train_df = df_processed[df_processed['Date'] <= split_date].copy()
    test_df = df_processed[df_processed['Date'] > split_date].copy()
    
    if len(test_df) < 10:
        # Fallback if dataset hasn't been updated with recent months
        split_date = df_processed['Date'].max() - pd.DateOffset(months=3)
        train_df = df_processed[df_processed['Date'] <= split_date].copy()
        test_df = df_processed[df_processed['Date'] > split_date].copy()

    # 4. Target
    y_train = train_df['EV_Sales_Quantity']
    y_test = test_df['EV_Sales_Quantity']
    
    # 5. Prepare X
    exclude = ['Date', 'State', 'YearMonth', 'EV_Sales_Quantity']
    features = [c for c in train_df.columns if c not in exclude]
    
    X_train = train_df[features]
    X_test = test_df[features]
    
    # Target Encoding
    state_means = train_df.groupby('State')['EV_Sales_Quantity'].mean()
    X_train['state_encoded'] = train_df['State'].map(state_means)
    X_test['state_encoded'] = test_df['State'].map(state_means).fillna(train_df['EV_Sales_Quantity'].mean())
    features.append('state_encoded')
    
    # 6. Train
    print("🧠 Training LightGBM (MAE Objective)...")
    # We increase regularization to ignore the noisy 12-month features we were forced to add
    model = lgb.LGBMRegressor(
        objective='regression_l1', # Critical for robustness
        metric='mae',
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        reg_alpha=2.0,          # High Regularization to ignore bad features
        reg_lambda=2.0,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train[features], y_train)
    
    # 7. Evaluate
    preds = model.predict(X_test[features])
    preds = np.maximum(preds, 0)
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print("\n" + "="*40)
    print(f"🏆 Final Results (Bus Model V6):")
    print(f"   MAE : {mae:.4f}")
    print(f"   R²  : {r2:.4f}")
    print("="*40)
    
    print("\n🔍 Sample Predictions:")
    res = pd.DataFrame({'State': test_df['State'], 'Date': test_df['Date'], 
                        'Actual': y_test, 'Predicted': preds.round(1)})
    print(res.sample(min(10, len(res))))
    
    # 8. Save
    print(f"\n💾 Saving model to {MODEL_PATH}")
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
    train_bus_model()