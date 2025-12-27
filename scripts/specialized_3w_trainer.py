#!/usr/bin/env python3
"""
Fixed Specialized 3-Wheeler Trainer
CRITICAL FIX: Excludes 2025 data before training
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import warnings
import joblib

warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "specialized_3w_monthly_model.pkl"

def load_and_aggregate_monthly(path):
    print("🔄 Loading and Aggregating Data to Monthly Frequency...")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 🔥 CRITICAL FIX: Exclude 2025 data
    print(f"   Before filtering: {len(df)} records")
    df = df[df['Date'] < '2025-01-01'].copy()
    print(f"   After filtering (2021-2024): {len(df)} records")
    
    # Filter for 3-Wheelers
    df = df[df['Vehicle_Category'] == '3-Wheelers'].copy()
    
    # Create Year-Month column for grouping
    df['YearMonth'] = df['Date'].dt.to_period('M')
    
    # Aggregate: Sum sales per state per month
    monthly_df = df.groupby(['State', 'YearMonth']).agg({
        'EV_Sales_Quantity': 'sum',
        'Date': 'first'
    }).reset_index()
    
    # Convert YearMonth back to timestamp
    monthly_df['Date'] = monthly_df['YearMonth'].dt.to_timestamp()
    monthly_df = monthly_df.sort_values(['State', 'Date'])
    
    print(f"📊 Original Rows: {len(df)} -> Monthly Rows: {len(monthly_df)}")
    print(f"   Date Range: {monthly_df['Date'].min().date()} to {monthly_df['Date'].max().date()}")
    print(f"   Max Monthly Sales: {monthly_df['EV_Sales_Quantity'].max()}")
    print(f"   Avg Monthly Sales: {monthly_df['EV_Sales_Quantity'].mean():.2f}")
    
    return monthly_df

def create_monthly_features(df):
    print("🔧 Creating Monthly Features...")
    df = df.copy()
    
    # Time Features
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['quarter'] = df['Date'].dt.quarter
    
    # Lags (Previous Months)
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_month_{lag}'] = df.groupby('State')['EV_Sales_Quantity'].shift(lag)
    
    # Rolling Trends
    for w in [3, 6, 12]:
        g = df.groupby('State')['EV_Sales_Quantity']
        df[f'roll_mean_{w}m'] = g.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'roll_std_{w}m'] = g.rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True)
        df[f'roll_max_{w}m'] = g.rolling(window=w, min_periods=1).max().reset_index(level=0, drop=True)

    # Momentum
    df['momentum'] = df['roll_mean_3m'] / (df['roll_mean_12m'] + 1)
    
    # Expanding Mean
    df['state_avg'] = df.groupby('State')['EV_Sales_Quantity'].expanding().mean().reset_index(level=0, drop=True)

    # Clean
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(0)
    
    return df

def train_monthly_model():
    print("🚀 Monthly 3-Wheeler Trainer Started (Fixed)")
    print("=" * 60)
    
    # 1. Load & Aggregate
    df = load_and_aggregate_monthly(DATA_PATH)
    
    if len(df) < 50:
        print("❌ Insufficient data after filtering. Need at least 50 monthly records.")
        return
    
    # 2. Features
    df_processed = create_monthly_features(df)
    
    # 3. Split - Use last 6 months for testing (more reliable than 3)
    split_date = df_processed['Date'].max() - pd.DateOffset(months=6)
    print(f"\n📅 Splitting at {split_date.date()} (Last 6 months for Test)")
    
    train_df = df_processed[df_processed['Date'] <= split_date].copy()
    test_df = df_processed[df_processed['Date'] > split_date].copy()
    
    print(f"   Train: {len(train_df)} months ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"   Test: {len(test_df)} months ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
    
    if len(test_df) < 5:
        print("⚠️ Warning: Very small test set. Results may not be reliable.")
    
    y_train = train_df['EV_Sales_Quantity']
    y_test = test_df['EV_Sales_Quantity']
    
    # 4. Prepare X
    exclude = ['Date', 'State', 'YearMonth', 'EV_Sales_Quantity']
    features = [c for c in train_df.columns if c not in exclude]
    
    X_train = train_df[features]
    X_test = test_df[features]
    
    # State encoding (Target Encoding)
    state_means = train_df.groupby('State')['EV_Sales_Quantity'].mean()
    X_train['state_encoded'] = train_df['State'].map(state_means)
    X_test['state_encoded'] = test_df['State'].map(state_means).fillna(train_df['EV_Sales_Quantity'].mean())
    
    features.append('state_encoded')
    
    print(f"\n📊 Features: {len(features)}")
    print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 5. Model
    print("\n🧠 Training LightGBM on Monthly Aggregates...")
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=20,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train[features], y_train)
    
    # 6. Evaluate
    preds = model.predict(X_test[features])
    preds = np.maximum(preds, 0)
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(np.mean((y_test - preds) ** 2))
    
    print("\n" + "="*60)
    print(f"🏆 Final Results (Monthly Model):")
    print(f"   MAE : {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²  : {r2:.4f}")
    print("="*60)
    
    # Show sample predictions
    print("\n🔍 Sample Predictions:")
    res = pd.DataFrame({
        'State': test_df['State'], 
        'Date': test_df['Date'], 
        'Actual': y_test, 
        'Predicted': preds.round(1)
    })
    print(res.head(10).to_string(index=False))
    
    # Quality check
    if r2 < 0:
        print("\n❌ Model performance is poor (negative R²)")
        print("   This usually means:")
        print("   - Test data has very different patterns than training")
        print("   - Feature engineering needs improvement")
        print("   - Model needs more regularization")
        return
    elif r2 < 0.3:
        print("\n⚠️ Model performance is below expectations (R² < 0.3)")
        print("   Consider gathering more data or improving features")
    elif r2 < 0.6:
        print("\n✅ Model performance is acceptable (R² = {:.2f})".format(r2))
    else:
        print("\n🎉 Model performance is good! (R² = {:.2f})".format(r2))
    
    # Save model
    print(f"\n💾 Saving model to {MODEL_PATH}")
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, MODEL_PATH)
    print("✅ Model saved successfully!")

if __name__ == "__main__":
    train_monthly_model()