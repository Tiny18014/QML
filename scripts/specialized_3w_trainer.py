#!/usr/bin/env python3
"""
Specialized 3-Wheeler Trainer V4 (Monthly Aggregation Fix)
Solves the "Daily vs Monthly" mismatch by aggregating all history to Monthly totals.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "specialized_3w_monthly_model.pkl"

def load_and_aggregate_monthly(path):
    print("🔄 Loading and Aggregating Data to Monthly Frequency...")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter for 3-Wheelers
    df = df[df['Vehicle_Category'] == '3-Wheelers'].copy()
    
    # Create Year-Month column for grouping
    df['YearMonth'] = df['Date'].dt.to_period('M')
    
    # Aggregate: Sum sales, take first of other columns
    # We group by State and YearMonth
    monthly_df = df.groupby(['State', 'YearMonth']).agg({
        'EV_Sales_Quantity': 'sum',
        'Date': 'first' # Just to keep a date column
    }).reset_index()
    
    # Convert YearMonth back to timestamp (1st of the month) for processing
    monthly_df['Date'] = monthly_df['YearMonth'].dt.to_timestamp()
    
    # Sort
    monthly_df = monthly_df.sort_values(['State', 'Date'])
    
    print(f"📊 Original Rows: {len(df)} -> Monthly Rows: {len(monthly_df)}")
    print(f"   Max Monthly Sales: {monthly_df['EV_Sales_Quantity'].max()}")
    
    return monthly_df

def create_monthly_features(df):
    print("🔧 Creating Monthly Features...")
    df = df.copy()
    
    # 1. Time Features
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['quarter'] = df['Date'].dt.quarter
    
    # 2. Lags (Now these represent Previous Months!)
    # Lag 1 = Last Month, Lag 12 = Last Year
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_month_{lag}'] = df.groupby('State')['EV_Sales_Quantity'].shift(lag)
    
    # 3. Rolling Trends (Quarterly, Half-Yearly)
    for w in [3, 6, 12]:
        g = df.groupby('State')['EV_Sales_Quantity']
        df[f'roll_mean_{w}m'] = g.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'roll_std_{w}m'] = g.rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True)
        df[f'roll_max_{w}m'] = g.rolling(window=w, min_periods=1).max().reset_index(level=0, drop=True)

    # 4. Momentum (Last 3 months vs Last 12 months)
    df['momentum'] = df['roll_mean_3m'] / (df['roll_mean_12m'] + 1)
    
    # 5. Expanding Mean (Long term state average)
    df['state_avg'] = df.groupby('State')['EV_Sales_Quantity'].expanding().mean().reset_index(level=0, drop=True)

    # Clean
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(0)
    
    return df

def train_monthly_model():
    print("🚀 Monthly 3-Wheeler Trainer Started")
    
    # 1. Load & Aggregate
    df = load_and_aggregate_monthly(DATA_PATH)
    
    # 2. Features
    df_processed = create_monthly_features(df)
    
    # 3. Split
    # Last 3 months for testing
    split_date = df_processed['Date'].max() - pd.DateOffset(months=3)
    print(f"📅 Splitting at {split_date.date()} (Last 3 months Test)")
    
    train_df = df_processed[df_processed['Date'] <= split_date].copy()
    test_df = df_processed[df_processed['Date'] > split_date].copy()
    
    y_train = train_df['EV_Sales_Quantity']
    y_test = test_df['EV_Sales_Quantity']
    
    # 4. Prepare X
    exclude = ['Date', 'State', 'YearMonth', 'EV_Sales_Quantity']
    features = [c for c in train_df.columns if c not in exclude]
    
    X_train = train_df[features]
    X_test = test_df[features]
    
    # State encoding
    # For monthly data with fewer rows, Target Encoding is safer than OneHot/Categorical
    # Calculate average sales per state in Training only
    state_means = train_df.groupby('State')['EV_Sales_Quantity'].mean()
    X_train['state_encoded'] = train_df['State'].map(state_means)
    X_test['state_encoded'] = test_df['State'].map(state_means).fillna(train_df['EV_Sales_Quantity'].mean())
    
    features.append('state_encoded')
    
    # 5. Model
    print("🧠 Training LightGBM on Monthly Aggregates...")
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=20,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train[features], y_train)
    
    # 6. Evaluate
    preds = model.predict(X_test[features])
    preds = np.maximum(preds, 0)
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print("\n" + "="*40)
    print(f"🏆 Final Results (Monthly Model):")
    print(f"   MAE : {mae:.4f}")
    print(f"   R²  : {r2:.4f}")
    print("="*40)
    
    print("\n🔍 Sample Predictions:")
    res = pd.DataFrame({'State': test_df['State'], 'Date': test_df['Date'], 
                        'Actual': y_test, 'Predicted': preds})
    print(res.sample(min(10, len(res))))
    
    if r2 > 0.6:
        print(f"\n💾 Saving model to {MODEL_PATH}")
        import joblib
        MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
    train_monthly_model()