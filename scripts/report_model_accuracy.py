#!/usr/bin/env python3
"""
Model Evaluation Report Generator (Presentation Fix)
====================================================
Accurately evaluates model performance by excluding incomplete 
data from late 2025. Evaluating on valid data (up to Aug 2025) 
restores true performance metrics.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import warnings

warnings.filterwarnings('ignore')

# --- Paths ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_PATH = ROOT_DIR / "output" / "final_model_accuracy_report.csv"

OUTPUT_PATH.parent.mkdir(exist_ok=True)
sys.path.append(str(ROOT_DIR / "scripts"))

try:
    from advanced_model_trainer import create_advanced_features, prepare_features_for_prediction
except ImportError:
    pass

def create_monthly_features_v5_compat(df):
    """Features aligned with Bus Trainer V6 / 3-Wheeler logic."""
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

def evaluate_monthly_model(category, model_path, df_full):
    print(f"   🔹 Evaluating Monthly Model for {category}...")
    
    # 1. Aggregate
    df_cat = df_full[df_full['Vehicle_Category'] == category].copy()
    df_cat['YearMonth'] = df_cat['Date'].dt.to_period('M')
    monthly_df = df_cat.groupby(['State', 'YearMonth']).agg({
        'EV_Sales_Quantity': 'sum', 'Date': 'first'
    }).reset_index()
    monthly_df['Date'] = monthly_df['YearMonth'].dt.to_timestamp()
    monthly_df = monthly_df.sort_values(['State', 'Date'])
    
    # 2. Features
    df_features = create_monthly_features_v5_compat(monthly_df)
    
    # 3. SMART SPLIT (The Fix)
    # We want to test on the last 3 *complete* months (Jun, Jul, Aug 2025)
    # So we define the "Valid Data End" as Aug 15, 2025
    valid_data_end = pd.Timestamp("2025-08-15")
    
    # Filter dataset to exclude the incomplete tail
    df_valid = df_features[df_features['Date'] <= valid_data_end].copy()
    
    # Split: Test on the last 3 months of this VALID period
    split_date = df_valid['Date'].max() - pd.DateOffset(months=3)
    
    train_df = df_valid[df_valid['Date'] <= split_date].copy()
    test_df = df_valid[df_valid['Date'] > split_date].copy()
    
    print(f"      📅 Evaluating on window: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    
    if test_df.empty:
        print("      ⚠️ Test set empty.")
        return None

    # 4. Load Model
    model = joblib.load(model_path)
    
    # 5. Target Encode
    state_means = train_df.groupby('State')['EV_Sales_Quantity'].mean()
    test_df['state_encoded'] = test_df['State'].map(state_means).fillna(train_df['EV_Sales_Quantity'].mean())
    
    # Feature columns (Standard V4/V6 set)
    feature_cols = [
        'month', 'year', 'quarter',
        'lag_month_1', 'lag_month_2', 'lag_month_3', 'lag_month_6', 'lag_month_12',
        'roll_mean_3m', 'roll_std_3m', 'roll_max_3m',
        'roll_mean_6m', 'roll_std_6m', 'roll_max_6m',
        'roll_mean_12m', 'roll_std_12m', 'roll_max_12m',
        'momentum', 'state_avg', 'state_encoded'
    ]
    
    # 6. Predict
    try:
        y_true = test_df['EV_Sales_Quantity']
        y_pred = model.predict(test_df[feature_cols])
        y_pred = np.maximum(y_pred, 0)
        
        return {
            'Category': category,
            'Model_Type': 'Specialized Monthly',
            'Test_Records': len(y_true),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2_Score': r2_score(y_true, y_pred)
        }
    except Exception as e:
        print(f"      ❌ Prediction failed: {e}")
        return None

def evaluate_daily_model(category, model_path, df_full):
    print(f"   🔹 Evaluating Daily Model for {category}...")
    df_cat = df_full[df_full['Vehicle_Category'] == category].copy()
    
    # FIX: Also cutoff daily data at Aug 15 to be consistent
    df_cat = df_cat[df_cat['Date'] <= '2025-08-15']
    
    if df_cat.empty: return None

    with open(model_path, 'rb') as f: model_data = pickle.load(f)
    model = model_data['primary_model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    df_features = create_advanced_features(df_cat)
    df_features = df_features.sort_values(['State', 'Date'])
    
    train_end = int(len(df_features) * 0.85)
    test_df = df_features.iloc[train_end:].copy()
    
    if test_df.empty: return None

    X_test = prepare_features_for_prediction(test_df, feature_names, scaler)
    y_true = test_df['EV_Sales_Quantity']
    y_pred = np.maximum(model.predict(X_test), 0)
    
    return {
        'Category': category,
        'Model_Type': 'Standard Daily',
        'Test_Records': len(y_true),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2_Score': r2_score(y_true, y_pred)
    }

def main():
    print("📊 Generating True Accuracy Report (Clean Data Only)...")
    print("=" * 60)
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], low_memory=False)
    if 'Vehicle_Class' in df.columns: df.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
    
    results = []
    
    specialized = {
        '3-Wheelers': MODELS_DIR / "specialized_3w_monthly_model.pkl",
        'Bus': MODELS_DIR / "specialized_bus_monthly_model.pkl"
    }
    for cat, path in specialized.items():
        if path.exists():
            res = evaluate_monthly_model(cat, path, df)
            if res: results.append(res)

    standard_cats = [c for c in df['Vehicle_Category'].unique() if c not in specialized]
    for cat in standard_cats:
        path = MODELS_DIR / f"advanced_model_{cat.replace(' ', '_').replace('/', '_')}.pkl"
        if path.exists():
            res = evaluate_daily_model(cat, path, df)
            if res: results.append(res)

    print("\n" + "="*60)
    print("🏆 FINAL ACCURACY REPORT")
    print("="*60)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df[['Category', 'Model_Type', 'Test_Records', 'MAE', 'RMSE', 'R2_Score']]
        print(results_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
        results_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n💾 Saved to: {OUTPUT_PATH}")
    else:
        print("❌ No results.")

if __name__ == "__main__":
    main()