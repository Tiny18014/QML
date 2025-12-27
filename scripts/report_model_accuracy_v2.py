#!/usr/bin/env python3
"""
Comprehensive Model Accuracy Checker (Post-Retraining)
Tests all models on clean 2021-2024 data with proper feature alignment.
Supports both Daily (Advanced) and Monthly (Specialized) models.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import holidays
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# Paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"

# Model configuration
MODELS_INFO = {
    "advanced_model_2-Wheelers.pkl": {"category": "2-Wheelers", "type": "daily"},
    "advanced_model_4-Wheelers.pkl": {"category": "4-Wheelers", "type": "daily"},
    "specialized_bus_monthly_model.pkl": {"category": "Bus", "type": "monthly"},
    "specialized_3w_monthly_model.pkl": {"category": "3-Wheelers", "type": "monthly"},
    "advanced_model_Others.pkl": {"category": "Others", "type": "daily"},
}

print("🎯 Comprehensive Model Accuracy Checker")
print("=" * 70)

# ===================== DAILY ADVANCED FEATURE ENGINEERING =====================
def create_advanced_features(df):
    """
    Feature engineering for Daily models (2W, 4W, Others).
    Matches logic in scripts/advanced_model_trainer.py
    """
    df = df.copy()
    
    if 'Vehicle_Category' in df.columns:
        df['Vehicle_Category'] = df['Vehicle_Category'].fillna('Unknown')

    df['Date'] = pd.to_datetime(df['Date'])
    df['time_index'] = (df['Date'] - df['Date'].min()).dt.days
    df = df.sort_values(['State', 'Vehicle_Category', 'Date'])
    
    # Basic temporal features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
    
    # Holiday features
    years_in_data = df['year'].unique()
    in_holidays = holidays.country_holidays('IN', years=years_in_data)
    df['is_holiday'] = df['Date'].isin(in_holidays).astype(int)

    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    # Zero-Inflated Features
    df['is_sale'] = (df['EV_Sales_Quantity'] > 0).astype(int)
    
    # Create a grouping key
    g = df.groupby(['State', 'Vehicle_Category'])
    
    # Last sale index
    df['last_sale_idx'] = g['is_sale'].transform(
        lambda x: x.cumsum().shift().fillna(0)
    )
    
    # Rolling count of non-zero sales in past 30 days
    df['sales_frequency_30d'] = g['is_sale'].rolling(window=30).sum().reset_index(level=[0,1], drop=True)
    
    df = df.drop(columns=['is_sale', 'last_sale_idx'])
    
    # Advanced lag features
    for lag in [1, 2, 3, 7, 14, 30, 60, 90]:
        df[f'lag_{lag}'] = g['EV_Sales_Quantity'].shift(lag)
        
    # Rolling statistics
    for window in [7, 14, 30, 60, 90]:
        grouped_rolling = g['EV_Sales_Quantity'].rolling(window=window, min_periods=1)
        df[f'rolling_mean_{window}'] = grouped_rolling.mean().reset_index(level=[0,1], drop=True)
        df[f'rolling_std_{window}'] = grouped_rolling.std().reset_index(level=[0,1], drop=True)
        df[f'rolling_min_{window}'] = grouped_rolling.min().reset_index(level=[0,1], drop=True)
        df[f'rolling_max_{window}'] = grouped_rolling.max().reset_index(level=[0,1], drop=True)
        df[f'rolling_median_{window}'] = grouped_rolling.median().reset_index(level=[0,1], drop=True)
        
    # EMA
    for span in [7, 14, 30, 60, 90]:
        df[f'ema_{span}'] = g['EV_Sales_Quantity'].ewm(span=span).mean().reset_index(level=[0,1], drop=True)
        
    # Seasonal/Trend
    df['seasonal_7'] = g['EV_Sales_Quantity'].rolling(window=7, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    df['seasonal_30'] = g['EV_Sales_Quantity'].rolling(window=30, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    df['seasonal_60'] = g['EV_Sales_Quantity'].rolling(window=60, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    
    df['trend_7'] = g['EV_Sales_Quantity'].rolling(window=7, min_periods=1).mean().diff().reset_index(level=[0,1], drop=True)
    df['trend_30'] = g['EV_Sales_Quantity'].rolling(window=30, min_periods=1).mean().diff().reset_index(level=[0,1], drop=True)
    df['trend_60'] = g['EV_Sales_Quantity'].rolling(window=60, min_periods=1).mean().diff().reset_index(level=[0,1], drop=True)
    
    df['volatility_7'] = g['EV_Sales_Quantity'].rolling(window=7, min_periods=1).std().reset_index(level=[0,1], drop=True)
    df['volatility_30'] = g['EV_Sales_Quantity'].rolling(window=30, min_periods=1).std().reset_index(level=[0,1], drop=True)
    
    # Interaction features
    df['interaction_month_state'] = df['month'].astype(str) + '_' + df['State']
    df['interaction_dow_state'] = df['day_of_week'].astype(str) + '_' + df['State']
    
    # Convert interactions to category codes
    df['interaction_month_state'] = df['interaction_month_state'].astype('category').cat.codes
    df['interaction_dow_state'] = df['interaction_dow_state'].astype('category').cat.codes
    
    # State encoding
    df['State'] = df['State'].astype('category').cat.codes
    
    # Fill NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

# ===================== MONTHLY SPECIALIZED FEATURE ENGINEERING =====================
def create_monthly_features(df):
    """
    Feature engineering for Monthly models (3W, Bus).
    Matches logic in scripts/specialized_3w_trainer.py and specialized_bus_trainer.py
    """
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

    # State Encoding
    df['State'] = df['State'].astype('category').cat.codes

    # Clean
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(0)
    
    return df

def aggregate_to_monthly(df):
    """Aggregates daily data to monthly frequency."""
    df = df.copy()
    df['YearMonth'] = df['Date'].dt.to_period('M')
    monthly_df = df.groupby(['State', 'YearMonth']).agg({
        'EV_Sales_Quantity': 'sum',
        'Date': 'first'
    }).reset_index()
    monthly_df['Date'] = monthly_df['YearMonth'].dt.to_timestamp()
    monthly_df = monthly_df.sort_values(['State', 'Date'])
    return monthly_df

# ===================== MAIN EXECUTION =====================

def main():
    print("📊 Loading Data...")
    df_raw = pd.read_csv(DATA_PATH)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    
    # 🔥 CRITICAL: Filter out 2025 data (incomplete)
    df_raw = df_raw[df_raw['Date'] < '2025-01-01']
    print(f"   Data loaded: {len(df_raw)} records (2021-2024)")
    
    results = []
    
    for model_file, info in MODELS_INFO.items():
        category = info['category']
        model_type = info['type']
        model_path = MODELS_DIR / model_file
        
        print(f"\n🔍 Testing Model: {category} ({model_type})")
        
        if not model_path.exists():
            print(f"   ❌ Model file not found: {model_file}")
            continue
            
        # 1. Filter Data for Category
        df_cat = df_raw[df_raw['Vehicle_Category'] == category].copy()
        if df_cat.empty:
            print(f"   ⚠️ No data found for category: {category}")
            continue
            
        # 2. Prepare Data & Features
        if model_type == 'monthly':
            # Aggregate to Monthly
            df_agg = aggregate_to_monthly(df_cat)
            # Create Features
            df_features = create_monthly_features(df_agg)
            # Use last 6 months for testing (consistent with trainer)
            split_date = df_features['Date'].max() - pd.DateOffset(months=6)
            test_df = df_features[df_features['Date'] > split_date].copy()
            
        else: # daily
            # Create Features
            df_features = create_advanced_features(df_cat)
            # Use last 60 days for testing
            split_date = df_features['Date'].max() - pd.Timedelta(days=60)
            test_df = df_features[df_features['Date'] > split_date].copy()
            
        if test_df.empty:
            print("   ⚠️ Not enough data for testing.")
            continue
            
        # 3. Load Model
        try:
            loaded_obj = joblib.load(model_path)
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            continue
            
        # 4. Predict
        try:
            if isinstance(loaded_obj, dict) and 'primary_model' in loaded_obj:
                # Advanced Model (Daily)
                model = loaded_obj['primary_model']
                scaler = loaded_obj['scaler']
                feature_names = loaded_obj['feature_names']
                
                # Ensure all features exist
                for col in feature_names:
                    if col not in test_df.columns:
                        test_df[col] = 0
                
                X_test = test_df[feature_names]
                X_test_scaled = scaler.transform(X_test)
                preds = model.predict(X_test_scaled)
                
            else:
                # Specialized Model (Monthly)
                model = loaded_obj
                # Exclude non-feature columns
                exclude_cols = ['State', 'Date', 'YearMonth', 'EV_Sales_Quantity', 'Vehicle_Category']
                features = [c for c in test_df.columns if c not in exclude_cols]
                
                # Ensure numeric types
                X_test = test_df[features].select_dtypes(include=[np.number])
                preds = model.predict(X_test)
                
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            continue
            
        # 5. Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        print(f"   ✅ Results: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
        
        results.append({
            'category': category,
            'type': model_type,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'records': len(test_df)
        })
        
    # Save Results
    if results:
        res_df = pd.DataFrame(results)
        res_path = OUTPUT_DIR / "final_model_accuracy_report.csv"
        res_df.to_csv(res_path, index=False)
        print("\n📄 Final Report Saved to:", res_path)
        print(res_df)
    else:
        print("\n❌ No results generated.")

if __name__ == "__main__":
    main()
