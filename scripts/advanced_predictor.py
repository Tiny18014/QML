#!/usr/bin/env python3
"""
Advanced EV Demand Forecasting Predictor
Uses the high-performance model for accurate predictions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
MODEL_PATH = ROOT_DIR / "models" / "advanced_ev_model.pkl"
SCALER_PATH = ROOT_DIR / "models" / "feature_scaler.pkl"
FEATURE_NAMES_PATH = ROOT_DIR / "models" / "feature_names.pkl"

def load_advanced_model():
    """Load the advanced model and related components."""
    try:
        # Load model bundle
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load scaler
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature names
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        
        print(f"✅ Advanced model loaded successfully")
        print(f"   Primary model: {type(model_data['primary_model'])}")
        print(f"   Ensemble models: {len(model_data['ensemble_models'])}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Test R²: {model_data['test_scores']['optimized']['R2']:.4f}")
        
        return model_data, scaler, feature_names
        
    except Exception as e:
        print(f"❌ Failed to load advanced model: {e}")
        return None, None, None

def create_advanced_features(df):
    """Create the same advanced features used during training."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle column name differences
    if 'Vehicle_Class' in df.columns and 'Vehicle_Category' not in df.columns:
        df['Vehicle_Category'] = df['Vehicle_Class']
        print("✅ Mapped Vehicle_Class to Vehicle_Category for compatibility")
    
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
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    # Advanced lag features
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'lag_{lag}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1).mean().values
        df[f'rolling_std_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1).std().values
        df[f'rolling_min_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1).min().values
        df[f'rolling_max_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1).max().values
        df[f'rolling_median_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1).median().values
    
    # Exponential moving averages
    for span in [7, 14, 30]:
        df[f'ema_{span}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].ewm(span=span).mean().values
    
    # Seasonal decomposition features
    df['seasonal_7'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=7, min_periods=1).mean().values
    df['seasonal_30'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=30, min_periods=1).mean().values
    
    # Trend features (simplified)
    df['trend_7'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=7, min_periods=1).mean().diff().values
    df['trend_30'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=30, min_periods=1).mean().diff().values
    
    # Volatility features
    df['volatility_7'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=7, min_periods=1).std().values
    df['volatility_30'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=30, min_periods=1).std().values
    
    # Cross-category features
    category_means = df.groupby(['State', 'Date'])['EV_Sales_Quantity'].mean().reset_index()
    category_means = category_means.rename(columns={'EV_Sales_Quantity': 'state_daily_mean'})
    df = df.merge(category_means, on=['State', 'Date'], how='left')
    
    # State-level features
    state_means = df.groupby('State')['EV_Sales_Quantity'].mean().reset_index()
    state_means = state_means.rename(columns={'EV_Sales_Quantity': 'state_overall_mean'})
    df = df.merge(state_means, on='State', how='left')
    
    # Category-level features
    category_overall_means = df.groupby('Vehicle_Category')['EV_Sales_Quantity'].mean().reset_index()
    category_overall_means = category_overall_means.rename(columns={'EV_Sales_Quantity': 'category_overall_mean'})
    df = df.merge(category_overall_means, on='Vehicle_Category', how='left')
    
    # Interaction features
    df['state_category_interaction'] = df['state_overall_mean'] * df['category_overall_mean']
    df['sales_ratio_to_state_mean'] = df['EV_Sales_Quantity'] / (df['state_daily_mean'] + 1)
    df['sales_ratio_to_category_mean'] = df['EV_Sales_Quantity'] / (df['category_overall_mean'] + 1)
    
    # Fill NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Ensure we have proper feature variation
    print(f"✅ Created {len(df.columns)} features for prediction")
    print(f"   Sample feature values - State codes: {df['State'].nunique()}, Category codes: {df['Vehicle_Category'].nunique()}")
    print(f"   Temporal features: month={df['month'].nunique()}, day_of_week={df['day_of_week'].nunique()}")
    
    return df

def prepare_features_for_prediction(df, feature_names, scaler):
    """Prepare features for prediction using the same preprocessing as training."""
    # Handle column name differences
    if 'Vehicle_Class' in df.columns and 'Vehicle_Category' not in df.columns:
        df['Vehicle_Category'] = df['Vehicle_Class']
    
    # Convert categorical variables to strings first to avoid category conflicts
    df['State'] = df['State'].astype(str)
    df['Vehicle_Category'] = df['Vehicle_Category'].astype(str)
    
    # Create state and category codes for internal use
    df['state_code'] = pd.Categorical(df['State']).codes
    df['category_code'] = pd.Categorical(df['Vehicle_Category']).codes
    
    # Replace State and Vehicle_Category with their codes for the model
    df['State'] = df['state_code']
    df['Vehicle_Category'] = df['category_code']
    
    # Select only the features that the model expects
    available_features = [col for col in feature_names if col in df.columns]
    missing_features = [col for col in feature_names if col not in df.columns]
    
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
        # Fill missing features with 0
        for feature in missing_features:
            df[feature] = 0
    
    # Create feature matrix
    X = df[feature_names].copy()
    
    # Ensure all features are numerical
    for col in X.columns:
        if X[col].dtype == 'object':
            # Convert string columns to numerical using category codes
            X[col] = pd.Categorical(X[col]).codes
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    # Scale features using existing scaler
    X_scaled = scaler.transform(X)
    
    return X_scaled

def predict_with_advanced_model(df, model_data, scaler, feature_names):
    """Make predictions using the advanced model."""
    try:
        # Create advanced features
        df_advanced = create_advanced_features(df)
        
        # Prepare features for prediction
        X_scaled = prepare_features_for_prediction(df_advanced, feature_names, scaler)
        
        # Make predictions with primary model
        primary_model = model_data['primary_model']
        predictions = primary_model.predict(X_scaled)
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        
        print(f"✅ Advanced model predictions completed")
        print(f"   Input samples: {len(df)}")
        print(f"   Features used: {X_scaled.shape[1]}")
        print(f"   Predictions range: {predictions.min():.2f} to {predictions.max():.2f}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Advanced model prediction failed: {e}")
        return None

def main():
    """Test the advanced model with sample data."""
    print("🚀 Advanced EV Demand Forecasting Predictor")
    print("=" * 50)
    
    # Load the advanced model
    model_data, scaler, feature_names = load_advanced_model()
    
    if model_data is None:
        print("❌ Could not load advanced model")
        return
    
    # Create sample data for testing
    print("\n🧪 Testing with sample data...")
    
    # Sample data structure
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'State': ['Maharashtra'] * 10,
        'Vehicle_Category': ['2-Wheelers'] * 10,
        'EV_Sales_Quantity': [100, 120, 110, 130, 125, 140, 135, 150, 145, 160]
    })
    
    # Make predictions
    predictions = predict_with_advanced_model(sample_data, model_data, scaler, feature_names)
    
    if predictions is not None:
        print(f"\n📊 Sample Predictions:")
        for i, (date, actual, pred) in enumerate(zip(sample_data['Date'], sample_data['EV_Sales_Quantity'], predictions)):
            print(f"   {date.strftime('%Y-%m-%d')}: Actual={actual}, Predicted={pred:.1f}")
    
    print("\n✅ Advanced predictor ready for integration!")

if __name__ == "__main__":
    main()
