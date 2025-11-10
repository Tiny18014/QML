import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import holidays # <-- Necessary import for is_holiday feature
from sklearn.preprocessing import RobustScaler 

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
ROOT_DIR = Path(".") # Current directory for execution
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
SPLIT_DATE = datetime(2025, 1, 1) # Date where evaluation data (Test Set) begins

# --- MODEL LOADING ---

def load_model_components(category_name):
    """Load the dedicated model bundle for a specific category."""
    category_filename = category_name.replace(" ", "_").replace("/", "_")
    model_path = MODELS_DIR / f"advanced_model_{category_filename}.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data.get('primary_model')
        scaler = model_data.get('scaler')
        feature_names = model_data.get('feature_names')
        
        if not all([model, scaler, feature_names]):
             raise ValueError("Model file is missing one or more required components.")
        
        print(f"✅ Loaded model for '{category_name}'. Features: {len(feature_names)}")
        return model, scaler, feature_names
        
    except FileNotFoundError:
        print(f"❌ Model file not found at {model_path}. Skipping evaluation for this category.")
        return None, None, None
    except Exception as e:
        print(f"❌ Failed to load model for '{category_name}': {e}. Skipping.")
        return None, None, None

# --- FEATURE ENGINEERING (INCLUDES FIXES) ---

def create_advanced_features(df):
    """Create the same advanced features used during training."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle column name differences
    if 'Vehicle_Class' in df.columns and 'Vehicle_Category' not in df.columns:
        df['Vehicle_Category'] = df['Vehicle_Class']
    
    # Sort data for correct lag/rolling calculation across all data points
    df = df.sort_values(['State', 'Vehicle_Category', 'Date'])
    
    # === CRITICAL FIXES APPLIED HERE (time_index and is_holiday) ===
    df['time_index'] = (df['Date'] - df['Date'].min()).dt.days
    df['year'] = df['Date'].dt.year # Needed for is_holiday and other features
    years_in_data = df['year'].unique()
    in_holidays = holidays.country_holidays('IN', years=years_in_data)
    df['is_holiday'] = df['Date'].isin(in_holidays).astype(int)
    # =============================================================

    # Basic temporal features
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
    # 🟢 ADDED 60 and 90 day lags
    for lag in [1, 2, 3, 7, 14, 30, 60, 90]:
        df[f'lag_{lag}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].shift(lag)
    
    # Rolling statistics
    # 🟢 ADDED 60 and 90 day windows
    for window in [7, 14, 30, 60, 90]:
        grouped_rolling = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1)
        df[f'rolling_mean_{window}'] = grouped_rolling.mean().reset_index(level=[0, 1], drop=True)
        df[f'rolling_std_{window}'] = grouped_rolling.std().reset_index(level=[0, 1], drop=True)
        df[f'rolling_min_{window}'] = grouped_rolling.min().reset_index(level=[0, 1], drop=True)
        df[f'rolling_max_{window}'] = grouped_rolling.max().reset_index(level=[0, 1], drop=True)
        df[f'rolling_median_{window}'] = grouped_rolling.median().reset_index(level=[0, 1], drop=True)

    # Exponential moving averages
    # 🟢 ADDED 60 and 90 day spans
    for span in [7, 14, 30, 60, 90]:
        df[f'ema_{span}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].ewm(span=span).mean().reset_index(level=[0, 1], drop=True)
    # Seasonal decomposition features
    # 🟢 ADDED 60 day seasonal mean
    df['seasonal_7'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=7, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df['seasonal_30'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=30, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df['seasonal_60'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=60, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    
    # Trend features (simplified)
    # 🟢 ADDED 60 day trend diff
    df['trend_7'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=7, min_periods=1).mean().diff().reset_index(level=[0, 1], drop=True)
    df['trend_30'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=30, min_periods=1).mean().diff().reset_index(level=[0, 1], drop=True)
    df['trend_60'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=60, min_periods=1).mean().diff().reset_index(level=[0, 1], drop=True)
    
    # Volatility features
    # 🟢 ADDED 60 day volatility
    df['volatility_7'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=7, min_periods=1).std().reset_index(level=[0, 1], drop=True)
    df['volatility_30'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=30, min_periods=1).std().reset_index(level=[0, 1], drop=True)
    df['volatility_60'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=60, min_periods=1).std().reset_index(level=[0, 1], drop=True)
    # Cross-category features
    category_means = df.groupby(['State', 'Date'])['EV_Sales_Quantity'].mean().reset_index().rename(columns={'EV_Sales_Quantity': 'state_daily_mean'})
    df = df.merge(category_means, on=['State', 'Date'], how='left')
    
    state_means = df.groupby('State')['EV_Sales_Quantity'].mean().reset_index().rename(columns={'EV_Sales_Quantity': 'state_overall_mean'})
    df = df.merge(state_means, on='State', how='left')
    
    category_overall_means = df.groupby('Vehicle_Category')['EV_Sales_Quantity'].mean().reset_index().rename(columns={'EV_Sales_Quantity': 'category_overall_mean'})
    df = df.merge(category_overall_means, on='Vehicle_Category', how='left')
    
    # Interaction features
    df['state_category_interaction'] = df['state_overall_mean'] * df['category_overall_mean']
    df['sales_ratio_to_state_mean'] = df['EV_Sales_Quantity'] / (df['state_daily_mean'] + 1)
    df['sales_ratio_to_category_mean'] = df['EV_Sales_Quantity'] / (df['category_overall_mean'] + 1)
    
    # Fill NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    return df

# --- FEATURE PREPARATION (INCLUDES FIX FOR STRING COLUMNS) ---

def prepare_features_for_prediction(df, feature_names, scaler):
    """Prepare features for prediction using the same preprocessing as training."""
    
    # Ensure categorical columns are correctly handled for encoding consistency
    if 'Vehicle_Class' in df.columns and 'Vehicle_Category' not in df.columns:
        df['Vehicle_Category'] = df['Vehicle_Class']

    df_encoded = df.copy()
    
    # 1. Convert core categorical features to category codes
    df_encoded['State'] = pd.Categorical(df_encoded['State']).codes
    df_encoded['Vehicle_Category'] = pd.Categorical(df_encoded['Vehicle_Category']).codes
    
    # 2. CRITICAL FIX: Loop through expected features and encode any remaining 'object' (string) types.
    # This specifically catches columns like 'Month_Name' or 'Year' (if string) which caused the error.
    for col in feature_names:
        if col in df_encoded.columns and df_encoded[col].dtype == 'object':
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
            
    # Select only the features that the model expects
    X = df_encoded[feature_names].copy()
    
    # Fill any remaining NaN values before scaling
    X = X.fillna(0)
    
    # Scale features using existing scaler (TRANSFORM ONLY)
    X_scaled = scaler.transform(X)
    
    return X_scaled

# --- MAIN EVALUATION FUNCTION ---

def evaluate_models():
    """Loops through all categories, loads the dedicated model, and evaluates its performance on 2025 data."""
    
    print("🚀 Starting Dedicated Model Evaluation on EV_Dataset.csv")
    print("=" * 60)
    
    df_full = pd.read_csv(DATA_PATH)
    df_full['Date'] = pd.to_datetime(df_full['Date'])
    
    # Clean/fill Vehicle_Category if necessary
    if 'Vehicle_Class' in df_full.columns and 'Vehicle_Category' not in df_full.columns:
        df_full.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
    df_full['Vehicle_Category'] = df_full['Vehicle_Category'].fillna('Unknown')
    
    unique_categories = df_full['Vehicle_Category'].unique()
    
    # Split data: everything up to SPLIT_DATE for history, everything after for test
    df_history = df_full[df_full['Date'] < SPLIT_DATE].copy()
    df_test_data = df_full[df_full['Date'] >= SPLIT_DATE].copy()

    if df_test_data.empty:
        print(f"⚠️ Test data is empty. No records found on or after {SPLIT_DATE}. Evaluation stopped.")
        return

    evaluation_results = {}
    
    for category in unique_categories:
        print("\n" + "-" * 50)
        print(f"🧪 Evaluating Model for Category: {category}")

        # 1. Load Model Components
        model, scaler, feature_names = load_model_components(category)
        if model is None:
            continue

        # 2. Prepare Data for Evaluation
        df_category_history = df_history[df_history['Vehicle_Category'] == category]
        df_category_test = df_test_data[df_test_data['Vehicle_Category'] == category]
        
        # Combine the end of the history data with the test data (last 30 days for lag features)
        df_combined = pd.concat([df_category_history.tail(30), df_category_test], ignore_index=True)
        
        # 3. Create Features
        df_advanced = create_advanced_features(df_combined)
        
        # 4. Filter to Test Set and Separate Target
        df_eval = df_advanced[df_advanced['Date'] >= SPLIT_DATE].copy()
        
        if 'EV_Sales_Quantity' not in df_eval.columns or df_eval['EV_Sales_Quantity'].isnull().all():
            print(f"⚠️ Skipping '{category}': Test set is missing target variable 'EV_Sales_Quantity'.")
            continue
            
        y_test = df_eval['EV_Sales_Quantity']

        # 5. Prepare Features and Predict
        try:
            X_test_scaled = prepare_features_for_prediction(df_eval, feature_names, scaler)
            y_pred = model.predict(X_test_scaled)
            y_pred = np.maximum(y_pred, 0)
        except Exception as e:
            print(f"❌ Prediction failed for '{category}': {e}")
            continue

        # 6. Calculate Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        evaluation_results[category] = {'MAE': mae, 'R2': r2}
        
        print(f"  -> Evaluation Complete:")
        print(f"  -> Samples: {len(y_test)}")
        print(f"  -> MAE: {mae:.4f}")
        print(f"  -> R²: {r2:.4f}")

    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION SUMMARY")
    print("=" * 60)
    
    summary_df = pd.DataFrame.from_dict(evaluation_results, orient='index')
    if not summary_df.empty:
        print(summary_df.sort_values('MAE').to_markdown(floatfmt=".4f"))
        summary_df.to_csv('model_evaluation_summary_final.csv')
        print("\n💾 Detailed summary saved to model_evaluation_summary_final.csv")
    else:
        print("No successful model evaluations were completed.")

if __name__ == "__main__":
    evaluate_models()