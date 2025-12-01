import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import logging
import holidays

# --- Configuration ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / 'data' / 'EV_Dataset.csv'
MODELS_PATH = BASE_PATH / 'models'
OUTPUT_PATH = BASE_PATH / 'output'

def create_advanced_features(df):
    """
    Replicates the feature engineering from advanced_model_trainer.py
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Basic temporal features - Create BOTH Capitalized and Lowercase to satisfy all models
    df['Year'] = df['Date'].dt.year
    df['year'] = df['Year']
    
    df['Month'] = df['Date'].dt.month
    df['month'] = df['Month']
    
    df['Day'] = df['Date'].dt.day
    df['day'] = df['Day']
    
    df['Quarter'] = df['Date'].dt.quarter
    df['quarter'] = df['Quarter']
    
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['day_of_week'] = df['Day_of_Week']
    
    df['Week_of_Year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['week_of_year'] = df['Week_of_Year']
    
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    df['day_of_year'] = df['Day_of_Year']
    
    df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
    df['is_weekend'] = df['Is_Weekend']
    
    df['Is_Month_Start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_start'] = df['Is_Month_Start']
    
    df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
    df['is_month_end'] = df['Is_Month_End']
    
    df['Is_Quarter_Start'] = df['Date'].dt.is_quarter_start.astype(int)
    df['is_quarter_start'] = df['Is_Quarter_Start']
    
    df['Is_Quarter_End'] = df['Date'].dt.is_quarter_end.astype(int)
    df['is_quarter_end'] = df['Is_Quarter_End']
    
    # Holiday features
    years_in_data = df['year'].unique()
    # Handle potential empty years if df is empty (unlikely here)
    if len(years_in_data) > 0:
        in_holidays = holidays.country_holidays('IN', years=years_in_data)
        df['is_holiday'] = df['Date'].isin(in_holidays).astype(int)
    else:
        df['is_holiday'] = 0

    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    # Zero-Inflated Features
    df['is_sale'] = (df['EV_Sales_Quantity'] > 0).astype(int)
    g = df.groupby(['State', 'Vehicle_Category'])
    df['sales_frequency_30d'] = g['is_sale'].rolling(window=30, min_periods=1).sum().reset_index(level=[0,1], drop=True)
    df = df.drop(columns=['is_sale'])

    # Also add time_index if it was used
    df['time_index'] = (df['Date'] - df['Date'].min()).dt.days

    # Lag features
    for lag in [1, 2, 3, 7, 14, 30, 60, 90]:
        df[f'lag_{lag}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30, 60, 90]:
        grouped_rolling = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1)
        df[f'rolling_mean_{window}'] = grouped_rolling.mean().reset_index(level=[0, 1], drop=True)
        df[f'rolling_std_{window}'] = grouped_rolling.std().reset_index(level=[0, 1], drop=True)
        df[f'rolling_min_{window}'] = grouped_rolling.min().reset_index(level=[0, 1], drop=True)
        df[f'rolling_max_{window}'] = grouped_rolling.max().reset_index(level=[0, 1], drop=True)
        df[f'rolling_median_{window}'] = grouped_rolling.median().reset_index(level=[0, 1], drop=True)
    
    # Exponential moving averages
    for span in [7, 14, 30, 60, 90]:
        df[f'ema_{span}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].ewm(span=span).mean().reset_index(level=[0, 1], drop=True)
    
    # Seasonal decomposition features
    df['seasonal_7'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=7, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df['seasonal_30'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=30, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df['seasonal_60'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=60, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    
    # Trend features
    df['trend_7'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=7, min_periods=1).mean().diff().reset_index(level=[0, 1], drop=True)
    df['trend_30'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=30, min_periods=1).mean().diff().reset_index(level=[0, 1], drop=True)
    df['trend_60'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=60, min_periods=1).mean().diff().reset_index(level=[0, 1], drop=True)
    
    # Volatility features
    df['volatility_7'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=7, min_periods=1).std().reset_index(level=[0, 1], drop=True)
    df['volatility_30'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=30, min_periods=1).std().reset_index(level=[0, 1], drop=True)
    df['volatility_60'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=60, min_periods=1).std().reset_index(level=[0, 1], drop=True)
    
    # Cross-category features
    category_means = df.groupby(['State', 'Date'])['EV_Sales_Quantity'].mean().reset_index()
    category_means = category_means.rename(columns={'EV_Sales_Quantity': 'state_daily_mean'})
    df = df.merge(category_means, on=['State', 'Date'], how='left')
    
    # State-level features
    if 'state_overall_mean' not in df.columns:
        state_means = df.groupby('State')['EV_Sales_Quantity'].mean().reset_index()
        state_means = state_means.rename(columns={'EV_Sales_Quantity': 'state_overall_mean'})
        df = df.merge(state_means, on='State', how='left')
    
    # Category-level features
    if 'category_overall_mean' not in df.columns:
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
    
    return df

def generate_forecast(category, state, days=90):
    logging.info(f"Starting forecast for {category} in {state} for {days} days...")
    
    # 1. Load Model
    model_name = f"advanced_model_{category.replace(' ', '_')}.pkl"
    model_path = MODELS_PATH / model_name
    
    if not model_path.exists():
        logging.error(f"Model file not found: {model_path}")
        return
        
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    model = model_data['primary_model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    saved_state_encoding = model_data.get('state_encoding', None)
    
    # 2. Load Data
    df_full = pd.read_csv(DATA_PATH)
    df_full['Date'] = pd.to_datetime(df_full['Date'])
    
    # Filter for the specific category to match training context
    df_cat = df_full[df_full['Vehicle_Category'] == category].copy()
    
    # --- PRE-CALCULATE GLOBAL FEATURES ---
    # Calculate these BEFORE filtering to a single state
    
    # 1. Category Overall Mean (Global for this category)
    global_category_mean = df_cat['EV_Sales_Quantity'].mean()
    
    # 2. State Overall Mean (Specific to the target state)
    state_mean_val = df_cat[df_cat['State'] == state]['EV_Sales_Quantity'].mean()
    
    # 3. State Daily Mean (Seasonality per state)
    df_cat['day_of_week'] = df_cat['Date'].dt.dayofweek
    state_daily_means = df_cat[df_cat['State'] == state].groupby('day_of_week')['EV_Sales_Quantity'].mean()
    
    # --- FILTER TO TARGET STATE ---
    # This is the key optimization. We only process the target state.
    df_state = df_cat[df_cat['State'] == state].copy()
    
    if df_state.empty:
        logging.error(f"No data found for state {state} and category {category}")
        return

    # 3. Prepare for Recursive Forecasting
    last_date = df_state['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    # --- State Encoding Logic ---
    if saved_state_encoding:
        state_to_code = {s: i for i, s in enumerate(saved_state_encoding)}
        logging.info(f"Loaded State Encodings from Model: {len(state_to_code)} states")
    else:
        logging.warning("No state encoding found in model! Falling back to dynamic encoding (risky).")
        all_states = sorted(df_cat['State'].unique()) # Use df_cat to get ALL states
        state_to_code = {s: i for i, s in enumerate(all_states)}
        
    cat_to_code = {category: 0} 

    # Create future dataframe structure
    future_df = pd.DataFrame({
        'Date': future_dates,
        'State': state,
        'Vehicle_Category': category,
        'EV_Sales_Quantity': np.nan # To be filled
    })
    
    # Append future rows to history
    df_combined = pd.concat([df_state, future_df], ignore_index=True)
    df_combined = df_combined.sort_values(['Date']) # Only one state, so sort by Date
    
    # 4. Recursive Loop
    for date in future_dates:
        # a. Apply feature engineering (Now fast, on ~1.5k rows)
        df_features = create_advanced_features(df_combined)
        
        # b. Inject Pre-calculated Global Features
        # Overwrite category_overall_mean with the GLOBAL one
        df_features['category_overall_mean'] = global_category_mean
        
        # Ensure state_overall_mean is present
        if 'state_overall_mean' not in df_features.columns:
             df_features['state_overall_mean'] = state_mean_val
             
        # Ensure state_daily_mean is present
        if 'state_daily_mean' not in df_features.columns:
             df_features['day_of_week'] = df_features['Date'].dt.dayofweek
             df_features['state_daily_mean'] = df_features['day_of_week'].map(state_daily_means)
        
        # Re-calculate interactions
        df_features['state_category_interaction'] = df_features['state_overall_mean'] * df_features['category_overall_mean']
        df_features['sales_ratio_to_state_mean'] = df_features['EV_Sales_Quantity'] / (df_features['state_daily_mean'] + 1)
        df_features['sales_ratio_to_category_mean'] = df_features['EV_Sales_Quantity'] / (df_features['category_overall_mean'] + 1)
        
        # Fill NaNs again just in case
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        df_features[numeric_columns] = df_features[numeric_columns].fillna(0)

        # c. Encode Categoricals
        df_encoded = df_features.copy()
        # Apply the fixed mappings
        df_encoded['State'] = df_encoded['State'].map(state_to_code).fillna(0).astype(int)
        df_encoded['Vehicle_Category'] = df_encoded['Vehicle_Category'].map(cat_to_code).fillna(0).astype(int)
        
        # d. Extract the specific row to predict
        target_mask = (df_features['Date'] == date)
        
        if not target_mask.any():
            logging.error(f"Could not find row for {date}")
            break
            
        # Get the row from df_encoded using the mask
        row = df_encoded[target_mask]
        
        # Select features
        # Ensure all features exist
        missing_cols = [c for c in feature_names if c not in row.columns]
        if missing_cols:
             for c in missing_cols:
                 row[c] = 0
        
        X_row = row[feature_names]
        
        # Scale
        X_scaled = scaler.transform(X_row)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        
        # Handle Log Transform (if applicable)
        if category in ['Bus', '3-Wheelers']:
            pred = np.expm1(pred)
            
        pred = max(0, pred) # Clip negative
        
        # Update df_combined with the prediction
        update_mask = (df_combined['Date'] == date)
        df_combined.loc[update_mask, 'EV_Sales_Quantity'] = pred
        
        logging.info(f"Predicted for {date.date()}: {pred:.2f}")
        
        if date == future_dates[0]:
             logging.info(f"DEBUG: State: {state}, Category: {category}")
             logging.info(f"DEBUG: Global Category Mean: {global_category_mean}")
             logging.info(f"DEBUG: State Mean: {state_mean_val}")
             logging.info(f"DEBUG: Row State Mean Feature: {row['state_overall_mean'].values[0]}")
             logging.info(f"DEBUG: Row Category Mean Feature: {row['category_overall_mean'].values[0]}")
             logging.info(f"DEBUG: Raw Features (first 5): {X_row.iloc[0, :5].values}")
             logging.info(f"DEBUG: Scaled Features (first 5): {X_scaled[0, :5]}")

    # 5. Save Results
    forecast_result = df_combined[df_combined['Date'] > last_date]
    output_file = OUTPUT_PATH / f"forecast_{state}_{category}.csv"
    forecast_result[['Date', 'EV_Sales_Quantity']].to_csv(output_file, index=False)
    logging.info(f"Forecast saved to {output_file}")
    
    # Print summary
    print(forecast_result[['Date', 'EV_Sales_Quantity']].head())
    print(f"Total predicted sales for next {days} days: {forecast_result['EV_Sales_Quantity'].sum():.2f}")

if __name__ == "__main__":
    # Test for Karnataka 4-Wheelers
    generate_forecast("4-Wheelers", "Karnataka", 90)
    
    # Test for Delhi Bus
    generate_forecast("Bus", "Delhi", 90)
