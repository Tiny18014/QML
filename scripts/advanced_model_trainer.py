#!/usr/bin/env python3
"""
Advanced EV Demand Forecasting Model Trainer
Creates a high-performance model using state-of-the-art techniques
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from pathlib import Path
import warnings
import os
import optuna
from datetime import datetime, timedelta
import joblib
import holidays
warnings.filterwarnings('ignore')

# Define paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "advanced_ev_model.pkl"
SCALER_PATH = ROOT_DIR / "models" / "feature_scaler.pkl"
FEATURE_NAMES_PATH = ROOT_DIR / "models" / "feature_names.pkl"

print("🚀 Advanced EV Demand Forecasting Model Trainer")
print("=" * 60)

def create_advanced_features(df):
    """Create comprehensive feature set for high-performance prediction."""
    print("🔧 Creating advanced features...")
    
    df = df.copy()
    
    # *** FIX 1: Handle missing categorical data BEFORE grouping ***
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
    
    # Advanced lag features
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'lag_{lag}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].shift(lag)
    
    # *** FIX 2: More robust assignment for rolling calculations ***
    # Rolling statistics
    for window in [7, 14, 30]:
        grouped_rolling = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1)
        df[f'rolling_mean_{window}'] = grouped_rolling.mean().reset_index(level=[0, 1], drop=True)
        df[f'rolling_std_{window}'] = grouped_rolling.std().reset_index(level=[0, 1], drop=True)
        df[f'rolling_min_{window}'] = grouped_rolling.min().reset_index(level=[0, 1], drop=True)
        df[f'rolling_max_{window}'] = grouped_rolling.max().reset_index(level=[0, 1], drop=True)
        df[f'rolling_median_{window}'] = grouped_rolling.median().reset_index(level=[0, 1], drop=True)
    
    # Exponential moving averages
    for span in [7, 14, 30]:
        df[f'ema_{span}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].ewm(span=span).mean().reset_index(level=[0, 1], drop=True)
    
    # (The rest of the function remains the same...)
    # Seasonal decomposition features
    df['seasonal_7'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=7, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df['seasonal_30'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=30, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    
    # Trend features (simplified)
    df['trend_7'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=7, min_periods=1).mean().diff().reset_index(level=[0, 1], drop=True)
    df['trend_30'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=30, min_periods=1).mean().diff().reset_index(level=[0, 1], drop=True)
    
    # Volatility features
    df['volatility_7'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=7, min_periods=1).std().reset_index(level=[0, 1], drop=True)
    df['volatility_30'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=30, min_periods=1).std().reset_index(level=[0, 1], drop=True)

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
    
    print(f"✅ Created {len(df.columns)} features")
    return df

def prepare_data_for_training(df, feature_subset=None):
    """Prepare data with proper encoding and scaling."""
    print("📊 Preparing data for training...")
    if df.empty:
        print("⚠️ Received empty training DataFrame; returning minimal placeholder.")
        return np.zeros((1, 1)), pd.Series([0]), [], RobustScaler()
    
    df['State'] = df['State'].astype('category')
    df['Vehicle_Category'] = df['Vehicle_Category'].astype('category')
    
    if feature_subset:
        feature_columns = list(set(feature_subset + ['State', 'Vehicle_Category']))
        feature_columns = [f for f in feature_columns if f in df.columns]
    else:
        feature_columns = [col for col in df.columns if col not in ['Date', 'EV_Sales_Quantity']]
    
    if 'Vehicle_Class' in feature_columns:
        feature_columns.remove('Vehicle_Class')
    
    df_encoded = df.copy()
    df_encoded['State'] = df_encoded['State'].cat.codes
    df_encoded['Vehicle_Category'] = df_encoded['Vehicle_Category'].cat.codes
    for col in feature_columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
    
    X = df_encoded[feature_columns]
    y = df_encoded['EV_Sales_Quantity']
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Prepared {X_scaled.shape[1]} features for {X_scaled.shape[0]} samples")
    return X_scaled, y, feature_columns, scaler

def create_ensemble_model():
    """Create an ensemble of multiple models for better performance."""
    models = {
        'lightgbm': lgb.LGBMRegressor(
            objective='regression',
            metric='mae',
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'lightgbm_optimized': lgb.LGBMRegressor(
            objective='regression',
            metric='mae',
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=10,
            min_child_samples=15,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'random_forest': RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    }
    
    return models

def train_models(X_train, y_train, X_val, y_val, feature_names):
    """Train multiple models and return the best ensemble."""
    print("🎯 Training ensemble models...")
    
    models = create_ensemble_model()
    trained_models = {}
    scores = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train the model
        if name.startswith('lightgbm'):
            if X_val is not None and len(X_val) > 0:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
            else:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val if X_val is not None and len(X_val) > 0 else X_train)
        
        # Calculate metrics
        tgt_true = y_val if X_val is not None and len(X_val) > 0 else y_train
        mae = mean_absolute_error(tgt_true, y_pred)
        mse = mean_squared_error(tgt_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(tgt_true, y_pred)
        
        scores[name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
        trained_models[name] = model
        
        print(f"  {name}: MAE={mae:.2f}, R²={r2:.4f}")
    
    # Find best models
    best_models = {}
    for metric in ['MAE', 'R2']:
        if metric == 'MAE':
            best_model = min(scores.items(), key=lambda x: x[1][metric])
        else:
            best_model = max(scores.items(), key=lambda x: x[1][metric])
        
        best_models[metric] = {
            'name': best_model[0],
            'model': trained_models[best_model[0]],
            'score': best_model[1]
        }
    
    print("\n🏆 Best Models:")
    for metric, info in best_models.items():
        print(f"  {metric}: {info['name']} (Score: {info['score'][metric]:.4f})")
    
    return trained_models, scores, best_models

def create_optimized_model(X_train, y_train, X_val, y_val):
    """Create an optimized model using Optuna."""
    print("🔬 Creating optimized model with Optuna...")
    
    def objective(trial):
        params = {
            'objective': 'regression', 'metric': 'mae', 'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 20.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 20.0, log=True),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        y_pred = model.predict(X_val)
        return mean_absolute_error(y_val, y_pred)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_params}")
    
    final_model = lgb.LGBMRegressor(objective='regression', metric='mae', n_estimators=2000, **study.best_params)
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    return final_model, study.best_params

def load_model_metrics(path):
    """Loads metrics from an existing model file."""
    try:
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data.get('test_scores', {}).get('optimized', {}).get('MAE', float('inf'))
    except FileNotFoundError:
        # This is expected if a model for a category doesn't exist yet
        return float('inf')
    except Exception as e:
        print(f"⚠️ Could not load metrics for {path.name}: {e}")
        return float('inf')

# In scripts/advanced_model_trainer.py, replace the entire main() function

# In scripts/advanced_model_trainer.py, replace your entire main() function with this

def main():
    """Loops through each vehicle category and trains a separate, validated model for each."""
    ROOT_DIR = Path(__file__).parent.parent.resolve()
    DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
    MODELS_DIR = ROOT_DIR / "models"
    
    print("📈 Loading data...")
    df_full = pd.read_csv(DATA_PATH)

    if 'Vehicle_Category' not in df_full.columns and 'Vehicle_Class' in df_full.columns:
        df_full.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
    
    df_full['Vehicle_Category'] = df_full['Vehicle_Category'].fillna('Unknown')
    categories = df_full['Vehicle_Category'].unique()
    print(f"\nFound {len(categories)} categories to train: {categories}")

    top_features = [
        'state_daily_mean', 'time_index', 'rolling_max_7', 'rolling_max_14',
        'rolling_mean_30', 'day_of_year', 'lag_1', 'day', 'month_sin', 'rolling_mean_7'
    ]

    for category in categories:
        print("\n" + "="*60)
        print(f"🚗 Training model for category: {category}")
        
        df_category = df_full[df_full['Vehicle_Category'] == category].copy()
        if len(df_category) < 100:
            print(f"⚠️ Skipping '{category}' due to insufficient data ({len(df_category)} records).")
            continue

        df_advanced = create_advanced_features(df_category)
        
        df_advanced = df_advanced.sort_values('Date')
        train_end = int(len(df_advanced) * 0.7)
        val_end = int(len(df_advanced) * 0.85)
        train_df, val_df, test_df = df_advanced.iloc[:train_end], df_advanced.iloc[train_end:val_end], df_advanced.iloc[val_end:]

        X_train, y_train, feature_names, scaler = prepare_data_for_training(train_df, feature_subset=top_features)
        X_val, y_val, _, _ = prepare_data_for_training(val_df, feature_subset=top_features)
        X_test, y_test, _, _ = prepare_data_for_training(test_df, feature_subset=top_features)

        optimized_model, best_params = create_optimized_model(X_train, y_train, X_val, y_val)

        print("\n🧪 Final Evaluation on Test Set:")
        y_pred_optimized = optimized_model.predict(X_test)
        mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
        r2_optimized = r2_score(y_test, y_pred_optimized)
        print(f"  --> Results for '{category}': MAE={mae_optimized:.4f}, R²={r2_optimized:.4f}")
        
        category_filename = category.replace(" ", "_").replace("/", "_")
        model_path = MODELS_DIR / f"advanced_model_{category_filename}.pkl"
        
        old_mae = load_model_metrics(model_path)
        print(f"  Comparison: New Model MAE ({mae_optimized:.4f}) vs. Old Model MAE ({old_mae:.4f})")
        
        if mae_optimized < old_mae:
            print(f"  🎉 New model for '{category}' is better! Saving...")
            model_data = {
                'primary_model': optimized_model, 'scaler': scaler,
                'feature_names': feature_names, 'best_params': best_params,
                'test_scores': {'optimized': {'MAE': mae_optimized, 'R2': r2_optimized}}
            }
            model_path.parent.mkdir(exist_ok=True)
            with open(model_path, 'wb') as f: pickle.dump(model_data, f)
        else:
            print(f"  ⚠️ New model for '{category}' did not perform better. Keeping previous version.")

    print("\n" + "="*60)
    print("🎉 All models trained and validated successfully! 🎉")
    
def prepare_features_for_prediction(df, feature_names, scaler):
    """
    Prepares a dataframe for prediction using a pre-fitted scaler.
    It only TRANSFORMS the data, it does not re-fit the scaler.
    """
    # Select the same features the model was trained on
    feature_columns = [f for f in feature_names if f in df.columns]
    
    # Ensure categorical columns are present and set the type
    df['State'] = df['State'].astype('category')
    df['Vehicle_Category'] = df['Vehicle_Category'].astype('category')
    
    # Encode categorical variables
    df_encoded = df.copy()
    df_encoded['State'] = df_encoded['State'].cat.codes
    df_encoded['Vehicle_Category'] = df_encoded['Vehicle_Category'].cat.codes
    
    # Select the final feature set
    X = df_encoded[feature_columns]
    
    # Use the pre-fitted scaler to transform the data
    X_scaled = scaler.transform(X)
    
    return X_scaled

if __name__ == "__main__":
    main()