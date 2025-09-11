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
import optuna
from datetime import datetime, timedelta
import joblib

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
    df['Date'] = pd.to_datetime(df['Date'])
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
    
    print(f"✅ Created {len(df.columns)} features")
    return df

def prepare_data_for_training(df):
    """Prepare data with proper encoding and scaling."""
    print("📊 Preparing data for training...")
    
    # Convert categorical variables
    df['State'] = df['State'].astype('category')
    df['Vehicle_Category'] = df['Vehicle_Category'].astype('category')
    
    # Create feature matrix
    feature_columns = [col for col in df.columns if col not in ['Date', 'EV_Sales_Quantity']]
    
    # Encode categorical variables
    df_encoded = df.copy()
    df_encoded['State'] = df_encoded['State'].cat.codes
    df_encoded['Vehicle_Category'] = df_encoded['Vehicle_Category'].cat.codes
    
    X = df_encoded[feature_columns]
    y = df_encoded['EV_Sales_Quantity']
    
    # Scale features
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
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
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
            'objective': 'regression',
            'metric': 'mae',
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        y_pred = model.predict(X_val)
        return mean_absolute_error(y_val, y_pred)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_params}")
    
    # Train final model with best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'mae',
        'n_estimators': 1000,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    })
    
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    return final_model, study.best_params

def main():
    """Main training function."""
    print("📈 Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Vehicle categories: {df['Vehicle_Category'].unique()}")
    print(f"States: {len(df['State'].unique())}")
    
    # Create advanced features
    df_advanced = create_advanced_features(df)
    
    # Split data by index proportions (robust for short date ranges in CI)
    df_advanced['Date'] = pd.to_datetime(df_advanced['Date'])
    df_advanced = df_advanced.sort_values('Date')

    total_rows = len(df_advanced)
    train_end = max(1, int(total_rows * 0.7))
    val_end = max(train_end + 1, int(total_rows * 0.85))

    train_df = df_advanced.iloc[:train_end].copy()
    val_df = df_advanced.iloc[train_end:val_end].copy()
    test_df = df_advanced.iloc[val_end:].copy()

    # Guarantee at least 1 sample in train
    if train_df.empty and total_rows > 0:
        train_df = df_advanced.iloc[:1].copy()
        val_df = df_advanced.iloc[1:2].copy() if total_rows > 1 else df_advanced.iloc[:0].copy()
        test_df = df_advanced.iloc[2:].copy() if total_rows > 2 else df_advanced.iloc[:0].copy()

    # Final fallback: if still empty (defensive), use all data as train
    if train_df.empty:
        train_df = df_advanced.copy()
        val_df = df_advanced.iloc[:0].copy()
        test_df = df_advanced.iloc[:0].copy()

    print(f"\n📊 Data splits (final):")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    print(f"\n📊 Data splits:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Prepare training data
    X_train, y_train, feature_names, scaler = prepare_data_for_training(train_df)
    X_val, y_val, _, _ = prepare_data_for_training(val_df)
    X_test, y_test, _, _ = prepare_data_for_training(test_df)
    
    # Train ensemble models
    trained_models, scores, best_models = train_models(X_train, y_train, X_val, y_val, feature_names)
    
    # Create optimized model
    optimized_model, best_params = create_optimized_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print("\n🧪 Final Evaluation on Test Set:")
    test_predictions = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        test_predictions[name] = y_pred
        print(f"  {name}: MAE={mae:.2f}, R²={r2:.4f}")
    
    # Optimized model evaluation
    y_pred_optimized = optimized_model.predict(X_test)
    mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
    r2_optimized = r2_score(y_test, y_pred_optimized)
    print(f"  Optimized: MAE={mae_optimized:.2f}, R²={r2_optimized:.4f}")
    
    # Create ensemble prediction (average of top 3 models)
    top_models = sorted(scores.items(), key=lambda x: x[1]['R2'], reverse=True)[:3]
    ensemble_pred = np.mean([test_predictions[name] for name, _ in top_models], axis=0)
    mae_ensemble = mean_absolute_error(y_test, ensemble_pred)
    r2_ensemble = r2_score(y_test, ensemble_pred)
    print(f"  Ensemble (Top 3): MAE={mae_ensemble:.2f}, R²={r2_ensemble:.4f}")
    
    # Save the best model and metadata
    print("\n💾 Saving models...")
    
    # Save the optimized model as primary
    model_data = {
        'primary_model': optimized_model,
        'ensemble_models': trained_models,
        'scaler': scaler,
        'feature_names': feature_names,
        'best_params': best_params,
        'scores': scores,
        'test_scores': {
            'optimized': {'MAE': mae_optimized, 'R2': r2_optimized},
            'ensemble': {'MAE': mae_ensemble, 'R2': r2_ensemble}
        },
        'training_info': {
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'feature_count': len(feature_names),
            'training_date': datetime.now().isoformat()
        }
    }
    
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Save scaler separately
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open(FEATURE_NAMES_PATH, 'wb') as f:
        pickle.dump(feature_names, f)
    
    print(f"✅ Models saved to {MODEL_PATH}")
    print(f"✅ Scaler saved to {SCALER_PATH}")
    print(f"✅ Feature names saved to {FEATURE_NAMES_PATH}")
    
    print("\n🎉 Training completed!")
    print(f"🏆 Best R² Score: {r2_optimized:.4f}")
    print(f"🏆 Best MAE Score: {mae_optimized:.2f}")
    
    return model_data

if __name__ == "__main__":
    main()
