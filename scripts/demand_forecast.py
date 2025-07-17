import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path
import sys
import optuna

# --- 1. DEFINE PATHS ---
try:
    ROOT_DIR = Path(__file__).parent.parent.resolve()
except NameError:
    ROOT_DIR = Path.cwd()

DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "ev_model.pkl"

print(f"Data path: {DATA_PATH}")
print(f"Model path: {MODEL_PATH}")


# --- 2. FEATURE ENGINEERING FUNCTION ---
def create_historical_features(df):
    """Creates lag and rolling window features."""
    df = df.sort_values(by=['State', 'Date']).copy()
    grouped = df.groupby(['State'], observed=False)
    
    new_feature_cols = [
        'lag_1_day', 'lag_7_days', 'rolling_mean_7_days', 
        'rolling_std_7_days', 'rolling_mean_30_days'
    ]

    df['lag_1_day'] = grouped['EV_Sales_Quantity'].shift(1)
    df['lag_7_days'] = grouped['EV_Sales_Quantity'].shift(7)
    df['rolling_mean_7_days'] = grouped['EV_Sales_Quantity'].shift(1).rolling(window=7, min_periods=1).mean()
    df['rolling_std_7_days'] = grouped['EV_Sales_Quantity'].shift(1).rolling(window=7, min_periods=1).std()
    df['rolling_mean_30_days'] = grouped['EV_Sales_Quantity'].shift(1).rolling(window=30, min_periods=1).mean()
    
    df[new_feature_cols] = df[new_feature_cols].fillna(0)
    return df


# --- 3. LOAD AND PREPROCESS DATA ---
print("\nLoading and preprocessing data...")
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date', 'EV_Sales_Quantity'], inplace=True)
df['EV_Sales_Quantity'] = pd.to_numeric(df['EV_Sales_Quantity'], errors='coerce').fillna(0).astype(int)

# --- 4. CREATE BASE FEATURES ---
print("Engineering base features...")
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['quarter'] = df['Date'].dt.quarter
df['day_of_week'] = df['Date'].dt.dayofweek
df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
df['State'] = df['State'].astype('category')
df['Vehicle_Category'] = df['Vehicle_Category'].astype('category')


# --- 5. CATEGORY-SPECIFIC MODEL TRAINING LOOP ---
all_models = {}
categories = df['Vehicle_Category'].unique()

for category in categories:
    print("\n" + "="*80)
    print(f"Processing Category: {category}")
    print("="*80)

    category_df = df[df['Vehicle_Category'] == category].copy()
    category_df = create_historical_features(category_df)
    train_df = category_df[category_df['Date'] < '2023-01-01'].copy()
    val_df = category_df[category_df['Date'] >= '2023-01-01'].copy()

    if val_df.empty:
        print(f"Skipping {category} due to no validation data.")
        continue

    features = [
        'year', 'month', 'day', 'quarter', 'day_of_week', 'week_of_year',
        'is_weekend', 'State', 'lag_1_day', 'lag_7_days', 'rolling_mean_7_days',
        'rolling_std_7_days', 'rolling_mean_30_days', 'month_sin', 'month_cos',
        'day_of_week_sin', 'day_of_week_cos'
    ]
    target = 'EV_Sales_Quantity'

    X_train = train_df[features].copy()
    y_train_raw = train_df[target]
    X_val = val_df[features].copy()
    y_val_raw = val_df[target]

    state_categories_for_model = X_train['State'].cat.categories

    # --- PART 1: TRAIN THE CLASSIFIER (Will sales happen?) ---
    print(f"--- Training Classifier for {category} ---")
    y_train_class = (y_train_raw > 0).astype(int)
    y_val_class = (y_val_raw > 0).astype(int)
    
    classifier = lgb.LGBMClassifier(objective='binary', random_state=42)
    classifier.fit(X_train, y_train_class, eval_set=[(X_val, y_val_class)],
                   eval_metric='logloss', callbacks=[lgb.early_stopping(50, verbose=False)])
    print("[SUCCESS] Classifier trained.")

    # --- PART 2: TRAIN THE REGRESSOR (How many sales?) ---
    print(f"--- Training Regressor for {category} ---")
    # Filter for days with actual sales to train the regressor
    train_reg_df = train_df[train_df[target] > 0].copy()
    if train_reg_df.empty:
        print(f"Skipping regressor for {category} due to no sales data in training period.")
        regressor = None
    else:
        X_train_reg = train_reg_df[features]
        y_train_reg = np.log1p(train_reg_df[target])
        X_val_reg = X_val.copy() # Use the full validation set for evaluation
        y_val_reg = np.log1p(y_val_raw)

        def objective(trial):
            params = {
                'objective': 'regression_l1', 'metric': 'l1', 'n_estimators': 2000,
                'verbosity': -1, 'boosting_type': 'gbdt', 'seed': 42, 'n_jobs': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_reg, y_train_reg, eval_set=[(X_val_reg, y_val_reg)],
                      eval_metric='l1', callbacks=[lgb.early_stopping(50, verbose=False)])
            return model.best_score_['valid_0']['l1']

        print(f"Starting hyperparameter tuning for regressor...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)
        print(f"Best hyperparameters for regressor: {study.best_params}")

        print(f"Training final regressor...")
        best_params = study.best_params
        best_params.update({'objective': 'regression_l1', 'metric': 'l1', 'n_estimators': 2000,
                            'seed': 42, 'n_jobs': -1})
        regressor = lgb.LGBMRegressor(**best_params)
        regressor.fit(X_train_reg, y_train_reg, eval_set=[(X_val_reg, y_val_reg)],
                        eval_metric='l1', callbacks=[lgb.early_stopping(50, verbose=True)])
        print("[SUCCESS] Regressor trained.")

    # Bundle both models together
    model_bundle = {
        'classifier': classifier,
        'regressor': regressor,
        'state_categories': state_categories_for_model
    }
    all_models[category] = model_bundle
    print(f"[SUCCESS] Final model bundle for {category} created.")

# --- 6. SAVE THE MODEL BUNDLE ---
print("\n" + "="*80)
print(f"Saving all {len(all_models)} specialized models to {MODEL_PATH}...")
MODEL_PATH.parent.mkdir(exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(all_models, f)
print(f"[SUCCESS] Model bundle successfully saved!")
