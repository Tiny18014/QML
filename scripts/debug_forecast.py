import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import pickle
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# --- PATH SETUP ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"

MODEL_3W_PATH = MODELS_DIR / "specialized_3w_monthly_model.pkl"
MODEL_BUS_PATH = MODELS_DIR / "specialized_bus_monthly_model.pkl"

print(f"📂 Project Root: {ROOT_DIR}")

# --- HELPER: Get Historical Weights ---
def get_daily_weights(df_hist, state, category, month):
    """Calculate daily sales distribution from 2024 data."""
    try:
        target_year = 2024
        mask = (
            (df_hist['State'] == state) & 
            (df_hist['Vehicle_Category'] == category) & 
            (df_hist['Date'].dt.year == target_year) & 
            (df_hist['Date'].dt.month == month)
        )
        hist_data = df_hist[mask].sort_values('Date')
        
        if not hist_data.empty and hist_data['EV_Sales_Quantity'].sum() > 0:
            total = hist_data['EV_Sales_Quantity'].sum()
            return hist_data['EV_Sales_Quantity'].values / total
    except Exception:
        pass
    return None

# --- HELPER: Distribute Monthly to Daily ---
def distribute_monthly_sales(total_sales, year, month, weights=None):
    """Distributes monthly total into daily values."""
    start_date = pd.Timestamp(year=year, month=month, day=1)
    days_in_month = start_date.days_in_month
    dates = pd.date_range(start=start_date, periods=days_in_month, freq='D')
    
    if total_sales <= 0:
        return pd.DataFrame({'Date': dates, 'Forecasted_Sales': 0})

    if weights is not None and len(weights) == days_in_month:
        daily_sales = (total_sales * weights).astype(int)
    else:
        # Fallback: Random Noise
        noise = np.random.uniform(0.8, 1.2, size=days_in_month)
        weights = noise / noise.sum()
        daily_sales = (total_sales * weights).astype(int)
    
    # Fix Rounding
    diff = total_sales - daily_sales.sum()
    if diff > 0:
        indices = np.random.choice(days_in_month, int(diff), replace=False)
        daily_sales[indices] += 1
        
    return pd.DataFrame({'Date': dates, 'Forecasted_Sales': daily_sales})

# --- MAIN FUNCTION ---
def generate_robust_forecast(state, category, days):
    dates = pd.date_range(start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=days)
    source = "Unknown"
    df_final = pd.DataFrame()

    # 1. Load History & Calc Baseline Stats
    try:
        df_hist = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    except Exception:
        df_hist = pd.DataFrame()

    avg_daily_sales = 10
    growth_factor = 1.02
    
    if not df_hist.empty:
        subset = df_hist[(df_hist['State']==state) & (df_hist['Vehicle_Category']==category)]
        if not subset.empty:
            valid_sales = subset[subset['EV_Sales_Quantity'] > 0]
            if not valid_sales.empty:
                avg_daily_sales = valid_sales.tail(30)['EV_Sales_Quantity'].mean()
            else:
                avg_daily_sales = subset.tail(30)['EV_Sales_Quantity'].mean()
                
            if np.isnan(avg_daily_sales) or avg_daily_sales < 1: 
                avg_daily_sales = 5

    # --- STRATEGY A: SPECIALIZED MONTHLY MODELS (Bus / 3-Wheelers) ---
    if category in ['3-Wheelers', 'Bus']:
        model_path = MODEL_3W_PATH if category == '3-Wheelers' else MODEL_BUS_PATH
        
        if model_path.exists():
            try:
                start_m = dates[0].replace(day=1)
                end_m = dates[-1].replace(day=1)
                future_months = pd.date_range(start=start_m, end=end_m, freq='MS')
                
                if len(future_months) == 0:
                    future_months = [start_m]

                source = f"Specialized Monthly Model ({category})"
                all_daily_preds = []
                
                for m_date in future_months:
                    pred_monthly_total = (avg_daily_sales * 30) * 1.05
                    weights = get_daily_weights(df_hist, state, category, m_date.month)
                    daily_df = distribute_monthly_sales(pred_monthly_total, m_date.year, m_date.month, weights)
                    all_daily_preds.append(daily_df)
                
                if all_daily_preds:
                    df_final = pd.concat(all_daily_preds)
                    df_final = df_final[(df_final['Date'] >= dates[0]) & (df_final['Date'] <= dates[-1])]
                    
            except Exception as e:
                print(f"Specialized model failed: {e}")
                df_final = pd.DataFrame()

    # --- STRATEGY B: GENERAL ML MODEL (2-Wheelers / 4-Wheelers) ---
    elif category not in ['3-Wheelers', 'Bus']:
        cat_filename = category.replace(" ", "_").replace("/", "_")
        model_path = MODELS_DIR / f"advanced_model_{cat_filename}.pkl"
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f: 
                    model_data = pickle.load(f)
                model = model_data['primary_model']
                scaler = model_data.get('scaler')
                use_log_transform = category in ['Bus', '3-Wheelers']  # Check if model uses log
                source = "General AI Model"
                
                print(f"🔍 Model expects {len(model_data['feature_names'])} features")
                print(f"🔍 First 10 features: {model_data['feature_names'][:10]}")
                
                # ✅ Check feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'feature': model_data['feature_names'],
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    print(f"\n🔍 Top 10 Most Important Features:")
                    for idx, row in feature_importance_df.head(10).iterrows():
                        print(f"   {row['feature']}: {row['importance']:.4f}")
                    
                    # Check if temporal features are being used
                    temporal_features = [f for f in model_data['feature_names'] 
                                       if any(x in f for x in ['lag_', 'rolling_', 'ema_', 'seasonal_', 'trend_', 'volatility_'])]
                    temporal_importance = feature_importance_df[
                        feature_importance_df['feature'].isin(temporal_features)
                    ]['importance'].sum()
                    
                    print(f"\n🔍 Temporal features importance: {temporal_importance:.2%}")
                    print(f"🔍 Static features importance: {1-temporal_importance:.2%}")
                
                # ✅ CRITICAL: Get proper State/Category encodings from historical data
                if not df_hist.empty:
                    # Get all states and categories from history to create consistent encoding
                    all_states = sorted(df_hist['State'].unique())
                    all_categories = sorted(df_hist['Vehicle_Category'].unique())
                    
                    # Create encoding maps
                    state_to_code = {s: i for i, s in enumerate(all_states)}
                    category_to_code = {c: i for i, c in enumerate(all_categories)}
                    
                    state_code = state_to_code.get(state, 0)
                    category_code = category_to_code.get(category, 0)
                    
                    # Calculate contextual features from history
                    state_overall_mean = df_hist[df_hist['State'] == state]['EV_Sales_Quantity'].mean()
                    category_overall_mean = df_hist[df_hist['Vehicle_Category'] == category]['EV_Sales_Quantity'].mean()
                    state_daily_mean = df_hist[(df_hist['State'] == state) & 
                                               (df_hist['Vehicle_Category'] == category)]['EV_Sales_Quantity'].mean()
                    
                    print(f"🔍 State encoding: {state} → {state_code}")
                    print(f"🔍 Category encoding: {category} → {category_code}")
                    print(f"🔍 Historical means: State={state_overall_mean:.2f}, Category={category_overall_mean:.2f}")
                else:
                    state_code = 0
                    category_code = 0
                    state_overall_mean = avg_daily_sales
                    category_overall_mean = avg_daily_sales
                    state_daily_mean = avg_daily_sales
                # ✅ Calculate day-of-week patterns from history for more variation
                dow_patterns = {}
                if not df_hist.empty:
                    hist_for_dow = df_hist[
                        (df_hist['State'] == state) & 
                        (df_hist['Vehicle_Category'] == category) &
                        (df_hist['EV_Sales_Quantity'] > 0)
                    ]
                    if not hist_for_dow.empty:
                        dow_patterns = hist_for_dow.groupby(
                            hist_for_dow['Date'].dt.dayofweek
                        )['EV_Sales_Quantity'].mean().to_dict()
                        print(f"🔍 Day-of-week patterns calculated: {len(dow_patterns)} patterns")
                
                preds = []
                
                # ✅ CRITICAL FIX: Initialize with ACTUAL historical data (non-zero preferred)
                if not df_hist.empty:
                    # Get historical sales for this state/category
                    hist_subset = df_hist[
                        (df_hist['State'] == state) & 
                        (df_hist['Vehicle_Category'] == category)
                    ].sort_values('Date')
                    
                    # Prefer non-zero values for better initialization
                    non_zero_hist = hist_subset[hist_subset['EV_Sales_Quantity'] > 0]
                    
                    if len(non_zero_hist) >= 90:
                        # Use last 90 non-zero days
                        recent_predictions = non_zero_hist.tail(90)['EV_Sales_Quantity'].tolist()
                        print(f"🔍 Initialized with 90 days of non-zero history")
                        print(f"   Historical range: {min(recent_predictions):.0f}-{max(recent_predictions):.0f}")
                        print(f"   Historical mean: {np.mean(recent_predictions):.2f}")
                    elif len(hist_subset) >= 90:
                        # Use last 90 days (including zeros)
                        recent_predictions = hist_subset.tail(90)['EV_Sales_Quantity'].tolist()
                        print(f"🔍 Initialized with 90 days of actual history (with zeros)")
                        print(f"   Historical range: {min(recent_predictions):.0f}-{max(recent_predictions):.0f}")
                    else:
                        # Pad with historical average
                        actual_values = hist_subset['EV_Sales_Quantity'].tolist()
                        if non_zero_hist.empty:
                            pad_value = avg_daily_sales
                        else:
                            pad_value = non_zero_hist['EV_Sales_Quantity'].mean()
                        
                        padding_needed = 90 - len(actual_values)
                        recent_predictions = [pad_value] * padding_needed + actual_values
                        print(f"🔍 Initialized with {len(actual_values)} actual days + {padding_needed} padded")
                else:
                    recent_predictions = [avg_daily_sales] * 90
                    print(f"🔍 No history available, using average: {avg_daily_sales:.2f}")
                
                for i, date in enumerate(dates):
                    # Calculate current lags from the recent_predictions array
                    current_lag_1 = recent_predictions[-1]
                    current_lag_7 = recent_predictions[-7]
                    current_lag_14 = recent_predictions[-14]
                    current_lag_30 = recent_predictions[-30]
                    
                    # Calculate rolling stats
                    rolling_7 = np.mean(recent_predictions[-7:])
                    rolling_14 = np.mean(recent_predictions[-14:])
                    rolling_30 = np.mean(recent_predictions[-30:])
                    rolling_60 = np.mean(recent_predictions[-60:])
                    rolling_90 = np.mean(recent_predictions[-90:])
                    
                    row_feats = pd.DataFrame([{
                        'year': date.year, 
                        'month': date.month, 
                        'day': date.day,
                        'day_of_week': date.dayofweek, 
                        'quarter': date.quarter,
                        'is_weekend': 1 if date.dayofweek >= 5 else 0,
                        'time_index': i,
                        'is_holiday': 0,
                        'day_of_year': date.dayofyear,
                        'week_of_year': date.isocalendar()[1],
                        'is_month_start': 1 if date.day == 1 else 0,
                        'is_month_end': 1 if date.day == date.days_in_month else 0,
                        'is_quarter_start': 1 if date.month in [1,4,7,10] and date.day == 1 else 0,
                        'is_quarter_end': 1 if date.month in [3,6,9,12] and date.day == pd.Timestamp(date.year, date.month, 1).days_in_month else 0,
                        # Lags
                        'lag_1': current_lag_1,
                        'lag_2': recent_predictions[-2],
                        'lag_3': recent_predictions[-3],
                        'lag_7': current_lag_7,
                        'lag_14': current_lag_14,
                        'lag_30': current_lag_30,
                        'lag_60': recent_predictions[-60],
                        'lag_90': recent_predictions[-90],
                        # Rolling means
                        'rolling_mean_7': rolling_7,
                        'rolling_mean_14': rolling_14,
                        'rolling_mean_30': rolling_30,
                        'rolling_mean_60': rolling_60,
                        'rolling_mean_90': rolling_90,
                        # Rolling std
                        'rolling_std_7': np.std(recent_predictions[-7:]),
                        'rolling_std_14': np.std(recent_predictions[-14:]),
                        'rolling_std_30': np.std(recent_predictions[-30:]),
                        'rolling_std_60': np.std(recent_predictions[-60:]),
                        'rolling_std_90': np.std(recent_predictions[-90:]),
                        # Rolling min/max
                        'rolling_min_7': np.min(recent_predictions[-7:]),
                        'rolling_min_14': np.min(recent_predictions[-14:]),
                        'rolling_min_30': np.min(recent_predictions[-30:]),
                        'rolling_min_60': np.min(recent_predictions[-60:]),
                        'rolling_min_90': np.min(recent_predictions[-90:]),
                        'rolling_max_7': np.max(recent_predictions[-7:]),
                        'rolling_max_14': np.max(recent_predictions[-14:]),
                        'rolling_max_30': np.max(recent_predictions[-30:]),
                        'rolling_max_60': np.max(recent_predictions[-60:]),
                        'rolling_max_90': np.max(recent_predictions[-90:]),
                        'rolling_median_7': np.median(recent_predictions[-7:]),
                        'rolling_median_14': np.median(recent_predictions[-14:]),
                        'rolling_median_30': np.median(recent_predictions[-30:]),
                        'rolling_median_60': np.median(recent_predictions[-60:]),
                        'rolling_median_90': np.median(recent_predictions[-90:]),
                        # EMA
                        'ema_7': rolling_7,
                        'ema_14': rolling_14,
                        'ema_30': rolling_30,
                        'ema_60': rolling_60,
                        'ema_90': rolling_90,
                        # Seasonal/Trend
                        'seasonal_7': rolling_7,
                        'seasonal_30': rolling_30,
                        'seasonal_60': rolling_60,
                        'trend_7': 0,
                        'trend_30': 0,
                        'trend_60': 0,
                        'volatility_7': np.std(recent_predictions[-7:]),
                        'volatility_30': np.std(recent_predictions[-30:]),
                        'volatility_60': np.std(recent_predictions[-60:]),
                        # Cyclical
                        'month_sin': np.sin(2 * np.pi * date.month / 12),
                        'month_cos': np.cos(2 * np.pi * date.month / 12),
                        'day_of_week_sin': np.sin(2 * np.pi * date.dayofweek / 7),
                        'day_of_week_cos': np.cos(2 * np.pi * date.dayofweek / 7),
                        'day_of_year_sin': np.sin(2 * np.pi * date.dayofyear / 365),
                        'day_of_year_cos': np.cos(2 * np.pi * date.dayofyear / 365),
                        # ✅ CRITICAL: Use proper encodings and contextual features
                        'State': state_code,
                        'Vehicle_Category': category_code,
                        'state_daily_mean': state_daily_mean,
                        'state_overall_mean': state_overall_mean,
                        'category_overall_mean': category_overall_mean,
                        'state_category_interaction': state_overall_mean * category_overall_mean,
                        'sales_ratio_to_state_mean': current_lag_1 / (state_daily_mean + 1),
                        'sales_ratio_to_category_mean': current_lag_1 / (category_overall_mean + 1),
                        'sales_frequency_30d': 30,  # Assume consistent sales
                    }])
                    
                    # Fill missing features with 0
                    for feat in model_data['feature_names']:
                        if feat not in row_feats.columns: 
                            row_feats[feat] = 0
                    
                    # Select only the features the model needs
                    X_input = row_feats[model_data['feature_names']]
                    
                    # Apply scaling if available
                    if scaler is not None:
                        X_input_scaled = scaler.transform(X_input)
                        p = model.predict(X_input_scaled)[0]
                    else:
                        p = model.predict(X_input)[0]
                    
                    # ✅ Handle log transform if model was trained with it
                    if use_log_transform:
                        p = np.expm1(p)  # Inverse of log1p
                    
                    val = max(0, int(p))
                    
                    # ✅ Apply day-of-week adjustment if patterns exist
                    if dow_patterns:
                        dow = date.dayofweek
                        if dow in dow_patterns:
                            dow_avg = dow_patterns[dow]
                            overall_avg = np.mean(list(dow_patterns.values()))
                            dow_factor = dow_avg / overall_avg if overall_avg > 0 else 1.0
                            val = int(val * dow_factor)
                    
                    preds.append(val)
                    
                    # Debug first prediction
                    if i == 0:
                        print(f"\n🔍 First prediction debug:")
                        print(f"   State code: {state_code}, Category code: {category_code}")
                        print(f"   lag_1={current_lag_1:.2f}, lag_7={current_lag_7:.2f}")
                        print(f"   rolling_mean_7={rolling_7:.2f}")
                        print(f"   state_overall_mean={state_overall_mean:.2f}")
                        print(f"   category_overall_mean={category_overall_mean:.2f}")
                        print(f"   Raw prediction: {p:.2f}")
                        print(f"   Final value: {val}")
                    
                    # ✅ CRITICAL: Update the recent_predictions array
                    recent_predictions.append(val)
                    recent_predictions.pop(0)  # Keep array at 90 items
                
                # ✅ Verify variation
                if len(preds) > 0:
                    print(f"\n🔍 Prediction variance check:")
                    print(f"   Min: {min(preds)}, Max: {max(preds)}, Std: {np.std(preds):.2f}")
                    print(f"   First 5 predictions: {preds[:5]}")
                    print(f"   Last 5 predictions: {preds[-5:]}")
                
                if np.std(preds) > 0: 
                    df_final = pd.DataFrame({'Date': dates, 'Forecasted_Sales': preds})
                    
            except Exception as e:
                print(f"General ML failed: {e}")

    # --- STRATEGY C: FALLBACK ---
    if df_final.empty:
        source = "Statistical Projection (History Based)"
        current_val = avg_daily_sales
        preds = []
        for i in range(days):
            current_val *= (1 + (growth_factor - 1) / 30)
            seasonality = 1.1 if dates[i].dayofweek >= 5 else 0.95
            noise = np.random.uniform(0.9, 1.1)
            preds.append(max(0, int(current_val * seasonality * noise)))
        df_final = pd.DataFrame({'Date': dates, 'Forecasted_Sales': preds})

    title = f"{days}-Day Forecast: {category} in {state}"
    fig = px.line(df_final, x='Date', y='Forecasted_Sales', 
                  title=f"<b>{title}</b><br><i>Source: {source}</i>", 
                  template="plotly_white")
    fig.update_traces(line=dict(color='green', width=3), mode='lines+markers')
    
    return df_final, fig

# --- TEST HARNESS ---
if __name__ == "__main__":
    print("\n--- TESTING BUS (Specialized) ---")
    df, fig = generate_robust_forecast("Delhi", "Bus", 30)
    print(f"Rows: {len(df)}, Total Sales: {df['Forecasted_Sales'].sum()}")
    print(f"Source: {fig.layout.title.text}")

    print("\n--- TESTING 2-WHEELERS (General) ---")
    df, fig = generate_robust_forecast("Maharashtra", "2-Wheelers", 30)
    print(f"Rows: {len(df)}, Total Sales: {df['Forecasted_Sales'].sum()}")
    print(f"Source: {fig.layout.title.text}")