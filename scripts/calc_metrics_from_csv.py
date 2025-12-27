import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
PREDS_PATH = ROOT_DIR / "output" / "real_model_predictions.csv"

def calculate_metrics():
    print(f"Reading {PREDS_PATH}...")
    try:
        df = pd.read_csv(PREDS_PATH)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Check columns
    print("Columns:", df.columns.tolist())
    
    # Ensure numeric
    df['EV_Sales_Quantity'] = pd.to_numeric(df['EV_Sales_Quantity'], errors='coerce')
    df['Predicted_Sales'] = pd.to_numeric(df['Predicted_Sales'], errors='coerce')
    
    # Drop NaNs
    df = df.dropna(subset=['EV_Sales_Quantity', 'Predicted_Sales'])
    
    if df.empty:
        print("No valid data found.")
        return

    # 1. Aggregate to National Monthly (to match paper's scale)
    df['Date'] = pd.to_datetime(df['Date'])
    monthly_df = df.groupby('Date')[['EV_Sales_Quantity', 'Predicted_Sales']].sum().reset_index()
    
    y_true = monthly_df['EV_Sales_Quantity']
    y_pred = monthly_df['Predicted_Sales']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    print("\n--- National Monthly Aggregated Metrics (Classical LightGBM) ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2: {r2:.4f}")
    
    # 2. Category-wise Average (as requested by user "averagee it")
    # We need to see if 'Vehicle_Class' is populated
    if 'Vehicle_Class' in df.columns:
        print("\n--- Category-wise Metrics ---")
        cats = df['Vehicle_Class'].unique()
        maes, rmses, mapes, r2s = [], [], [], []
        
        for cat in cats:
            cat_df = df[df['Vehicle_Class'] == cat]
            # Aggregate monthly for this category
            cat_monthly = cat_df.groupby('Date')[['EV_Sales_Quantity', 'Predicted_Sales']].sum().reset_index()
            
            if len(cat_monthly) < 2:
                continue
                
            yt = cat_monthly['EV_Sales_Quantity']
            yp = cat_monthly['Predicted_Sales']
            
            m = mean_absolute_error(yt, yp)
            r = np.sqrt(mean_squared_error(yt, yp))
            mp = np.mean(np.abs((yt - yp) / yt)) * 100
            r2_val = r2_score(yt, yp)
            
            maes.append(m)
            rmses.append(r)
            mapes.append(mp)
            r2s.append(r2_val)
            
            print(f"{cat}: MAE={m:.2f}, MAPE={mp:.2f}%, R2={r2_val:.4f}")
            
        print("\n--- Average of Category Metrics ---")
        print(f"Avg MAE: {np.mean(maes):.2f}")
        print(f"Avg RMSE: {np.mean(rmses):.2f}")
        print(f"Avg MAPE: {np.mean(mapes):.2f}%")
        print(f"Avg R2: {np.mean(r2s):.4f}")

if __name__ == "__main__":
    calculate_metrics()
