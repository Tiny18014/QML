import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings
import re

warnings.filterwarnings('ignore')

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_FILE = DATA_DIR / "EV_Dataset.csv"

# --- Helper Functions ---

def extract_state_name_from_header(header_text):
    try:
        if "of " in header_text and " (" in header_text:
            start_idx = header_text.find("of ") + 3
            end_idx = header_text.find(" (")
            if start_idx < end_idx:
                return header_text[start_idx:end_idx].strip()
        return None
    except:
        return None

def map_vehicle_category(category):
    if not isinstance(category, str): return 'Others'
    cat_upper = category.upper().strip()
    
    direct_map = {
        'TWO WHEELER(NT)': '2-Wheelers', 'TWO WHEELER(T)': '2-Wheelers',
        'THREE WHEELER(T)': '3-Wheelers', 'THREE WHEELER(NT)': '3-Wheelers',
        'E-RICKSHAW(P)': '3-Wheelers', 'E-RICKSHAW WITH CART (G)': '3-Wheelers',
        'MOTOR CAR': '4-Wheelers', 'LIGHT MOTOR VEHICLE': '4-Wheelers',
        'BUS': 'Bus', 'HEAVY PASSENGER VEHICLE': 'Bus'
    }
    
    if cat_upper in direct_map: return direct_map[cat_upper]
    if any(x in cat_upper for x in ['TWO WHEELER', 'M-CYCLE', 'SCOOTER']): return '2-Wheelers'
    if any(x in cat_upper for x in ['THREE WHEELER', 'RICKSHAW']): return '3-Wheelers'
    if any(x in cat_upper for x in ['MOTOR CAR', 'LMV', 'JEEP']): return '4-Wheelers'
    if 'BUS' in cat_upper: return 'Bus'
    
    return 'Others'

def get_historical_weights(master_df, state, category, target_month):
    """
    Calculates the daily distribution pattern (weights) from the previous year (2024).
    Returns a list of weights for each day of the month.
    """
    if master_df.empty: return None
    
    # Look at 2024 data for the same State, Category, and Month
    target_year = 2024
    mask = (
        (master_df['State'] == state) & 
        (master_df['Vehicle_Category'] == category) & 
        (master_df['Date'].dt.year == target_year) & 
        (master_df['Date'].dt.month == target_month)
    )
    
    hist_data = master_df[mask].sort_values('Date')
    
    if hist_data.empty or hist_data['EV_Sales_Quantity'].sum() == 0:
        return None
    
    # Calculate weights
    total_hist_sales = hist_data['EV_Sales_Quantity'].sum()
    weights = hist_data['EV_Sales_Quantity'].values / total_hist_sales
    
    return weights

def distribute_monthly_to_daily(row, master_df):
    """
    Splits a single monthly row into daily rows using historical patterns.
    """
    year = row['Year']
    month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
    month = month_map.get(row['Month_Name'], 1)
    
    start_date = pd.Timestamp(year=year, month=month, day=1)
    days_in_month = start_date.days_in_month
    dates = pd.date_range(start=start_date, periods=days_in_month, freq='D')
    
    total_sales = row['EV_Sales_Quantity']
    
    if total_sales <= 0:
        return pd.DataFrame({'Date': dates, 'State': row['State'], 'Vehicle_Category': row['Vehicle_Category'], 'EV_Sales_Quantity': 0})

    # --- PATTERN MATCHING ---
    # Try to get weights from 2024 data
    weights = get_historical_weights(master_df, row['State'], row['Vehicle_Category'], month)
    
    # Validate weights length (leap years etc might cause mismatch, though unlikely for 2024/2025)
    if weights is not None and len(weights) == days_in_month:
        # Use Historical Pattern
        daily_sales = (total_sales * weights).astype(int)
    else:
        # Fallback: Random Distribution
        # print(f"      ⚠️ No history pattern for {row['State']} - {row['Vehicle_Category']} (Month {month}). Using random.")
        noise = np.random.uniform(0.8, 1.2, size=days_in_month)
        weights = noise / noise.sum()
        daily_sales = (total_sales * weights).astype(int)
    
    # Fix Rounding to match exact total
    diff = total_sales - daily_sales.sum()
    if diff > 0:
        indices = np.random.choice(days_in_month, diff, replace=False)
        daily_sales[indices] += 1
    elif diff < 0:
        # If we overshot (rare with floor), remove from highest days
        indices = np.argsort(daily_sales)[::-1][:abs(diff)]
        daily_sales[indices] -= 1
    
    return pd.DataFrame({
        'Date': dates,
        'State': row['State'],
        'Vehicle_Category': row['Vehicle_Category'],
        'EV_Sales_Quantity': daily_sales
    })

def process_single_state_file(file_path):
    try:
        print(f"   -> Processing {file_path.name}...")
        df_header = pd.read_excel(file_path, header=None, nrows=1)
        state_name = None
        if not df_header.empty:
            state_name = extract_state_name_from_header(str(df_header.iloc[0, 0]))
            print(f"      -> Detected State: {state_name}")

        df = pd.read_excel(file_path)
        data_start_row = None
        for idx, row in df.iterrows():
            if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip().isdigit():
                data_start_row = idx
                break
        
        if data_start_row is None: return None
        
        vehicle_data = []
        for idx in range(data_start_row, len(df)):
            row = df.iloc[idx]
            if pd.notna(row.iloc[0]):
                vehicle_class = str(row.iloc[1]).strip()
                if vehicle_class and vehicle_class.lower() != "nan":
                    month_cols = {2:'JAN', 3:'FEB', 4:'MAR', 5:'APR', 6:'MAY', 7:'JUN', 8:'JUL', 9:'AUG', 10:'SEP', 11:'OCT', 12:'NOV', 13:'DEC'}
                    for col_idx, m_name in month_cols.items():
                        if col_idx < len(row):
                            val = row.iloc[col_idx]
                            if pd.notna(val):
                                val_str = str(val).replace(',', '').strip()
                                if val_str.isdigit():
                                    vehicle_data.append({
                                        'State': state_name or 'Unknown',
                                        'Vehicle_Category': map_vehicle_category(vehicle_class),
                                        'Month_Name': m_name,
                                        'EV_Sales_Quantity': int(val_str),
                                        'Year': 2025
                                    })
        return pd.DataFrame(vehicle_data) if vehicle_data else None
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return None

# --- Main Execution ---
def merge_raw_data():
    if not RAW_DIR.exists(): return
    print(f"🔍 Scanning {RAW_DIR} for new data files...")
    raw_files = list(RAW_DIR.glob("*.xlsx"))
    if not raw_files: return

    # Load Master Data (Needed for Pattern Matching)
    if OUTPUT_FILE.exists():
        print("   -> Loading Master Dataset for Pattern Matching...")
        master_df = pd.read_csv(OUTPUT_FILE)
        master_df['Date'] = pd.to_datetime(master_df['Date'])
    else:
        master_df = pd.DataFrame(columns=['Date', 'State', 'Vehicle_Category', 'EV_Sales_Quantity'])

    all_daily_data = []
    
    for file in raw_files:
        if file.name.startswith("~$"): continue
        df_monthly = process_single_state_file(file)
        
        if df_monthly is not None and not df_monthly.empty:
            # Convert Monthly -> Daily using History Pattern
            for _, row in df_monthly.iterrows():
                # Pass master_df to find historical patterns
                df_daily_chunk = distribute_monthly_to_daily(row, master_df)
                all_daily_data.append(df_daily_chunk)

    if not all_daily_data:
        print("⚠️ No valid data extracted.")
        return

    print("🔗 Consolidating new data...")
    new_data_df = pd.concat(all_daily_data, ignore_index=True)
    
    # Merge with Master
    combined_df = pd.concat([master_df, new_data_df], ignore_index=True)
    
    print("🧹 Final Aggregation...")
    # We drop duplicates based on Date/State/Cat to prevent double-counting if script runs twice
    # But we sum if they are genuine new entries
    final_df = combined_df.groupby(['Date', 'State', 'Vehicle_Category'], as_index=False)['EV_Sales_Quantity'].sum()
    final_df.sort_values(['State', 'Vehicle_Category', 'Date'], inplace=True)
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Successfully updated {OUTPUT_FILE}")
    print(f"   -> Total Rows: {len(final_df)}")

if __name__ == "__main__":
    merge_raw_data()