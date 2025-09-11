import pandas as pd
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta
import numpy as np

# --- 1. DEFINE PATHS ---
try:
    ROOT_DIR = Path(__file__).parent.parent.resolve()
except NameError:
    # Fallback for interactive environments
    ROOT_DIR = Path.cwd()

DATA_DIR = ROOT_DIR / "data"
# The primary input is now the user's own dataset.
INPUT_CSV_PATH = DATA_DIR / "EV_Dataset.csv"
BACKUP_PATH = DATA_DIR / "EV_Dataset_backup.csv"


def generate_sample_data():
    """
    Generate realistic sample EV data for testing when the primary
    dataset is unavailable.
    """
    print("Generating sample EV data for development/testing...")
    states = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
        'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
        'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
        'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
        'Delhi', 'Chandigarh', 'Puducherry'
    ]
    categories = ['2-Wheelers', '3-Wheelers', '4-Wheelers', 'Bus', 'Others']
    start_date = datetime(2021, 1, 1)
    end_date = datetime.now()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D') # Daily frequency for sample
    data = []
    for date in date_range:
        for state in states:
            for category in categories:
                base_demand = {'2-Wheelers': 50, '3-Wheelers': 15, '4-Wheelers': 10, 'Bus': 2, 'Others': 5}
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                growth_factor = 1 + 0.2 * ((date - start_date).days / 365.25)
                state_factor = {'Maharashtra': 1.5, 'Karnataka': 1.4, 'Tamil Nadu': 1.3}.get(state, 0.8)
                final_qty = int(base_demand[category] * seasonal_factor * growth_factor * state_factor * np.random.uniform(0.7, 1.3))
                if np.random.random() < 0.3: final_qty = 0
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'State': state,
                    'Vehicle_Category': category,
                    'EV_Sales_Quantity': final_qty
                })
    return pd.DataFrame(data)


def standardize_data(df):
    """
    Standardize the data format from the local CSV to match the pipeline's requirements.
    """
    print("Standardizing data format...")
    # Rename columns to handle any variations in the source CSV
    df.rename(columns={
        'Registration Date': 'Date',
        'Month, Year': 'Date', # Handle different date formats
        'State': 'State',
        'Vehicle Category': 'Vehicle_Category',
        'Registrations': 'EV_Sales_Quantity',
        'Number of EVs Registered': 'EV_Sales_Quantity'
    }, inplace=True, errors='ignore')
    
    required_cols = ['Date', 'State', 'Vehicle_Category', 'EV_Sales_Quantity']
    if not all(col in df.columns for col in required_cols):
        print(f"[ERROR] The source data is missing required columns. Found: {df.columns.tolist()}")
        return None

    # Final processing
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['EV_Sales_Quantity'] = pd.to_numeric(df['EV_Sales_Quantity'], errors='coerce').fillna(0)
    df.dropna(subset=['Date'], inplace=True)
    
    # Aggregate data to handle any potential duplicate entries
    df_final = df.groupby(['Date', 'State', 'Vehicle_Category'])['EV_Sales_Quantity'].sum().reset_index()
    return df_final


def main():
    """
    Main function to validate and prepare the local EV dataset.
    """
    print("--- Starting Local Data Ingestion and Validation Process ---")
    DATA_DIR.mkdir(exist_ok=True)
    
    df = None
    if INPUT_CSV_PATH.exists():
        print(f"Found local dataset at: {INPUT_CSV_PATH}")
        df = pd.read_csv(INPUT_CSV_PATH)
    else:
        print(f"[WARNING] No local dataset found at {INPUT_CSV_PATH}.")
        response = input("Generate sample data for development? (y/n): ")
        if response.lower() == 'y':
            df = generate_sample_data()
        else:
            print("Data ingestion cancelled. No data available.")
            sys.exit(1)

    # Standardize and save the data, overwriting the source file to ensure it's clean
    df_final = standardize_data(df)
    if df_final is not None:
        # Create a backup before overwriting
        if INPUT_CSV_PATH.exists():
            pd.read_csv(INPUT_CSV_PATH).to_csv(BACKUP_PATH, index=False)
            print(f"Backup of original data created at: {BACKUP_PATH}")

        df_final.to_csv(INPUT_CSV_PATH, index=False)
        print(f"\n[SUCCESS] Data successfully validated and saved to: {INPUT_CSV_PATH}")
        print(f"Total records: {len(df_final)}")
        print(f"Date range: {df_final['Date'].min().date()} to {df_final['Date'].max().date()}")
    else:
        print("[ERROR] Failed to standardize data. Aborting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
