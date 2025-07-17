import pandas as pd
from pathlib import Path
import requests
import sys
from io import StringIO
import json
import time
from datetime import datetime, timedelta
import numpy as np

# --- 1. DEFINE PATHS ---
try:
    ROOT_DIR = Path(__file__).parent.parent.resolve()
except NameError:
    ROOT_DIR = Path.cwd()

DATA_DIR = ROOT_DIR / "data"
OUTPUT_CSV_PATH = DATA_DIR / "EV_Dataset.csv"
BACKUP_PATH = DATA_DIR / "EV_Dataset_backup.csv"

# --- 2. DEFINE DATA SOURCES ---
# Updated with working data sources and sample data generation
DATA_SOURCES = {
    # Government of India Open Data Platform
    "gov_india_ev_data": {
        "url": "https://www.data.gov.in/api/datastore/resource.json",
        "params": {
            "resource_id": "4dd42d9b-1de8-4b6b-9f4a-5f6c8b3a4e2d",  # Example resource ID
            "limit": 10000
        },
        "type": "api"
    },
    # Alternative: Manual CSV URLs (if available)
    "direct_csv": [
        "https://raw.githubusercontent.com/datasets/electric-vehicle-data/main/india_ev_registrations.csv",
        "https://data.gov.in/files/ogdp/morth/ev_registrations_state_wise.csv"
    ],
    # Kaggle dataset (requires manual download)
    "kaggle_backup": "https://www.kaggle.com/datasets/sid321axn/electric-vehicle-india-dataset"
}

def generate_sample_data():
    """
    Generate sample EV data for testing when real data sources are unavailable.
    This creates realistic-looking data for development purposes.
    """
    print("Generating sample EV data for testing...")
    
    # Indian states
    states = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
        'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
        'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
        'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
        'Delhi', 'Chandigarh', 'Puducherry'
    ]
    
    # Vehicle categories
    categories = ['2-Wheelers', '3-Wheelers', '4-Wheelers', 'Bus', 'Others']
    
    # Generate date range (last 3 years)
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    
    for date in date_range:
        for state in states:
            for category in categories:
                # Generate realistic sales quantities with seasonal patterns
                base_demand = {
                    '2-Wheelers': 50,
                    '3-Wheelers': 15,
                    '4-Wheelers': 10,
                    'Bus': 2,
                    'Others': 5
                }
                
                # Add seasonal variation
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                
                # Add growth trend
                years_since_start = (date - start_date).days / 365.25
                growth_factor = 1 + 0.2 * years_since_start
                
                # Add state-specific factors
                state_factors = {
                    'Maharashtra': 1.5, 'Karnataka': 1.4, 'Tamil Nadu': 1.3,
                    'Gujarat': 1.2, 'Delhi': 1.1, 'Uttar Pradesh': 1.0
                }
                state_factor = state_factors.get(state, 0.8)
                
                # Calculate final quantity with some randomness
                base_qty = base_demand[category]
                final_qty = int(base_qty * seasonal_factor * growth_factor * state_factor * np.random.uniform(0.5, 1.5))
                
                # Add some days with zero sales (realistic for smaller categories/states)
                if np.random.random() < 0.3:
                    final_qty = 0
                
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'State': state,
                    'Vehicle_Category': category,
                    'EV_Sales_Quantity': final_qty
                })
    
    return pd.DataFrame(data)

def fetch_from_api(source_config):
    """Fetch data from API endpoints."""
    try:
        if source_config["type"] == "api":
            response = requests.get(
                source_config["url"], 
                params=source_config.get("params", {}),
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            # Convert API response to DataFrame (structure depends on API)
            if "records" in data:
                return pd.DataFrame(data["records"])
            else:
                return pd.DataFrame(data)
                
    except Exception as e:
        print(f"API fetch failed: {e}")
        return None

def fetch_from_csv(urls):
    """Fetch data from CSV URLs."""
    for url in urls:
        try:
            print(f"Trying CSV source: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            return df
            
        except Exception as e:
            print(f"CSV fetch failed for {url}: {e}")
            continue
    
    return None

def standardize_data(df):
    """
    Standardize the data format regardless of source.
    """
    # Common column name mappings
    column_mappings = {
        'date': 'Date',
        'Date': 'Date',
        'state': 'State',
        'State': 'State',
        'state_name': 'State',
        'vehicle_category': 'Vehicle_Category',
        'Vehicle_Category': 'Vehicle_Category',
        'category': 'Vehicle_Category',
        'ev_sales': 'EV_Sales_Quantity',
        'EV_Sales_Quantity': 'EV_Sales_Quantity',
        'quantity': 'EV_Sales_Quantity',
        'registrations': 'EV_Sales_Quantity',
        'count': 'EV_Sales_Quantity'
    }
    
    # Try to map columns
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    # Ensure required columns exist
    required_cols = ['Date', 'State', 'Vehicle_Category', 'EV_Sales_Quantity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Data may need manual adjustment.")
        return None
    
    # Standardize data types
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['EV_Sales_Quantity'] = pd.to_numeric(df['EV_Sales_Quantity'], errors='coerce').fillna(0)
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['Date'])
    
    # Standardize vehicle categories
    category_mappings = {
        'TWO WHEELER': '2-Wheelers',
        'MOTOR CYCLE': '2-Wheelers',
        'SCOOTER': '2-Wheelers',
        'E-RICKSHAW': '3-Wheelers',
        'THREE WHEELER': '3-Wheelers',
        'FOUR WHEELER': '4-Wheelers',
        'CAR': '4-Wheelers',
        'BUS': 'Bus',
        'TRUCK': 'Others'
    }
    
    for old_cat, new_cat in category_mappings.items():
        df.loc[df['Vehicle_Category'].str.contains(old_cat, case=False, na=False), 'Vehicle_Category'] = new_cat
    
    return df

def backup_existing_data():
    """Create backup of existing data before updating."""
    if OUTPUT_CSV_PATH.exists():
        try:
            # Copy existing data to backup
            existing_df = pd.read_csv(OUTPUT_CSV_PATH)
            existing_df.to_csv(BACKUP_PATH, index=False)
            print(f"Backup created: {BACKUP_PATH}")
            return existing_df
        except Exception as e:
            print(f"Failed to create backup: {e}")
    return None

def fetch_and_prepare_data():
    """
    Main function to fetch, prepare, and save EV data.
    """
    print("Starting EV data ingestion process...")
    
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    # Create backup of existing data
    backup_df = backup_existing_data()
    
    df = None
    
    # Try to fetch from various sources
    print("\n1. Attempting to fetch from government API...")
    df = fetch_from_api(DATA_SOURCES["gov_india_ev_data"])
    
    if df is None:
        print("\n2. Attempting to fetch from CSV sources...")
        df = fetch_from_csv(DATA_SOURCES["direct_csv"])
    
    if df is None:
        print("\n3. All external sources failed.")
        
        # Check if backup exists and is recent (less than 7 days old)
        if backup_df is not None and OUTPUT_CSV_PATH.exists():
            file_age = datetime.now() - datetime.fromtimestamp(OUTPUT_CSV_PATH.stat().st_mtime)
            if file_age < timedelta(days=7):
                print("Using existing recent data (less than 7 days old).")
                print(f"Data file last modified: {datetime.fromtimestamp(OUTPUT_CSV_PATH.stat().st_mtime)}")
                return
        
        # Generate sample data for development
        response = input("\nNo data sources available. Generate sample data for development? (y/n): ")
        if response.lower() == 'y':
            df = generate_sample_data()
        else:
            print("Data ingestion cancelled.")
            sys.exit(1)
    
    # Standardize the data format
    print("\nStandardizing data format...")
    df = standardize_data(df)
    
    if df is None:
        print("Failed to standardize data. Please check the data format.")
        sys.exit(1)
    
    # Group by date, state, and category to handle duplicates
    print("Aggregating data...")
    df_final = df.groupby(['Date', 'State', 'Vehicle_Category'])['EV_Sales_Quantity'].sum().reset_index()
    
    # Save the prepared data
    df_final.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"\n[SUCCESS] Data successfully saved to: {OUTPUT_CSV_PATH}")
    print(f"Total records: {len(df_final)}")
    print(f"Date range: {df_final['Date'].min()} to {df_final['Date'].max()}")
    print(f"States: {df_final['State'].nunique()}")
    print(f"Vehicle categories: {df_final['Vehicle_Category'].unique()}")
    
    # Display sample data
    print("\nSample data:")
    print(df_final.head(10))

if __name__ == "__main__":
    try:
        fetch_and_prepare_data()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)