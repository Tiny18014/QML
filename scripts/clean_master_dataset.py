# scripts/clean_master_dataset.py
from pathlib import Path
import pandas as pd

# Define paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATASET_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"

def clean_dataset():
    """
    Loads the master dataset and removes any rows where the 
    State or Vehicle_Category is 'Unknown' or missing.
    """
    if not DATASET_PATH.exists():
        print(f"❌ Master dataset not found at: {DATASET_PATH}")
        return

    print(f"Loading master dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    initial_rows = len(df)
    print(f"Initial row count: {initial_rows}")
    
    # --- Data Cleaning Logic ---
    # Ensure the correct column name is used
    if 'Vehicle_Class' in df.columns and 'Vehicle_Category' not in df.columns:
        df.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)

    # Remove rows with 'Unknown' state
    df_cleaned = df[df['State'] != 'Unknown'].copy()
    
    # Remove rows with 'Unknown' or NaN vehicle category
    if 'Vehicle_Category' in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned['Vehicle_Category'] != 'Unknown']
        df_cleaned = df_cleaned.dropna(subset=['Vehicle_Category'])
    
    cleaned_rows = len(df_cleaned)
    rows_removed = initial_rows - cleaned_rows
    
    if rows_removed > 0:
        print(f"Found and removed {rows_removed} rows with bad data.")
        # Save the cleaned dataframe back to the same file
        df_cleaned.to_csv(DATASET_PATH, index=False)
        print(f"✅ Master dataset has been cleaned and saved. New row count: {cleaned_rows}")
    else:
        print("✅ No bad data found. Your dataset is already clean!")

if __name__ == "__main__":
    clean_dataset()