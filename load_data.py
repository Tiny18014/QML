import pandas as pd

# Load dataset
df = pd.read_csv('data/EV_Dataset.csv')  # Adjust path if needed

# Preview the first 5 rows
print("First 5 rows:\n", df.head())

# Show column names and data types
print("\nColumn info:\n", df.dtypes)

# Optional: Show unique states if available
if 'state' in df.columns:
    print("\nStates:\n", df['state'].unique())

