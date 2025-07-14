import pandas as pd

# Load the dataset
df = pd.read_csv('data/EV_Dataset.csv')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows with invalid dates or missing values
df.dropna(subset=['Date', 'EV_Sales_Quantity'], inplace=True)

# Convert sales to integer
df['EV_Sales_Quantity'] = df['EV_Sales_Quantity'].astype(int)

# Extract time-based features
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

# (Optional) Filter for recent years to keep it manageable
df = df[df['year'] >= 2020]

# Group by Date + State to sum sales (if there are multiple entries)
df_grouped = df.groupby(['Date', 'State']).agg({
    'EV_Sales_Quantity': 'sum',
    'day': 'first',
    'month': 'first',
    'year': 'first'
}).reset_index()

# Preview
print(df_grouped.head())

