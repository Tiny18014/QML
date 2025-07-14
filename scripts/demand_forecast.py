import pandas as pd
from river import linear_model, preprocessing, compose, metrics

# Load preprocessed data
df = pd.read_csv("data/EV_Dataset.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date', 'EV_Sales_Quantity'], inplace=True)
df['EV_Sales_Quantity'] = df['EV_Sales_Quantity'].astype(int)
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df = df[df['year'] >= 2020]
df_grouped = df.groupby(['Date', 'State']).agg({
    'EV_Sales_Quantity': 'sum',
    'day': 'first',
    'month': 'first',
    'year': 'first'
}).reset_index()

# Features and target
features = ['day', 'month', 'year', 'State']
target = 'EV_Sales_Quantity'

# River model
model = compose.Pipeline(
    preprocessing.OneHotEncoder(),              # encode state
    preprocessing.StandardScaler(),             # scale features
    linear_model.LinearRegression()             # online linear regression
)

# Metric to track
metric = metrics.MAE()

# Simulate online learning
for _, row in df_grouped.iterrows():
    x = {
        'day': row['day'],
        'month': row['month'],
        'year': row['year'],
        'State': row['State']
    }
    y = row['EV_Sales_Quantity']
    y_pred = model.predict_one(x) or 0
    model.learn_one(x, y)
    metric.update(y, y_pred)

# Final error
print(f"Mean Absolute Error: {metric.get():.2f}")
