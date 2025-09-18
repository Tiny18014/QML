# scripts/analyze_categories.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATASET_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"

def analyze():
    """Loads the dataset and creates plots to analyze sales by vehicle category."""
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, parse_dates=['Date'])
    
    # Categories to investigate
    categories = ['2-Wheelers', '3-Wheelers', '4-Wheelers', 'Bus']
    
    # 1. Plot Sales Over Time for Each Category
    print("Generating time series plot...")
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=df, x='Date', y='EV_Sales_Quantity', hue='Vehicle_Category', errorbar=None)
    plt.title('EV Sales Quantity Over Time by Vehicle Category')
    plt.ylabel('EV Sales Quantity')
    plt.xlabel('Date')
    plt.legend(title='Vehicle Category')
    plt.grid(True)
    plt.savefig('output/category_sales_timeseries.png')
    print("✅ Time series plot saved to output/category_sales_timeseries.png")

    # 2. Plot Boxplots to See Distribution and Outliers
    print("Generating distribution boxplot...")
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df, x='Vehicle_Category', y='EV_Sales_Quantity', order=categories)
    plt.title('Distribution of EV Sales Quantity by Vehicle Category')
    plt.ylabel('EV Sales Quantity (Log Scale)')
    plt.xlabel('Vehicle Category')
    plt.yscale('log') # Use a log scale to better visualize wide-ranging values
    plt.grid(True)
    plt.savefig('output/category_sales_distribution.png')
    print("✅ Distribution plot saved to output/category_sales_distribution.png")

if __name__ == "__main__":
    analyze()