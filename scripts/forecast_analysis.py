#!/usr/bin/env python3
"""
Advanced EV Model Forecasting Analysis
Analyzes the forecasting performance and creates visualization plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ForecastAnalyzer:
    """Analyzes the advanced EV model's forecasting performance."""
    
    def __init__(self, models_dir="models", output_dir="output"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model paths
        self.advanced_model_path = self.models_dir / "advanced_ev_model.pkl"
        self.predictions_path = self.output_dir / "advanced_model_predictions.csv"
        
    def load_advanced_model(self):
        """Load the advanced model and its components."""
        try:
            with open(self.advanced_model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            print("✅ Advanced model loaded successfully")
            print(f"   Primary model: {type(model_data['primary_model']).__name__}")
            print(f"   Features: {len(model_data['feature_names'])}")
            print(f"   Test R²: {model_data.get('test_r2', 'N/A')}")
            
            return model_data
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def load_predictions(self):
        """Load the predictions data."""
        try:
            if not self.predictions_path.exists():
                print("❌ Predictions file not found. Run advanced model first.")
                return None
            
            df = pd.read_csv(self.predictions_path)
            print(f"✅ Loaded predictions: {len(df)} records")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
            
            return df
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            return None
    
    def analyze_predictions(self, df):
        """Analyze the prediction quality and identify issues."""
        print("\n🔍 PREDICTION ANALYSIS")
        print("=" * 50)
        
        # Check for constant predictions
        unique_predictions = df['Advanced_Predictions'].nunique()
        print(f"Unique prediction values: {unique_predictions}")
        
        if unique_predictions == 1:
            print("⚠️  WARNING: All predictions are identical!")
            print(f"   Constant value: {df['Advanced_Predictions'].iloc[0]}")
            print("   This indicates a feature preparation issue.")
        
        # Basic statistics
        print(f"\n📊 Prediction Statistics:")
        print(f"   Min: {df['Advanced_Predictions'].min():.2f}")
        print(f"   Max: {df['Advanced_Predictions'].max():.2f}")
        print(f"   Mean: {df['Advanced_Predictions'].mean():.2f}")
        print(f"   Std: {df['Advanced_Predictions'].std():.2f}")
        
        # Actual vs Predicted comparison
        print(f"\n📈 Actual vs Predicted Comparison:")
        print(f"   Actual range: {df['EV_Sales_Quantity'].min()} to {df['EV_Sales_Quantity'].max()}")
        print(f"   Predicted range: {df['Advanced_Predictions'].min():.2f} to {df['Advanced_Predictions'].max():.2f}")
        
        # Calculate metrics
        actual = df['EV_Sales_Quantity'].values
        predicted = df['Advanced_Predictions'].values
        
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
        
        print(f"\n📊 Performance Metrics:")
        print(f"   MAE: {mae:.2f}")
        print(f"   MSE: {mse:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   R²: {r2:.4f}")
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'unique_predictions': unique_predictions
        }
    
    def create_forecasting_plots(self, df, metrics):
        """Create comprehensive forecasting visualization plots."""
        print("\n🎨 Creating forecasting plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced EV Model Forecasting Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Actual vs Predicted Scatter
        ax1 = axes[0, 0]
        ax1.scatter(df['EV_Sales_Quantity'], df['Advanced_Predictions'], alpha=0.6, s=20)
        ax1.plot([0, df['EV_Sales_Quantity'].max()], [0, df['EV_Sales_Quantity'].max()], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual EV Sales Quantity')
        ax1.set_ylabel('Predicted EV Sales Quantity')
        ax1.set_title('Actual vs Predicted Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add R² text
        ax1.text(0.05, 0.95, f'R² = {metrics["r2"]:.4f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 2: Time Series of Actual vs Predicted
        ax2 = axes[0, 1]
        # Sample data for better visualization (take every 10th record)
        sample_df = df.iloc[::10].copy()
        sample_df['Date'] = pd.to_datetime(sample_df['Date'])
        sample_df = sample_df.sort_values('Date')
        
        ax2.plot(sample_df['Date'], sample_df['EV_Sales_Quantity'], 'b-', label='Actual', alpha=0.7, linewidth=2)
        ax2.plot(sample_df['Date'], sample_df['Advanced_Predictions'], 'r--', label='Predicted', alpha=0.7, linewidth=2)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('EV Sales Quantity')
        ax2.set_title('Time Series: Actual vs Predicted')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 3: Residuals Plot
        ax3 = axes[1, 0]
        residuals = df['EV_Sales_Quantity'] - df['Advanced_Predictions']
        ax3.scatter(df['Advanced_Predictions'], residuals, alpha=0.6, s=20)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Predicted Values')
        ax3.set_ylabel('Residuals (Actual - Predicted)')
        ax3.set_title('Residuals Plot')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Distribution Comparison
        ax4 = axes[1, 1]
        ax4.hist(df['EV_Sales_Quantity'], bins=30, alpha=0.7, label='Actual', color='blue', density=True)
        ax4.hist(df['Advanced_Predictions'], bins=30, alpha=0.7, label='Predicted', color='red', density=True)
        ax4.set_xlabel('EV Sales Quantity')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution: Actual vs Predicted')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.output_dir / "forecasting_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {plot_path}")
        
        plt.show()
        
        return plot_path
    
    def create_category_analysis_plots(self, df):
        """Create category-specific analysis plots."""
        print("\n📊 Creating category analysis plots...")
        
        # Group by vehicle category
        category_stats = df.groupby('Vehicle_Class').agg({
            'EV_Sales_Quantity': ['mean', 'std', 'count'],
            'Advanced_Predictions': ['mean', 'std']
        }).round(2)
        
        print("\n📈 Category-wise Statistics:")
        print(category_stats)
        
        # Create category comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Category-wise Forecasting Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Category-wise Actual vs Predicted
        ax1 = axes[0, 0]
        category_means = df.groupby('Vehicle_Class').agg({
            'EV_Sales_Quantity': 'mean',
            'Advanced_Predictions': 'mean'
        })
        
        x = np.arange(len(category_means))
        width = 0.35
        
        ax1.bar(x - width/2, category_means['EV_Sales_Quantity'], width, label='Actual', alpha=0.7)
        ax1.bar(x + width/2, category_means['Advanced_Predictions'], width, label='Predicted', alpha=0.7)
        ax1.set_xlabel('Vehicle Category')
        ax1.set_ylabel('Average EV Sales Quantity')
        ax1.set_title('Category-wise: Actual vs Predicted Averages')
        ax1.set_xticks(x)
        ax1.set_xticklabels(category_means.index, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Category-wise MAE
        ax2 = axes[0, 1]
        category_mae = df.groupby('Vehicle_Class').apply(
            lambda x: np.mean(np.abs(x['EV_Sales_Quantity'] - x['Advanced_Predictions']))
        ).sort_values(ascending=False)
        
        ax2.bar(range(len(category_mae)), category_mae.values, alpha=0.7, color='orange')
        ax2.set_xlabel('Vehicle Category')
        ax2.set_ylabel('Mean Absolute Error (MAE)')
        ax2.set_title('Category-wise MAE')
        ax2.set_xticks(range(len(category_mae)))
        ax2.set_xticklabels(category_mae.index, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Category-wise R²
        ax3 = axes[1, 0]
        def calculate_r2(group):
            actual = group['EV_Sales_Quantity']
            predicted = group['Advanced_Predictions']
            if len(actual) < 2:
                return np.nan
            return 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
        
        category_r2 = df.groupby('Vehicle_Class').apply(calculate_r2).sort_values(ascending=False)
        
        colors = ['green' if x > 0 else 'red' for x in category_r2.values]
        ax3.bar(range(len(category_r2)), category_r2.values, alpha=0.7, color=colors)
        ax3.set_xlabel('Vehicle Category')
        ax3.set_ylabel('R² Score')
        ax3.set_title('Category-wise R²')
        ax3.set_xticks(range(len(category_r2)))
        ax3.set_xticklabels(category_r2.index, rotation=45)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Prediction Range Analysis
        ax4 = axes[1, 1]
        category_ranges = df.groupby('Vehicle_Class').agg({
            'EV_Sales_Quantity': lambda x: x.max() - x.min(),
            'Advanced_Predictions': lambda x: x.max() - x.min()
        })
        
        x = np.arange(len(category_ranges))
        width = 0.35
        
        ax4.bar(x - width/2, category_ranges['EV_Sales_Quantity'], width, label='Actual Range', alpha=0.7)
        ax4.bar(x + width/2, category_ranges['Advanced_Predictions'], width, label='Predicted Range', alpha=0.7)
        ax4.set_xlabel('Vehicle Category')
        ax4.set_ylabel('Range (Max - Min)')
        ax4.set_title('Category-wise: Actual vs Predicted Ranges')
        ax4.set_xticks(x)
        ax4.set_xticklabels(category_ranges.index, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.output_dir / "category_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Category analysis plot saved to: {plot_path}")
        
        plt.show()
        
        return plot_path
    
    def generate_forecast_report(self, df, metrics):
        """Generate a comprehensive forecasting report."""
        print("\n📋 GENERATING FORECAST REPORT")
        print("=" * 50)
        
        report = []
        report.append("ADVANCED EV MODEL FORECASTING REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Records: {len(df)}")
        report.append(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
        report.append("")
        
        # Performance Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 30)
        report.append(f"Mean Absolute Error (MAE): {metrics['mae']:.2f}")
        report.append(f"Mean Squared Error (MSE): {metrics['mse']:.2f}")
        report.append(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f}")
        report.append(f"R-squared (R²): {metrics['r2']:.4f}")
        report.append("")
        
        # Data Quality Issues
        if metrics['unique_predictions'] == 1:
            report.append("⚠️  DATA QUALITY ISSUES DETECTED")
            report.append("-" * 30)
            report.append("All predictions are identical, indicating:")
            report.append("1. Feature preparation problems")
            report.append("2. Model not receiving proper feature variation")
            report.append("3. Potential scaling or encoding issues")
            report.append("")
        
        # Category Performance
        report.append("CATEGORY PERFORMANCE")
        report.append("-" * 30)
        for category in df['Vehicle_Class'].unique():
            cat_data = df[df['Vehicle_Class'] == category]
            cat_mae = np.mean(np.abs(cat_data['EV_Sales_Quantity'] - cat_data['Advanced_Predictions']))
            cat_count = len(cat_data)
            report.append(f"{category}: MAE={cat_mae:.2f}, Records={cat_count}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        if metrics['r2'] < 0:
            report.append("1. Model performance is poor (negative R²)")
            report.append("2. Check feature engineering pipeline")
            report.append("3. Verify data preprocessing steps")
            report.append("4. Consider retraining with better features")
        elif metrics['r2'] < 0.5:
            report.append("1. Model performance is below average")
            report.append("2. Review feature selection")
            report.append("3. Check for data quality issues")
        else:
            report.append("1. Model performance is good")
            report.append("2. Consider fine-tuning for better results")
        
        # Save report
        report_path = self.output_dir / "forecast_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Report saved to: {report_path}")
        
        # Print report to console
        print('\n'.join(report))
        
        return report_path
    
    def run_analysis(self):
        """Run the complete forecasting analysis."""
        print("🚀 ADVANCED EV MODEL FORECASTING ANALYSIS")
        print("=" * 60)
        
        # Load model
        model_data = self.load_advanced_model()
        if model_data is None:
            return False
        
        # Load predictions
        df = self.load_predictions()
        if df is None:
            return False
        
        # Analyze predictions
        metrics = self.analyze_predictions(df)
        
        # Create plots
        try:
            self.create_forecasting_plots(df, metrics)
            self.create_category_analysis_plots(df)
        except Exception as e:
            print(f"⚠️  Plot creation failed: {e}")
        
        # Generate report
        self.generate_forecast_report(df, metrics)
        
        print("\n🎉 Analysis completed successfully!")
        return True

def main():
    """Main function to run the forecasting analysis."""
    analyzer = ForecastAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
