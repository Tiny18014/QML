#!/usr/bin/env python3
"""
Incremental Training for Advanced EV Model
Allows continuous model updates with new data without full retraining.
"""

import pandas as pd
import numpy as np
import pickle
import os
import logging
from pathlib import Path
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('output/incremental_training.log', mode='a', encoding='utf-8')
    ]
)

class IncrementalEVTrainer:
    """
    Incremental trainer for the advanced EV demand forecasting model.
    Allows continuous model updates with new data.
    """
    
    def __init__(self, models_dir="models", output_dir="output"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model paths
        self.advanced_model_path = self.models_dir / "advanced_ev_model.pkl"
        self.incremental_model_path = self.models_dir / "incremental_ev_model.pkl"
        self.backup_model_path = self.models_dir / "backup_advanced_model.pkl"
        
        # Training history
        self.training_history = []
        
    def load_existing_model(self):
        """Load the existing advanced model for incremental training."""
        try:
            if self.advanced_model_path.exists():
                with open(self.advanced_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                logging.info("Existing advanced model loaded successfully")
                return model_data
            else:
                logging.error("❌ No existing advanced model found")
                return None
        except Exception as e:
            logging.error(f"❌ Error loading existing model: {e}")
            return None
    
    def create_advanced_features(self, df):
        """Create advanced features for the model (same as training script)."""
        df = df.copy()
        
        # Check if Date column exists, if not try to create it
        if 'Date' not in df.columns:
            if 'Month_Name' in df.columns and 'Year' in df.columns:
                # Create Date from Month_Name and Year
                month_to_date = {
                    'JAN': '01-15', 'FEB': '02-15', 'MAR': '03-15',
                    'APR': '04-15', 'MAY': '05-15', 'JUN': '06-15',
                    'JUL': '07-15', 'AUG': '08-15', 'SEP': '09-15',
                    'OCT': '10-15', 'NOV': '11-15', 'DEC': '12-15'
                }
                df['Date'] = df['Year'].astype(str) + '-' + df['Month_Name'].map(month_to_date)
                logging.info("Created Date column from Month_Name and Year")
            else:
                # Create a default date if no date information available
                df['Date'] = pd.Timestamp('2025-01-01')
                logging.info("Created default Date column")
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Handle column name differences
        if 'Vehicle_Class' in df.columns and 'Vehicle_Category' not in df.columns:
            df['Vehicle_Category'] = df['Vehicle_Class']
            logging.info("Mapped Vehicle_Class to Vehicle_Category")
        
        df = df.sort_values(['State', 'Vehicle_Category', 'Date'])
        
        # Basic temporal features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['quarter'] = df['Date'].dt.quarter
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'lag_{lag}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].shift(lag).fillna(0)
        
        # Rolling window features
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1).mean().values
            df[f'rolling_std_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1).std().values
            df[f'rolling_min_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1).min().values
            df[f'rolling_max_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1).max().values
            df[f'rolling_median_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1).median().values
        
        # Exponential Moving Averages
        for window in [7, 14, 30]:
            df[f'ema_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].ewm(span=window).mean().values
        
        # Seasonal decomposition (simplified)
        for window in [7, 30]:
            df[f'seasonal_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1).mean().values
            df[f'trend_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1).mean().values
            df[f'volatility_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1).std().values
        
        # Cross-category features
        df['state_daily_mean'] = df.groupby(['State', 'Date'])['EV_Sales_Quantity'].transform('mean')
        df['state_overall_mean'] = df.groupby('State')['EV_Sales_Quantity'].transform('mean')
        df['category_overall_mean'] = df.groupby('Vehicle_Category')['EV_Sales_Quantity'].transform('mean')
        
        # Interaction features
        df['state_category_interaction'] = df['state_daily_mean'] * df['category_overall_mean']
        df['sales_ratio_to_state_mean'] = df['EV_Sales_Quantity'] / (df['state_overall_mean'] + 1e-8)
        df['sales_ratio_to_category_mean'] = df['EV_Sales_Quantity'] / (df['category_overall_mean'] + 1e-8)
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        return df
    
    def prepare_features_for_training(self, df, feature_names, scaler):
        """Prepare features for training using existing scaler."""
        # Handle column name differences
        if 'Vehicle_Class' in df.columns and 'Vehicle_Category' not in df.columns:
            df['Vehicle_Category'] = df['Vehicle_Class']
        
        # Convert categorical variables to strings first to avoid category conflicts
        df['State'] = df['State'].astype(str)
        df['Vehicle_Category'] = df['Vehicle_Category'].astype(str)
        
        # Create state and category codes
        df['state_code'] = pd.Categorical(df['State']).codes
        df['category_code'] = pd.Categorical(df['Vehicle_Category']).codes
        
        # Select only the features that the model expects
        available_features = [col for col in feature_names if col in df.columns]
        missing_features = [col for col in feature_names if col not in df.columns]
        
        if missing_features:
            logging.warning(f"Missing features: {missing_features}")
            # Fill missing features with 0
            for feature in missing_features:
                df[feature] = 0
        
        # Create feature matrix
        X = df[feature_names].copy()
        
        # Ensure all features are numerical
        for col in X.columns:
            if X[col].dtype == 'object':
                # Convert string columns to numerical using category codes
                X[col] = pd.Categorical(X[col]).codes
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        # Scale features using existing scaler
        X_scaled = scaler.transform(X)
        
        return X_scaled
    
    def incremental_train(self, new_data_path, validation_split=0.2, learning_rate=0.01, n_estimators=100):
        """
        Perform incremental training on the existing model with new data.
        
        Args:
            new_data_path: Path to new data file (CSV or Excel)
            validation_split: Fraction of data to use for validation
            learning_rate: Learning rate for incremental updates
            n_estimators: Number of additional estimators to add
        """
        try:
            logging.info("Starting incremental training...")
            
            # Step 1: Load existing model
            existing_model_data = self.load_existing_model()
            if existing_model_data is None:
                logging.error("❌ Cannot proceed without existing model")
                return False
            
            # Step 2: Load and preprocess new data
            logging.info(f"Loading new data from: {new_data_path}")
            
            # Check if this is a raw Excel file that needs preprocessing
            if str(new_data_path).endswith('.xlsx'):
                # Try to use the existing data push pipeline preprocessing
                try:
                    from run_pipeline import process_single_state_file
                    new_df = process_single_state_file(Path(new_data_path))
                    if new_df is None:
                        # Fallback to direct loading
                        new_df = pd.read_excel(new_data_path)
                        logging.info("Using direct Excel loading (raw data)")
                    else:
                        logging.info("Successfully preprocessed Excel data using data push pipeline")
                except ImportError:
                    # Fallback to direct loading
                    new_df = pd.read_excel(new_data_path)
                    logging.info("Using direct Excel loading (raw data)")
            else:
                new_df = pd.read_csv(new_data_path)
            
            logging.info(f"Loaded {len(new_df)} new records")
            
            # Step 3: Create features for new data
            logging.info("Creating advanced features for new data...")
            new_df_features = self.create_advanced_features(new_df)
            
            # Step 4: Prepare features using existing scaler
            feature_names = existing_model_data['feature_names']
            scaler = existing_model_data['scaler']
            
            X_new = self.prepare_features_for_training(new_df_features, feature_names, scaler)
            y_new = new_df_features['EV_Sales_Quantity'].values
            
            # Step 5: Split new data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_new, y_new, test_size=validation_split, random_state=42
            )
            
            # Step 6: Get existing model
            existing_model = existing_model_data['primary_model']
            
            # Step 7: Perform incremental training
            logging.info("Performing incremental training...")
            
            # Create a new model with the same parameters but continue training
            existing_params = existing_model.get_params()
            # Remove conflicting parameters
            if 'learning_rate' in existing_params:
                del existing_params['learning_rate']
            if 'n_estimators' in existing_params:
                del existing_params['n_estimators']
            if 'random_state' in existing_params:
                del existing_params['random_state']
            
            new_model = lgb.LGBMRegressor(
                **existing_params,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                random_state=42
            )
            
            # Train on new data
            new_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            # Step 8: Evaluate performance
            y_pred_train = new_model.predict(X_train)
            y_pred_val = new_model.predict(X_val)
            
            train_mae = mean_absolute_error(y_train, y_pred_train)
            train_r2 = r2_score(y_train, y_pred_train)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_r2 = r2_score(y_val, y_pred_val)
            
            # Step 9: Create ensemble prediction
            logging.info("Creating ensemble prediction...")
            
            # Combine predictions from existing and new model
            existing_pred = existing_model.predict(X_val)
            new_pred = new_model.predict(X_val)
            
            # Simple ensemble (average)
            ensemble_pred = (existing_pred + new_pred) / 2
            
            ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
            ensemble_r2 = r2_score(y_val, ensemble_pred)
            
            # Step 10: Save results and update model
            training_result = {
                'timestamp': datetime.now().isoformat(),
                'new_records': len(new_df),
                'train_mae': train_mae,
                'train_r2': train_r2,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'ensemble_mae': ensemble_mae,
                'ensemble_r2': ensemble_r2,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators
            }
            
            self.training_history.append(training_result)
            
            # Step 11: Create updated model bundle
            updated_model_data = existing_model_data.copy()
            updated_model_data['primary_model'] = new_model
            updated_model_data['ensemble_models'] = {
                'original_model': existing_model,
                'new_model': new_model,
                'ensemble_method': 'average'
            }
            updated_model_data['incremental_training_history'] = self.training_history
            updated_model_data['last_updated'] = datetime.now().isoformat()
            
            # Step 12: Save updated model
            logging.info("Saving updated model...")
            
            # Create backup of existing model
            if self.advanced_model_path.exists():
                import shutil
                shutil.copy2(self.advanced_model_path, self.backup_model_path)
                logging.info("Created backup of existing model")
            
            # Save updated model
            with open(self.incremental_model_path, 'wb') as f:
                pickle.dump(updated_model_data, f)
            
            # Also update the main advanced model
            with open(self.advanced_model_path, 'wb') as f:
                pickle.dump(updated_model_data, f)
            
            # Step 13: Save training results
            results_df = pd.DataFrame(self.training_history)
            results_path = self.output_dir / "incremental_training_results.csv"
            results_df.to_csv(results_path, index=False)
            
            # Step 14: Print summary
            self._print_training_summary(training_result)
            
            logging.info("Incremental training completed successfully!")
            return True
            
        except Exception as e:
            logging.error(f"❌ Incremental training failed: {e}")
            return False
    
    def _print_training_summary(self, result):
        """Print a summary of the incremental training results."""
        print("\n" + "="*60)
        print("🔄 INCREMENTAL TRAINING RESULTS")
        print("="*60)
        print(f"📊 New Records Processed: {result['new_records']}")
        print(f"🕒 Training Timestamp: {result['timestamp']}")
        print(f"⚙️  Learning Rate: {result['learning_rate']}")
        print(f"🌳 New Estimators: {result['n_estimators']}")
        print("\n📈 Performance Metrics:")
        print(f"   Training MAE: {result['train_mae']:.2f}")
        print(f"   Training R²: {result['train_r2']:.4f}")
        print(f"   Validation MAE: {result['val_mae']:.2f}")
        print(f"   Validation R²: {result['val_r2']:.4f}")
        print(f"   Ensemble MAE: {result['ensemble_mae']:.2f}")
        print(f"   Ensemble R²: {result['ensemble_r2']:.4f}")
        print("="*60)
    
    def get_training_history(self):
        """Get the complete training history."""
        return pd.DataFrame(self.training_history)
    
    def compare_models(self):
        """Compare original vs. updated model performance."""
        try:
            if not self.backup_model_path.exists():
                logging.warning("⚠️ No backup model found for comparison")
                return None
            
            # Load both models
            with open(self.backup_model_path, 'rb') as f:
                original_model = pickle.load(f)
            
            with open(self.advanced_model_path, 'rb') as f:
                updated_model = pickle.load(f)
            
            print("\n" + "="*60)
            print("🔍 MODEL COMPARISON")
            print("="*60)
            print(f"📅 Original Model: {original_model.get('last_updated', 'Unknown')}")
            print(f"📅 Updated Model: {updated_model.get('last_updated', 'Unknown')}")
            print(f"🔄 Incremental Updates: {len(updated_model.get('incremental_training_history', []))}")
            print("="*60)
            
            return {
                'original': original_model,
                'updated': updated_model
            }
            
        except Exception as e:
            logging.error(f"❌ Model comparison failed: {e}")
            return None

def main():
    """Main function to run incremental training."""
    print("🚗 EV Model Incremental Trainer")
    print("=" * 40)
    
    # Initialize trainer
    trainer = IncrementalEVTrainer()
    
    # Check if we have an existing model
    if not trainer.advanced_model_path.exists():
        print("❌ No existing advanced model found!")
        print("💡 Please train the advanced model first using advanced_model_trainer.py")
        return
    
    # Get new data path
    print("\n📁 Enter the path to your new data file:")
    print("   - Can be CSV or Excel file")
    print("   - Should have similar structure to existing data")
    print("   - Will be used to incrementally update the model")
    
    new_data_path = input("\n📂 New data file path: ").strip()
    
    if not new_data_path or not Path(new_data_path).exists():
        print("❌ Invalid file path!")
        return
    
    # Get training parameters
    print("\n⚙️  Training Parameters:")
    learning_rate = float(input("   Learning Rate (default 0.01): ") or "0.01")
    n_estimators = int(input("   Number of Estimators (default 100): ") or "100")
    
    # Run incremental training
    success = trainer.incremental_train(
        new_data_path=new_data_path,
        learning_rate=learning_rate,
        n_estimators=n_estimators
    )
    
    if success:
        print("\n🎉 Incremental training completed successfully!")
        print("💡 Your model has been updated with new knowledge!")
        
        # Show comparison
        trainer.compare_models()
        
        # Show training history
        history = trainer.get_training_history()
        if not history.empty:
            print(f"\n📊 Training History ({len(history)} updates):")
            print(history[['timestamp', 'new_records', 'ensemble_r2', 'ensemble_mae']].tail())
    else:
        print("\n❌ Incremental training failed!")
        print("💡 Check the logs for details")

if __name__ == "__main__":
    main()
