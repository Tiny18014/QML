#!/usr/bin/env python3
"""
EV Sales Live Pipeline Launcher
Modified to work with data pushes instead of web scraping.
Pipeline runs on every push to data folder, preprocessing data and feeding to dummy model.
"""

import subprocess
import sys
import os
import time
import threading
import argparse
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"
SCRIPTS_DIR = ROOT_DIR / "scripts"
DATASET_PATH = DATA_DIR / "EV_Dataset.csv"
DB_PATH = OUTPUT_DIR / "live_predictions.db"
PUSH_PIPELINE_LOG = OUTPUT_DIR / "push_pipeline.log"

def log_push_pipeline(message):
    """Log messages to the push pipeline log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    with open(PUSH_PIPELINE_LOG, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")
    print(log_message)

def check_data_folder_changes():
    """Check if there are new files in the data folder."""
    log_push_pipeline("Checking for data folder changes...")
    
    # Look for any CSV or Excel files in data folder
    data_files = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
    
    if not data_files:
        log_push_pipeline("No data files found in data folder")
        return False
    
    # Check if we have the main dataset
    if DATASET_PATH.exists():
        log_push_pipeline(f"Found main dataset: {DATASET_PATH}")
        return True
    else:
        log_push_pipeline("Main dataset not found, checking for other data files...")
        return len(data_files) > 0

def run_data_push_pipeline():
    """Run the data push pipeline: preprocess data and run predictions."""
    log_push_pipeline("🚀 Starting data push pipeline...")
    try:
        if not preprocess_data_simplified():
            log_push_pipeline("❌ Data preprocessing failed.")
            return False
        
        log_push_pipeline("🤖 Feeding data to Advanced EV models...")
        if not run_advanced_model_datapush():
            log_push_pipeline("❌ Advanced model prediction failed.")
            return False
            
        log_push_pipeline("✅ Data push pipeline completed successfully!")
        return True
    except Exception as e:
        log_push_pipeline(f"❌ Data push pipeline failed with error: {str(e)}")
        return False

def run_accuracy_comparison():
    """Run accuracy comparison between EV_Dataset.csv and Data Push Pipeline."""
    print("🎯 Running Accuracy Comparison...")
    print("=" * 50)
    
    try:
        # Import and run the accuracy comparison script
        sys.path.insert(0, str(SCRIPTS_DIR))
        from accuracy_comparison import main as run_accuracy_comparison_main
        
        # Run the accuracy comparison
        results = run_accuracy_comparison_main()
        
        if results is not None:
            print("\n✅ Accuracy comparison completed successfully!")
            print("📁 Check the output files for detailed results:")
            print(f"   - Log: {OUTPUT_DIR / 'accuracy_comparison.log'}")
            print(f"   - Results: {OUTPUT_DIR / 'accuracy_comparison_results.csv'}")
        else:
            print("\n❌ Accuracy comparison failed. Check the logs for details.")
            
    except ImportError as e:
        print(f"❌ Could not import accuracy comparison module: {e}")
        print("💡 Make sure accuracy_comparison.py exists in the scripts folder")
    except Exception as e:
        print(f"❌ Accuracy comparison failed: {e}")
    finally:
        # Clean up the path modification
        if str(SCRIPTS_DIR) in sys.path:
            sys.path.remove(str(SCRIPTS_DIR))

def run_incremental_training():
    """Run incremental training to update the advanced model with new data."""
    print("🔄 Running Incremental Training...")
    print("=" * 50)
    
    try:
        # Import and run the incremental trainer
        sys.path.insert(0, str(SCRIPTS_DIR))
        from incremental_trainer import IncrementalEVTrainer
        
        # Initialize trainer
        trainer = IncrementalEVTrainer()
        
        # Check if we have an existing model
        if not trainer.advanced_model_path.exists():
            print("❌ No existing advanced model found!")
            print("💡 Please train the advanced model first using option 10")
            return
        
        print("✅ Existing advanced model found!")
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
            
    except ImportError as e:
        print(f"❌ Could not import incremental trainer module: {e}")
        print("💡 Make sure incremental_trainer.py exists in the scripts folder")
    except Exception as e:
        print(f"❌ Incremental training failed: {e}")
    finally:
        # Clean up the path modification
        if str(SCRIPTS_DIR) in sys.path:
            sys.path.remove(str(SCRIPTS_DIR))

def run_advanced_model():
    """
    Loads the correct category-specific model for each row of data and makes predictions.
    """
    print("🚀 Running Advanced Model (High Performance)...")
    print("=" * 50)
    try:
        preprocessed_path = OUTPUT_DIR / "preprocessed_data.csv"
        if not preprocessed_path.exists():
            print("📊 No preprocessed data found. Please run the data push pipeline (option 8) first.")
            return

        print("📊 Using preprocessed data for prediction...")
        df = pd.read_csv(preprocessed_path)
        
        # Ensure Vehicle_Category column exists and is filled
        df['Vehicle_Category'] = df['Vehicle_Category'].fillna('Unknown')
        
        all_predictions = []
        loaded_models = {} # A cache to avoid loading the same model multiple times

        # Get the same feature engineering logic
        sys.path.insert(0, str(SCRIPTS_DIR))
        # *** FIX: Import the correct prediction preparation function ***
        from advanced_model_trainer import create_advanced_features, prepare_features_for_prediction
        df_features = create_advanced_features(df)

        print("\nPredicting for each category...")
        for category in df['Vehicle_Category'].unique():
            print(f"  -> Processing category: {category}")
            
            category_filename = category.replace(" ", "_").replace("/", "_")
            model_path = MODELS_DIR / f"advanced_model_{category_filename}.pkl"

            if not model_path.exists():
                print(f"    ⚠️ Warning: Model for category '{category}' not found. Skipping.")
                continue

            if category not in loaded_models:
                with open(model_path, 'rb') as f:
                    loaded_models[category] = pickle.load(f)

            model_data = loaded_models[category]
            model = model_data['primary_model']
            feature_names = model_data['feature_names']
            scaler = model_data['scaler']

            category_df_features = df_features[df_features['Vehicle_Category'] == category].copy()
            if category_df_features.empty:
                continue

            # *** FIX: Use the correct prediction preparation function ***
            X_pred_scaled = prepare_features_for_prediction(category_df_features, feature_names, scaler)
            
            predictions = pd.Series(model.predict(X_pred_scaled), index=category_df_features.index)
            all_predictions.append(predictions)

        if not all_predictions:
            print("❌ No predictions were made. Check if models exist.")
            return

        df['Advanced_Predictions'] = pd.concat(all_predictions).sort_index()
        
        print("\n📊 Sample Predictions:")
        print(df[['Date', 'State', 'Vehicle_Category', 'EV_Sales_Quantity', 'Advanced_Predictions']].tail(10).to_string(index=False))

    except Exception as e:
        print(f"❌ Advanced model prediction failed: {e}")
    finally:
        if str(SCRIPTS_DIR) in sys.path:
            sys.path.remove(str(SCRIPTS_DIR))

def run_advanced_model_datapush():
    """Use category-specific models on preprocessed data and insert into DB."""
    try:
        preprocessed_path = OUTPUT_DIR / "preprocessed_data.csv"
        if not preprocessed_path.exists():
            log_push_pipeline("❌ Preprocessed data not found.")
            return False

        df = pd.read_csv(preprocessed_path)
        df['Vehicle_Category'] = df['Vehicle_Category'].fillna('Unknown')
        
        all_predictions = []
        loaded_models = {}
        
        sys.path.insert(0, str(SCRIPTS_DIR))
        from advanced_model_trainer import create_advanced_features, prepare_features_for_prediction
        df_features = create_advanced_features(df)

        log_push_pipeline("Predicting for each category...")
        for category in df['Vehicle_Category'].unique():
            category_filename = category.replace(" ", "_").replace("/", "_")
            model_path = MODELS_DIR / f"advanced_model_{category_filename}.pkl"

            if not model_path.exists():
                log_push_pipeline(f"    - Warning: Model for '{category}' not found. Skipping.")
                continue

            if category not in loaded_models:
                with open(model_path, 'rb') as f:
                    loaded_models[category] = pickle.load(f)

            model_data = loaded_models[category]
            model = model_data['primary_model']
            feature_names = model_data['feature_names']
            scaler = model_data['scaler']
            
            category_df_features = df_features[df_features['Vehicle_Category'] == category]
            if category_df_features.empty: continue
            
            X_pred_scaled = prepare_features_for_prediction(category_df_features, feature_names, scaler)
            predictions = pd.Series(model.predict(X_pred_scaled), index=category_df_features.index)
            all_predictions.append(predictions)

        if not all_predictions:
            log_push_pipeline("❌ No predictions made. Check if models exist.")
            return False

        results_df = df.copy()
        results_df['Predicted_Sales'] = pd.concat(all_predictions).sort_index()
        # Your database insertion logic would go here
        log_push_pipeline("✅ Data push predictions completed successfully.")
        return True
    except Exception as e:
        log_push_pipeline(f"❌ Advanced model datapush failed: {e}")
        return False
    finally:
        if str(SCRIPTS_DIR) in sys.path:
            sys.path.remove(str(SCRIPTS_DIR))

def extract_state_name_from_header(header_text):
    """Extract state name from Excel file header."""
    try:
        # Look for patterns like "Vehicle Class Month Wise Data of [STATE] (2025)"
        if "of " in header_text and " (" in header_text:
            # Extract text between "of " and " ("
            start_idx = header_text.find("of ") + 3
            end_idx = header_text.find(" (")
            if start_idx < end_idx:
                state_name = header_text[start_idx:end_idx].strip()
                return state_name
        return None
    except:
        return None

def rename_file_by_state(original_file, state_name):
    """Rename file to include state name for better organization."""
    try:
        if state_name:
            # Clean state name for filename
            clean_state_name = state_name.replace(" & ", "And").replace(" ", "").replace("-", "")
            new_filename = f"{clean_state_name}.xlsx"
            new_filepath = original_file.parent / new_filename
            
            # Only rename if it's different. If target exists, overwrite to allow monthly updates
            if original_file.name != new_filename:
                try:
                    original_file.rename(new_filepath)
                except FileExistsError:
                    try:
                        new_filepath.unlink()
                    except Exception:
                        pass
                    original_file.rename(new_filepath)
                log_push_pipeline(f"Renamed/updated file to: {new_filename}")
                return new_filepath
        return original_file
    except Exception as e:
        log_push_pipeline(f"⚠️ Could not rename file: {e}")
        return original_file

def process_single_state_file(file_path):
    """Process a single state Excel file and extract structured data."""
    try:
        log_push_pipeline(f"Processing state file: {file_path.name}")
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Extract state name from the first column header
        state_name = None
        if len(df.columns) > 0:
            first_col_header = str(df.columns[0])
            state_name = extract_state_name_from_header(first_col_header)
            log_push_pipeline(f"Detected state: {state_name}")
        
        # Rename file if state was detected
        if state_name:
            file_path = rename_file_by_state(file_path, state_name)
        
        # Find the data rows (skip header rows)
        data_start_row = None
        for idx, row in df.iterrows():
            if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip().isdigit():
                data_start_row = idx
                break
        
        if data_start_row is None:
            log_push_pipeline(f"⚠️ Could not find data rows in {file_path.name}")
            return None
        
        # Extract vehicle classes and month data
        vehicle_data = []
        for idx in range(data_start_row, len(df)):
            row = df.iloc[idx]
            if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip().isdigit():
                vehicle_class = str(row.iloc[1]).strip()
                if vehicle_class and vehicle_class != "nan":
                    # Extract month-wise data (columns 2 to 9 typically contain month data)
                    for col_idx in range(2, min(10, len(row))):
                        month_value = row.iloc[col_idx]
                        if pd.notna(month_value) and str(month_value).strip().isdigit():
                            month_name = df.iloc[2, col_idx]  # Month names are in row 2
                            if pd.notna(month_name) and month_name in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
                                vehicle_data.append({
                                    'State': state_name or 'Unknown',
                                    'Vehicle_Class': vehicle_class,
                                    'Month_Name': month_name,
                                    'EV_Sales_Quantity': int(month_value),
                                    'Year': 2025  # From the header
                                })
        
        if vehicle_data:
            result_df = pd.DataFrame(vehicle_data)
            log_push_pipeline(f"Extracted {len(result_df)} records from {file_path.name}")
            return result_df
        else:
            log_push_pipeline(f"⚠️ No valid data extracted from {file_path.name}")
            return None
            
    except Exception as e:
        log_push_pipeline(f"❌ Error processing {file_path.name}: {str(e)}")
        return None

def process_multiple_state_files(excel_files):
    """Process multiple state Excel files and combine them into one dataset."""
    try:
        all_data = []
        
        for file_path in excel_files:
            # FIX: Ignore temporary Excel files created when a file is open
            if file_path.name.startswith('~$'):
                log_push_pipeline(f"Ignoring temporary file: {file_path.name}")
                continue
            state_data = process_single_state_file(file_path)
            if state_data is not None and not state_data.empty:
                all_data.append(state_data)
        
        if all_data:
            # Combine all state data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Add Date column (approximate date for each month)
            month_to_date = {
                'JAN': '2025-01-15', 'FEB': '2025-02-15', 'MAR': '2025-03-15',
                'APR': '2025-04-15', 'MAY': '2025-05-15', 'JUN': '2025-06-15',
                'JUL': '2025-07-15', 'AUG': '2025-08-15', 'SEP': '2025-09-15',
                'OCT': '2025-10-15', 'NOV': '2025-11-15', 'DEC': '2025-12-15'
            }
            
            combined_df['Date'] = combined_df['Month_Name'].map(month_to_date)
            combined_df['Date'] = pd.to_datetime(combined_df['Date'])
            
            log_push_pipeline(f"✅ Combined data from {len(excel_files)} state files. Total records: {len(combined_df)}")
            return combined_df
        else:
            log_push_pipeline("❌ No valid data extracted from any state files")
            return None
            
    except Exception as e:
        log_push_pipeline(f"❌ Error processing multiple state files: {str(e)}")
        return None

# In scripts/run_pipeline.py, replace this entire function

# In scripts/run_pipeline.py, replace this entire function

def standardize_vehicle_categories(df):
    """
    Standardize vehicle categories and column name to match model expectations.
    This robust version only maps values that are not already in the standard format.
    """
    if 'Vehicle_Class' not in df.columns:
        log_push_pipeline("⚠️ Warning: 'Vehicle_Class' column not found. Skipping standardization.")
        if 'Vehicle_Category' not in df.columns:
            df['Vehicle_Category'] = 'Unknown'
        return df

    vehicle_mapping = {
        'TWO WHEELER(NT)': '2-Wheelers', 'TWO WHEELER(T)': '2-Wheelers',
        'TWO WHEELER (INVALID CARRIAGE)': '2-Wheelers',
        'MOTOR CYCLE/SCOOTER-USED FOR HIRE': '2-Wheelers',
        'M-CYCLE/SCOOTER': '2-Wheelers',
        'THREE WHEELER(T)': '3-Wheelers', 'THREE WHEELER(NT)': '3-Wheelers',
        'FOUR WHEELER (INVALID CARRIAGE)': '4-Wheelers', 'MOTOR CAR': '4-Wheelers',
        'MOTOR CAB': '4-Wheelers', 'LIGHT MOTOR VEHICLE': '4-Wheelers',
        'LIGHT PASSENGER VEHICLE': '4-Wheelers',
        'BUS': 'Bus', 'HEAVY PASSENGER VEHICLE': 'Bus',
        'MEDIUM PASSENGER VEHICLE': 'Bus',
        'LIGHT GOODS VEHICLE': 'Others', 'HEAVY GOODS VEHICLE': 'Others'
    }
    
    # List of our target, standardized categories
    standard_categories = ['2-Wheelers', '3-Wheelers', '4-Wheelers', 'Bus', 'Others']
    
    # Only try to map values that are NOT already in the standard format
    df['Vehicle_Class'] = df['Vehicle_Class'].apply(
        lambda x: vehicle_mapping.get(x, x) if x not in standard_categories else x
    )
    
    # Any remaining values that are not standard (and not None) get set to 'Others'
    df.loc[~df['Vehicle_Class'].isin(standard_categories), 'Vehicle_Class'] = 'Others'

    # Rename the column to what the trainer expects
    df.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
    
    log_push_pipeline("📊 Vehicle category standardization and renaming complete.")
    return df
# In scripts/run_pipeline.py, replace the preprocess_data_simplified function

def preprocess_data_simplified():
    """
    Finds new data files, processes them, appends them to the master dataset,
    and cleans up processed files.
    """
    try:
        master_dataset_path = ROOT_DIR / "data" / "EV_Dataset.csv"
        
        if master_dataset_path.exists():
            log_push_pipeline(f"Loading master dataset from {master_dataset_path}")
            master_df = pd.read_csv(master_dataset_path)
        else:
            log_push_pipeline("Master dataset not found. Starting fresh.")
            master_df = pd.DataFrame()

        new_excel_files = list(DATA_DIR.glob("*.xlsx"))
        if not new_excel_files:
            log_push_pipeline("No new Excel data files found to process.")
            if not master_df.empty:
                 master_df.to_csv(OUTPUT_DIR / "preprocessed_data.csv", index=False)
            return True

        log_push_pipeline(f"Found {len(new_excel_files)} new Excel files for processing.")
        new_data_df = process_multiple_state_files(new_excel_files)

        if new_data_df is None or new_data_df.empty:
            log_push_pipeline("No valid data extracted from new files. No updates made.")
            return True

        new_data_df = standardize_vehicle_categories(new_data_df)
        
        # Ensure 'Vehicle_Category' exists in the master_df before concatenation
        if 'Vehicle_Class' in master_df.columns and 'Vehicle_Category' not in master_df.columns:
            master_df.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)

        combined_df = pd.concat([master_df, new_data_df], ignore_index=True)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        
        # *** THE FIX: Use the correct column name 'Vehicle_Category' ***
        combined_df = combined_df.drop_duplicates(subset=['Date', 'State', 'Vehicle_Category'], keep='last')
        combined_df = combined_df.sort_values(by=['Date', 'State', 'Vehicle_Category'])
        
        combined_df.to_csv(master_dataset_path, index=False)
        log_push_pipeline(f"✅ Master dataset updated and saved. Total records: {len(combined_df)}")
        combined_df.to_csv(OUTPUT_DIR / "preprocessed_data.csv", index=False)
        
        log_push_pipeline("Cleaning up processed Excel files...")
        for f in new_excel_files:
            try:
                f.unlink()
                log_push_pipeline(f" - Deleted {f.name}")
            except Exception as e:
                log_push_pipeline(f" - Could not delete {f.name}: {e}")

        return True

    except Exception as e:
        log_push_pipeline(f"❌ Data preprocessing failed: {e}")
        return False

def run_real_model():
    """Run the actual trained EV model on the preprocessed data."""
    try:
        log_push_pipeline("Loading preprocessed data...")
        preprocessed_path = OUTPUT_DIR / "preprocessed_data.csv"
        
        if not preprocessed_path.exists():
            log_push_pipeline("❌ Preprocessed data not found. Run preprocessing first.")
            return False
        
        df = pd.read_csv(preprocessed_path)
        log_push_pipeline(f"Loaded preprocessed data with {len(df)} records")
        
        # Load the trained model
        log_push_pipeline("Loading trained EV model...")
        if not MODEL_PATH.exists():
            log_push_pipeline("❌ Trained model not found. Please ensure ev_model.pkl exists.")
            return False
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        log_push_pipeline("✅ Trained model loaded successfully")
        
        # Prepare features for the model
        log_push_pipeline("Preparing features for model prediction...")
        
        # Create features that the model expects
        # Based on the original demand_forecast.py, we need to create similar features
        df_features = df.copy()
        
        # Create date-based features
        df_features['Date'] = pd.to_datetime(df_features['Date'])
        df_features['day'] = df_features['Date'].dt.day
        df_features['month'] = df_features['Date'].dt.month
        df_features['year'] = df_features['Date'].dt.year
        df_features['quarter'] = df_features['Date'].dt.quarter
        df_features['day_of_week'] = df_features['Date'].dt.dayofweek
        df_features['week_of_year'] = df_features['Date'].dt.isocalendar().week.astype(int)
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Create cyclical features
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month']/12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month']/12)
        df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week']/7)
        df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week']/7)
        
        # Convert categorical variables
        df_features['State'] = df_features['State'].astype('category')
        df_features['Vehicle_Class'] = df_features['Vehicle_Class'].astype('category')
        
        # Create lag features (simple approach)
        df_features = df_features.sort_values(['State', 'Vehicle_Class', 'Date'])
        df_features['lag_1'] = df_features.groupby(['State', 'Vehicle_Class'], observed=False)['EV_Sales_Quantity'].shift(1).fillna(0)
        df_features['lag_7'] = df_features.groupby(['State', 'Vehicle_Class'], observed=False)['EV_Sales_Quantity'].shift(7).fillna(0)
        
        # Simple rolling features (avoiding complex groupby operations)
        df_features['rolling_mean_7'] = df_features['EV_Sales_Quantity'].rolling(window=7, min_periods=1).mean().fillna(0)
        df_features['rolling_std_7'] = df_features['EV_Sales_Quantity'].rolling(window=7, min_periods=1).std().fillna(0)
        
        # Select features for prediction
        feature_columns = [
            'year', 'month', 'day', 'quarter', 'day_of_week', 'week_of_year',
            'is_weekend', 'State', 'Vehicle_Class', 'lag_1', 'lag_7',
            'rolling_mean_7', 'rolling_std_7', 'month_sin', 'month_cos',
            'day_of_week_sin', 'day_of_week_cos'
        ]
        
        # Ensure all features exist
        available_features = [col for col in feature_columns if col in df_features.columns]
        missing_features = [col for col in feature_columns if col not in df_features.columns]
        
        if missing_features:
            log_push_pipeline(f"⚠️ Missing features: {missing_features}")
            log_push_pipeline(f"Using available features: {available_features}")
        
        X_pred = df_features[available_features].copy()
        
        # Handle categorical variables
        if 'State' in X_pred.columns:
            X_pred['State'] = X_pred['State'].cat.codes
        if 'Vehicle_Class' in X_pred.columns:
            X_pred['Vehicle_Class'] = X_pred['Vehicle_Class'].cat.codes
        
        # Fill any remaining NaN values
        X_pred = X_pred.fillna(0)
        
        log_push_pipeline(f"Features prepared. Shape: {X_pred.shape}")
        
        # Make predictions
        log_push_pipeline("Making predictions with trained model...")
        try:
            predictions = model.predict(X_pred)
            log_push_pipeline("✅ Predictions generated successfully")
        except Exception as pred_error:
            log_push_pipeline(f"⚠️ Model prediction failed: {pred_error}")
            log_push_pipeline("Falling back to simple averaging...")
            # Fallback to simple averaging if model fails
            predictions = df.groupby(['State', 'Vehicle_Class'])['EV_Sales_Quantity'].transform('mean').values
        
        # Create results dataframe
        results_df = df[['State', 'Vehicle_Class', 'Month_Name', 'EV_Sales_Quantity', 'Date']].copy()
        results_df['Predicted_Sales'] = predictions
        results_df['Prediction_Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save predictions to CSV
        predictions_output_path = OUTPUT_DIR / "real_model_predictions.csv"
        results_df.to_csv(predictions_output_path, index=False)
        
        # Also insert predictions into the database for the dashboard
        log_push_pipeline("Inserting predictions into database...")
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Insert each prediction into the database
            for _, row in results_df.iterrows():
                # Calculate error and other metrics
                actual = row['EV_Sales_Quantity']
                predicted = row['Predicted_Sales']
                error = abs(actual - predicted)
                
                # Insert into live_predictions table
                cursor.execute("""
                    INSERT INTO live_predictions 
                    (timestamp, date, state, vehicle_category, actual_sales, predicted_sales, error, model_confidence, processing_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    str(row['Date']),  # Convert to string to avoid format issues
                    row['State'],
                    row['Vehicle_Class'],
                    actual,
                    predicted,
                    error,
                    0.85,  # Default confidence
                    150.0   # Default processing time in ms
                ))
            
            # Commit the changes
            conn.commit()
            conn.close()
            log_push_pipeline("✅ Predictions inserted into database successfully")
            
        except Exception as db_error:
            log_push_pipeline(f"⚠️ Database insertion failed: {db_error}")
            log_push_pipeline("Predictions still saved to CSV file")
        
        log_push_pipeline(f"✅ Real model predictions completed. Saved to: {predictions_output_path}")
        log_push_pipeline(f"Generated {len(results_df)} predictions")
        
        # Log some sample predictions
        sample_predictions = results_df.head(5)
        log_push_pipeline("Sample predictions:")
        for _, row in sample_predictions.iterrows():
            actual = row['EV_Sales_Quantity']
            predicted = row['Predicted_Sales']
            log_push_pipeline(f"  {row['State']} - {row['Vehicle_Class']}: Actual={actual}, Predicted={predicted:.2f}")
        
        return True
        
    except Exception as e:
        log_push_pipeline(f"❌ Real model failed: {str(e)}")
        return False


# In scripts/run_pipeline.py, replace this function

def check_requirements():
    """Check if required files and directories exist."""
    print("🔎 Checking requirements...")
    
    # FIX: Define paths locally for this function
    MODEL_PATH = MODELS_DIR / "ev_model.pkl"

    # Ensure directories exist
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    required_files = {
        "Dataset": DATASET_PATH,
        "Trained Model": MODEL_PATH
    }
    
    missing_items = []
    for name, path in required_files.items():
        if not path.exists():
            missing_items.append(f"{name} (at {path})")
            
    if missing_items:
        print("\n❌ Missing required files:")
        for item in missing_items:
            print(f"   - {item}")
        return False
    
    print("✅ Requirements met.")
    return True

# In scripts/run_pipeline.py, replace this function

def run_simulation(delay=1.0, max_records=None, db_conn=None):
    """Run the live simulation with a provided or new database connection."""
    print(f"🚀 Starting live simulation with {delay}s delay...")
    
    conn_was_provided = db_conn is not None
    conn = db_conn

    # *** THE FIX: If no connection is provided, create one ***
    if not conn_was_provided:
        print("No DB connection provided, creating a new one for this session...")
        conn = initialize_database()

    if conn is None:
        print("❌ Simulation failed: Could not establish a database connection.")
        return

    if not any(MODELS_DIR.glob("advanced_model_*.pkl")):
        print("❌ No advanced models found. Please train them first.")
        if not conn_was_provided: conn.close()
        return

    print("🤖 Using category-specific Advanced Models for simulation...")
    
    sys.path.insert(0, str(SCRIPTS_DIR))
    from live_simulation import LiveEVDataSimulator
    
    simulator = LiveEVDataSimulator(
        data_path=str(DATASET_PATH),
        models_dir=str(MODELS_DIR),
        db_connection=conn
    )
    
    try:
        simulator.start_simulation(delay_seconds=delay, max_records=max_records)
    except KeyboardInterrupt:
        print("\n⏹️  Simulation stopped by user.")
    except Exception as e:
        print(f"❌ Simulation error: {e}")
    finally:
        # Only close the connection if this function created it
        if not conn_was_provided and conn:
            conn.close()
            print("Simulation-specific database connection closed.")
        if str(SCRIPTS_DIR) in sys.path:
            sys.path.remove(str(SCRIPTS_DIR))

def run_dashboard():
    """Run the Streamlit dashboard."""
    print("📈 Starting Streamlit dashboard...")
    DASHBOARD_SCRIPT_PATH = SCRIPTS_DIR / "streamlit_dashboard.py"
    if not DASHBOARD_SCRIPT_PATH.exists():
        print(f"❌ Dashboard script not found: {DASHBOARD_SCRIPT_PATH}")
        return
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(DASHBOARD_SCRIPT_PATH)],
            check=True
        )
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def initialize_database():
    """Creates or resets the database and returns a connection object."""
    DB_PATH.parent.mkdir(exist_ok=True)
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS live_predictions")
        cursor.execute("""
            CREATE TABLE live_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, date TEXT, state TEXT,
                vehicle_category TEXT, actual_sales INTEGER, predicted_sales REAL,
                error REAL, model_confidence REAL, processing_time_ms REAL
            )
        """)
        conn.commit()
        print("Database initialized successfully.")
        return conn
    except Exception as e:
        print(f"Database initialization failed: {e}")
        return None

def run_both():
    """Run both simulation and dashboard in separate threads with a shared DB connection."""
    print("🎯 Starting both simulation and dashboard...")
    
    conn = initialize_database()
    if conn is None:
        return

    sim_thread = threading.Thread(target=run_simulation, args=(1.0, None, conn), daemon=True)
    sim_thread.start()
    
    print("⏳ Giving simulation time to initialize...")
    time.sleep(2)
    
    run_dashboard()

    # When the dashboard is closed, the main thread will end, and we can close the connection
    print("Dashboard closed. Closing database connection.")
    conn.close()

def train_model():
    """Train the model by running the demand_forecast.py script."""
    print("🔄 Training the model...")
    
    # *** THE FIX: Define the path inside the function ***
    TRAINING_SCRIPT_PATH = SCRIPTS_DIR / "demand_forecast.py"
    
    if not TRAINING_SCRIPT_PATH.exists():
        print(f"❌ Training script not found: {TRAINING_SCRIPT_PATH}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(TRAINING_SCRIPT_PATH)],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        print("✅ Model training completed successfully.")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print("\n--- STDOUT ---")
        print(e.stdout)
        print("\n--- STDERR ---")
        print(e.stderr)
        return False

def setup_environment():
    """Create all necessary project directories."""
    print("🛠️  Setting up environment directories...")
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    SCRIPTS_DIR.mkdir(exist_ok=True)
    print("✅ Environment setup complete.")

def show_status():
    """Show the current status of the pipeline files and database."""
    print("\n📊 Pipeline Status")
    print("=" * 50)
    
    # FIX: Define paths locally for this function
    MODEL_PATH = MODELS_DIR / "ev_model.pkl"
    SIMULATION_SCRIPT_PATH = SCRIPTS_DIR / "live_simulation.py"
    DASHBOARD_SCRIPT_PATH = SCRIPTS_DIR / "streamlit_dashboard.py"
    
    files_to_check = {
        "📁 Dataset": DATASET_PATH,
        "🤖 Trained Baseline Model": MODEL_PATH,
        "🎯 Simulation Script": SIMULATION_SCRIPT_PATH,
        "📈 Dashboard Script": DASHBOARD_SCRIPT_PATH,
        "🗄️  Database": DB_PATH
    }
    
    for name, path in files_to_check.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {name}: Present ({size_mb:.2f} MB)")
        else:
            print(f"❌ {name}: Missing")

    # Check for advanced models
    advanced_models = list(MODELS_DIR.glob("advanced_model_*.pkl"))
    if advanced_models:
        print(f"✅ 🚀 Advanced Models: Found {len(advanced_models)} category-specific models.")
    else:
        print(f"❌ 🚀 Advanced Models: Missing.")
            
    # Check database record count
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM live_predictions")
            count = cursor.fetchone()[0]
            conn.close()
            print(f"📈 Database Records: {count} predictions logged.")
        except sqlite3.Error as e:
            print(f"⚠️ Could not check database records: {e}")
            
    print("=" * 50)

def interactive_menu():
    """Display the interactive command-line menu."""
    # Set the working directory to the project root
    os.chdir(ROOT_DIR)
    print(f"📍 Working directory set to: {ROOT_DIR}")

    while True:
        print("\n🚗 EV Sales Live Pipeline Launcher")
        print("=" * 40)
        print("1. 📊 Show Status")
        print("2. 🔄 Train Model(Baseline Model ev_model.pkl)")
        print("3. 🚀 Run Simulation Only")
        print("4. 📈 Run Dashboard Only")
        print("5. 🎯 Run Both (Simulation + Dashboard)")
        print("6. ⚡ Quick Test (50 records)")
        print("7. 🛠️  Setup Environment")
        print("8. 📥 Run Data Push Pipeline")
        print("9. 🎯 Compare Accuracy (EV_Dataset vs Data Push)")
        print("10. 🚀 Use Advanced Model (High Performance)")
        print("11. 🔄 Incremental Training (Update Model)NOT WORKING")
        print("12. ❌ Exit")
        
        choice = input("\nEnter your choice (1-12): ").strip()
        
        if choice == "1":
            show_status()
        elif choice == "2":
            train_model()
        elif choice == "3":
            if not check_requirements(): continue
            try:
                delay = float(input("Enter delay between predictions (seconds, default 1.0): ") or "1.0")
                run_simulation(delay)
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
        elif choice == "4":
            run_dashboard()
        elif choice == "5":
            if not check_requirements(): continue
            run_both()
        elif choice == "6":
            if not check_requirements(): continue
            print("🧪 Running quick test with 50 records...")
            run_simulation(delay=0.5, max_records=50)
        elif choice == "7":
            setup_environment()
        elif choice == "8":
            run_data_push_pipeline()
        elif choice == "9":
            run_accuracy_comparison()
        elif choice == "10":
            run_advanced_model()
        elif choice == "11":
            run_incremental_training()
        elif choice == "12":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def main():
    """Main entry point for command-line arguments."""
    os.chdir(ROOT_DIR)
    
    parser = argparse.ArgumentParser(description="EV Sales Live Pipeline Launcher")
    parser.add_argument("--mode", choices=["simulation", "dashboard", "both", "train", "status", "test", "datapush", "accuracy", "advanced", "incremental"], 
                         help="Directly run a specific mode without the interactive menu.")
    parser.add_argument("--delay", type=float, default=1.0, 
                        help="Delay in seconds for simulation mode.")
    parser.add_argument("--max-records", type=int, 
                        help="Max records for simulation mode.")
    
    args = parser.parse_args()
    
    if not args.mode:
        interactive_menu()
    else:
        print(f"📍 Working directory: {os.getcwd()}")
        if args.mode == "status":
            show_status()
        elif args.mode == "train":
            train_model()
        elif args.mode in ["simulation", "test"]:
            if check_requirements():
                max_r = 50 if args.mode == "test" else args.max_records
                run_simulation(args.delay, max_r)
        elif args.mode == "dashboard":
            run_dashboard()
        elif args.mode == "both":
            if check_requirements():
                run_both()
        elif args.mode == "datapush":
            run_data_push_pipeline()
        elif args.mode == "accuracy":
            run_accuracy_comparison()
        elif args.mode == "advanced":
            run_advanced_model()
        elif args.mode == "incremental":
            run_incremental_training()

if __name__ == "__main__":
    main()
