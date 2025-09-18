# scripts/live_simulation.py - FINAL THREAD-SAFE VERSION
import pandas as pd
import numpy as np
import pickle
import sqlite3
import time
import logging
from pathlib import Path
import sys
from datetime import datetime
from collections import defaultdict
# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('output/simulation.log', mode='w')
    ]
)

# --- Import the feature engineering AND the new prediction prep function ---
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
from advanced_model_trainer import create_advanced_features, prepare_features_for_prediction

class LiveEVDataSimulator:
    def __init__(self, data_path, models_dir, db_connection):
        self.data_path = Path(data_path)
        self.models_dir = Path(models_dir)
        self.conn = db_connection
        self.loaded_models = {}
        self.test_data = self._load_and_prepare_test_data()
        self.is_running = False
        self.predictions_count = 0
        self.category_metrics = defaultdict(lambda: {'errors': [], 'count': 0})
        self.processing_times = []
    def _load_and_prepare_test_data(self):
        """Loads the dataset and creates features using the master recipe."""
        try:
            df = pd.read_csv(self.data_path, parse_dates=['Date'])
            df.dropna(subset=['Date', 'EV_Sales_Quantity'], inplace=True)
            df['Vehicle_Category'] = df['Vehicle_Category'].fillna('Unknown')
            
            logging.info("Preparing simulation data using master feature engineering...")
            featured_df = create_advanced_features(df)

            test_data = featured_df[featured_df['Date'].dt.year >= 2024].sort_values('Date').reset_index(drop=True)
            logging.info(f"Test data with all correct features loaded: {len(test_data)} records")
            return test_data
        except FileNotFoundError:
            logging.error(f"Data file not found at {self.data_path}")
            return pd.DataFrame()

    def start_simulation(self, delay_seconds=1.0, max_records=None):
        """Starts the live prediction simulation."""
        if self.conn is None:
            logging.error("Cannot start simulation, no database connection provided.")
            return

        self.is_running = True
        logging.info(f"Starting live simulation...")
        records_to_process = self.test_data.head(max_records) if max_records else self.test_data
        
        for index, row in records_to_process.iterrows():
            if not self.is_running: break
            
            start_time = time.time()
            try:
                category = row['Vehicle_Category']
                
                if category not in self.loaded_models:
                    category_filename = category.replace(" ", "_").replace("/", "_")
                    model_path = self.models_dir / f"advanced_model_{category_filename}.pkl"
                    if not model_path.exists():
                        logging.warning(f"Model for '{category}' not found. Skipping.")
                        continue
                    logging.info(f"Loading model for '{category}'...")
                    with open(model_path, 'rb') as f:
                        self.loaded_models[category] = pickle.load(f)

                model_data = self.loaded_models[category]
                model = model_data['primary_model']
                feature_names = model_data['feature_names']
                scaler = model_data['scaler'] # <-- Load the correct scaler

                # *** THE FIX: Use the new prediction-specific function ***
                current_row_df = pd.DataFrame([row])
                X_scaled = prepare_features_for_prediction(current_row_df, feature_names, scaler)
                
                prediction = model.predict(X_scaled)[0]
                prediction = max(0, prediction)

                actual = row['EV_Sales_Quantity']
                error = abs(actual - prediction)
                proc_time_ms = (time.time() - start_time) * 1000
                confidence = 0.95 

                self._log_to_db(row, actual, prediction, error, confidence, proc_time_ms)

                self._update_stats(category, error, proc_time_ms)
                if self.predictions_count % 10 == 0:
                    self._print_stats(row, actual, prediction, error)
            except Exception as e:
                logging.error(f"Error on record {index}: {e}", exc_info=True)
            
            time.sleep(delay_seconds)
        
        logging.info("Simulation completed.")
        self.stop_simulation()

    def _log_to_db(self, record, actual, predicted, error, confidence, proc_time_ms):
        """Logs a single prediction to the SQLite database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO live_predictions (timestamp, date, state, vehicle_category, actual_sales, 
            predicted_sales, error, model_confidence, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(), record['Date'].date().isoformat(), record['State'],
            record['Vehicle_Category'], int(actual), float(predicted), float(error),
            float(confidence), float(proc_time_ms)
        ))
        self.conn.commit()

    def stop_simulation(self):
        """Stops the simulation and closes the database connection."""
        self.is_running = False
        # The main script is now responsible for closing the connection
        logging.info("Simulation thread finished.")
    
    # --- ADDED: New methods for tracking and printing statistics ---
    def _update_stats(self, category, error, proc_time):
        """Updates the running statistics."""
        self.predictions_count += 1
        self.category_metrics[category]['errors'].append(error)
        self.category_metrics[category]['count'] += 1
        self.processing_times.append(proc_time)

    def _print_stats(self, record, actual, predicted, error):
        """Prints formatted live statistics to the console."""
        all_errors = [e for cat_data in self.category_metrics.values() for e in cat_data['errors']]
        if not all_errors: return

        overall_mae = np.mean(all_errors)
        avg_proc_time = np.mean(self.processing_times)

        print("\n" + "="*60)
        print(f"LIVE PREDICTION STATS - Total Predictions: {self.predictions_count}")
        print("="*60)
        print(f"Overall MAE: {overall_mae:.2f}")
        print(f"Avg Processing Time: {avg_proc_time:.2f}ms\n")
        print("Category Performance (MAE):")
        for cat, data in self.category_metrics.items():
            if data['errors']:
                cat_mae = np.mean(data['errors'])
                print(f"  {cat:<15}: {cat_mae:.2f}")
        print("\nLatest Prediction:")
        print(f"  Date: {record['Date'].date()}, State: {record['State']}, Category: {record['Vehicle_Category']}")
        print(f"  Actual: {int(actual)}, Predicted: {int(round(predicted))}, Error: {error:.2f}")
        print("="*60)