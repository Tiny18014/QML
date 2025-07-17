import pandas as pd
import numpy as np
import pickle
import sqlite3
import time
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('output/simulation.log', mode='w')
    ]
)

class LiveEVDataSimulator:
    """
    Simulates live EV sales predictions using a two-part (classifier/regressor)
    model bundle for each category.
    """
    def __init__(self, data_path, model_path, db_path="output/live_predictions.db"):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.db_path = Path(db_path)
        
        self.model_bundle = self._load_model_bundle()
        self.test_data = self._load_and_prepare_test_data()
        
        self.conn = self._init_database()
        self.is_running = False
        self.predictions_count = 0
        self.category_metrics = defaultdict(lambda: {'errors': [], 'percentage_errors': [], 'count': 0})
        self.processing_times = []

    def _load_model_bundle(self):
        """Loads the pickled dictionary of two-part model bundles."""
        try:
            with open(self.model_path, 'rb') as f:
                bundle = pickle.load(f)
            logging.info(f"Model bundle with {len(bundle)} models loaded successfully")
            return bundle
        except FileNotFoundError:
            logging.error(f"Model file not found at {self.model_path}")
            return None
        except Exception as e:
            logging.error(f"Error loading model bundle: {e}")
            return None

    def _create_historical_features(self, df):
        """Creates lag and rolling window features for a specific category."""
        df = df.sort_values(by=['State', 'Date']).copy()
        grouped = df.groupby(['State'], observed=False)
        
        new_feature_cols = [
            'lag_1_day', 'lag_7_days', 'rolling_mean_7_days', 
            'rolling_std_7_days', 'rolling_mean_30_days'
        ]

        df['lag_1_day'] = grouped['EV_Sales_Quantity'].shift(1)
        df['lag_7_days'] = grouped['EV_Sales_Quantity'].shift(7)
        df['rolling_mean_7_days'] = grouped['EV_Sales_Quantity'].shift(1).rolling(window=7, min_periods=1).mean()
        df['rolling_std_7_days'] = grouped['EV_Sales_Quantity'].shift(1).rolling(window=7, min_periods=1).std()
        df['rolling_mean_30_days'] = grouped['EV_Sales_Quantity'].shift(1).rolling(window=30, min_periods=1).mean()
        
        df[new_feature_cols] = df[new_feature_cols].fillna(0)
        return df

    def _load_and_prepare_test_data(self):
        """
        Loads the entire dataset, creates all features (including historical),
        and then filters for the test period.
        """
        try:
            df = pd.read_csv(self.data_path)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date', 'EV_Sales_Quantity'], inplace=True)
            df['EV_Sales_Quantity'] = pd.to_numeric(df['EV_Sales_Quantity'], errors='coerce').fillna(0).astype(int)

            df['day'] = df['Date'].dt.day
            df['month'] = df['Date'].dt.month
            df['year'] = df['Date'].dt.year
            df['quarter'] = df['Date'].dt.quarter
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
            df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
            df['State'] = df['State'].astype('category')
            df['Vehicle_Category'] = df['Vehicle_Category'].astype('category')

            all_category_dfs = []
            for category in df['Vehicle_Category'].unique():
                category_df = df[df['Vehicle_Category'] == category].copy()
                all_category_dfs.append(self._create_historical_features(category_df))
            
            featured_df = pd.concat(all_category_dfs)
            test_data = featured_df[featured_df['Date'].dt.year >= 2023].sort_values('Date').reset_index(drop=True)
            logging.info(f"Test data with pre-calculated features loaded: {len(test_data)} records")
            return test_data
        except FileNotFoundError:
            logging.error(f"Data file not found at {self.data_path}")
            return pd.DataFrame()

    def _init_database(self):
        """Initializes the SQLite database."""
        self.db_path.parent.mkdir(exist_ok=True)
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
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
            logging.info(f"Database initialized at {self.db_path}")
            return conn
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
            return None

    def start_simulation(self, delay_seconds=1.0, max_records=None):
        """Starts the live prediction simulation."""
        if not self.model_bundle:
            logging.error("Cannot start simulation: model bundle not loaded.")
            return

        self.is_running = True
        logging.info(f"Starting live simulation with {delay_seconds}s delay...")

        records_to_process = self.test_data.head(max_records) if max_records else self.test_data
        
        feature_columns = [
            'year', 'month', 'day', 'quarter', 'day_of_week', 'week_of_year',
            'is_weekend', 'State', 'lag_1_day', 'lag_7_days', 'rolling_mean_7_days',
            'rolling_std_7_days', 'rolling_mean_30_days', 'month_sin', 'month_cos',
            'day_of_week_sin', 'day_of_week_cos'
        ]

        for index, row in records_to_process.iterrows():
            if not self.is_running: break
            
            start_time = time.time()
            try:
                category = row['Vehicle_Category']
                
                bundle_for_category = self.model_bundle.get(category)
                
                if not bundle_for_category:
                    logging.warning(f"No model found for category: {category}. Skipping.")
                    continue
                
                classifier = bundle_for_category['classifier']
                regressor = bundle_for_category['regressor']
                state_cats_for_model = bundle_for_category['state_categories']

                features_df = pd.DataFrame([row])[feature_columns]
                features_df['State'] = pd.Categorical(features_df['State'], categories=state_cats_for_model)

                # --- TWO-PART PREDICTION LOGIC ---
                # Step 1: Predict if sales will happen
                will_have_sales = classifier.predict(features_df)[0]
                
                prediction = 0 # Default prediction is 0
                if will_have_sales == 1 and regressor is not None:
                    # Step 2: If sales are predicted, use the regressor to predict the amount
                    log_prediction = regressor.predict(features_df)[0]
                    prediction = np.expm1(log_prediction)
                
                prediction = max(0, prediction)

                actual = row['EV_Sales_Quantity']
                error = abs(actual - prediction)
                percentage_error = (error / actual) * 100 if actual > 0 else 0
                proc_time_ms = (time.time() - start_time) * 1000
                confidence = 0.90 

                self._log_to_db(row, actual, prediction, error, confidence, proc_time_ms)
                self._update_stats(category, error, percentage_error, proc_time_ms)
                if self.predictions_count % 10 == 0:
                    self._print_stats(row, actual, prediction, error, confidence)
            except Exception as e:
                logging.error(f"Error processing record {index}: {e}", exc_info=True)
            time.sleep(delay_seconds)
        
        logging.info("Simulation completed successfully")
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

    def _update_stats(self, category, error, percentage_error, proc_time):
        """Updates the running statistics."""
        self.predictions_count += 1
        self.category_metrics[category]['errors'].append(error)
        self.category_metrics[category]['percentage_errors'].append(percentage_error)
        self.category_metrics[category]['count'] += 1
        self.processing_times.append(proc_time)

    def _print_stats(self, record, actual, predicted, error, confidence):
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
        print("Category Performance:")
        for cat, data in self.category_metrics.items():
            if data['errors']:
                cat_mae = np.mean(data['errors'])
                cat_mape = np.mean(data['percentage_errors'])
                print(f"  {cat}: MAE = {cat_mae:.2f}, MAPE = {cat_mape:.2f}%")
        print("\nLatest Prediction:")
        print(f"  Date: {record['Date'].date()}, State: {record['State']}, Category: {record['Vehicle_Category']}")
        print(f"  Actual: {int(actual)}, Predicted: {int(predicted)}, Error: {error:.2f}")
        print("="*60)

    def stop_simulation(self):
        """Stops the simulation and closes the database connection."""
        self.is_running = False
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")
