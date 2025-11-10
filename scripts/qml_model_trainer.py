#!/usr/bin/env python3
"""
Optimized EV Sales Forecasting Model with GPU Acceleration (CUDA)
==================================================================
FIX v6 (Generalization & Feature Alignment):
- ADDED Time Index and Holiday features for improved time-series modeling.
- REDUCED ClassicalNN hidden size and increased Weight Decay to combat overfitting.
- Ensured index alignment for all features using .transform().
- Correctly sets MODEL_INPUT_DIM in the global scope for prediction agents.
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import os
import joblib
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import holidays # NEW: For holiday feature

# =============== DEVICE SETUP ===============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============== CONFIGURATION ===============
REDUCED_EPOCHS = 20
USE_FLOAT32 = True

SIMPLIFIED_QUANTUM = True
BATCH_SIZE = 32
SAVE_MODEL = True
SHOTS = 1000
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4 # Increased weight decay to combat overfitting
SCHEDULER_FACTOR = 0.7
SCHEDULER_PATIENCE = 5

# --- Precision types ---
torch_dtype = torch.float32 if USE_FLOAT32 else torch.float64
np_dtype = np.float32 if USE_FLOAT32 else np.float64

# --- Global Constant for Predictor ---
MODEL_INPUT_DIM = None

# =============== DATA PREPROCESSING ===============

def preprocess_data(df: pd.DataFrame):
    """
    Processes a raw DataFrame into feature/target tensors.
    """
    print("Preprocessing data...")
    
    # --- 1. Robust Data Cleaning ---
    df = df.copy() 
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Robust numeric conversion
    df["EV_Sales_Quantity"] = df["EV_Sales_Quantity"].astype(str).str.replace(",", "")
    df["EV_Sales_Quantity"] = pd.to_numeric(df["EV_Sales_Quantity"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    
    if 'Vehicle_Class' in df.columns and 'Vehicle_Category' not in df.columns:
        df.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)

    df['Vehicle_Category'] = df['Vehicle_Category'].fillna('Unknown')
    df['State'] = df['State'].fillna('Unknown')
    df['EV_Sales_Quantity'] = df['EV_Sales_Quantity'].fillna(0)
    df['Year'] = df['Year'].fillna(df['Year'].mode().iloc[0])

    print("Data cleaning complete.")

    # --- 2. Create Time-Series Features ---
    print("Creating advanced time-series features...")
    df = df.sort_values(by=['State', 'Vehicle_Category', 'Date'])
    
    # NEW FEATURE: Time Index (Linear Trend)
    df['time_index'] = (df['Date'] - df['Date'].min()).dt.days
    
    # NEW FEATURE: Holiday Flag
    years_in_data = df['Year'].unique()
    valid_years = [int(y) for y in years_in_data if not np.isnan(y)] 
    in_holidays = holidays.country_holidays('IN', years=valid_years)
    df['is_holiday'] = df['Date'].isin(in_holidays).astype(np_dtype) 

    # Cyclical Time Features
    df['month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
    df['day_of_year'] = df['Date'].dt.dayofyear
    
    # Lag and Rolling Features (Grouped by entity, using .transform for index alignment)
    group_keys = ['State', 'Vehicle_Category']
    g = df.groupby(group_keys)['EV_Sales_Quantity']
    
    # Use .transform() for Lag Features (Fixes Index Error)
    df['lag_1'] = g.transform(lambda x: x.shift(1))
    df['lag_7'] = g.transform(lambda x: x.shift(7))
    
    # Use .transform() for Rolling Features (Fixes Index Error)
    df['rolling_mean_7'] = g.transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['rolling_mean_30'] = g.transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    df['rolling_std_7'] = g.transform(lambda x: x.rolling(window=7, min_periods=1).std())
    
    # --- Fill all NaNs created by shift/rolling with 0 ---
    df['lag_1'] = df['lag_1'].fillna(0)
    df['lag_7'] = df['lag_7'].fillna(0)
    df['rolling_mean_7'] = df['rolling_mean_7'].fillna(0)
    df['rolling_mean_30'] = df['rolling_mean_30'].fillna(0)
    df['rolling_std_7'] = df['rolling_std_7'].fillna(0)

    print("Time-series features created.")

    # --- 3. Encoding & Normalization ---
    
    # One-hot encode BOTH State and Vehicle_Category (drop_first=False for feature consistency)
    df = pd.get_dummies(df, columns=["State", "Vehicle_Category"], drop_first=False)
    
    # Normalize Year and Time Index
    for col in ["Year", "time_index"]:
        col_min = df[col].min()
        col_max = df[col].max()
        if (col_max - col_min) > 0:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0

    # Calculate normalization parameters for the target
    y_mean = df["EV_Sales_Quantity"].mean()
    y_std = df["EV_Sales_Quantity"].std()
    
    # Normalize the target 'EV_Sales_Quantity'
    if y_std > 0:
        df["EV_Sales_Quantity"] = (df["EV_Sales_Quantity"] - y_mean) / y_std
    else:
        df["EV_Sales_Quantity"] = 0 

    # Drop non-feature columns
    df = df.drop(columns=["Month_Name", "Date", "Vehicle_Type"], errors="ignore")
    
    # --- 4. Final Processing ---
    
    df = df.fillna(0)
    
    # Separate X and y
    y = df["EV_Sales_Quantity"].values.astype(np_dtype)
    X = df.drop(columns=["EV_Sales_Quantity"]).values.astype(np_dtype)
    
    # Store normalization parameters and feature names
    norm_params = {'y_mean': y_mean, 'y_std': y_std, 'feature_names': df.drop(columns=["EV_Sales_Quantity"]).columns.tolist()}

    X_tensor = torch.tensor(X, dtype=torch_dtype).to(device)
    y_tensor = torch.tensor(y, dtype=torch_dtype).reshape(-1, 1).to(device)

    print(f"Data preprocessing complete. Dataset shape: {X_tensor.shape}")
    
    return X_tensor, y_tensor, norm_params

# =============== QUANTUM DEVICE ===============
print("Forcing CPU-only mode. Using lightning.qubit device.")
dev = qml.device(
    "lightning.qubit",
    wires=2,
    shots=None
)

@qml.qnode(dev, interface="torch")
def quantum_model(inputs, weights):
    qml.AngleEmbedding(inputs, wires=[0, 1])
    if SIMPLIFIED_QUANTUM:
        qml.StronglyEntanglingLayers(weights[0:1], wires=[0, 1])
    else:
        qml.StronglyEntanglingLayers(weights, wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# =============== CLASSICAL NN ===============
class ClassicalNN(nn.Module):
    def __init__(self, input_dim):
        super(ClassicalNN, self).__init__()
        # Further reduced hidden size to combat overfitting
        hidden_size = 2 if SIMPLIFIED_QUANTUM else 4 
        self.fc1 = nn.Linear(input_dim, hidden_size, dtype=torch_dtype)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2, dtype=torch_dtype)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class HybridModel(nn.Module):
    def __init__(self, input_dim):
        super(HybridModel, self).__init__()
        self.classical_nn = ClassicalNN(input_dim)
        weight_layers = 1 if SIMPLIFIED_QUANTUM else 3
        self.quantum_weights = nn.Parameter(torch.randn((weight_layers, 2, 3), dtype=torch_dtype))
    def forward(self, x):
        x_classical = self.classical_nn(x)
        return quantum_model(x_classical, self.quantum_weights)

# =============== TRAINING FUNCTION ===============
def train_model(X_tensor, y_tensor, input_dim, model_path=None):
    print(f"\nTraining model with configuration:")
    print(f"- Epochs: {REDUCED_EPOCHS}")
    print(f"- Precision: {'float32' if USE_FLOAT32 else 'float64'}")
    print(f"- Quantum circuit: {'simplified' if SIMPLIFIED_QUANTUM else 'full'}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Learning Rate: {LEARNING_RATE}")
    print(f"- Weight Decay: {WEIGHT_DECAY}")

    hybrid_model = HybridModel(input_dim).to(device)
    optimizer = optim.Adam(hybrid_model.parameters(), 
                           lr=LEARNING_RATE, 
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                     factor=SCHEDULER_FACTOR, 
                                                     patience=SCHEDULER_PATIENCE)
    loss_function = nn.MSELoss()
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    loss_history = []
    print(f"\nStarting training for {REDUCED_EPOCHS} epochs...")
    for epoch in range(REDUCED_EPOCHS):
        epoch_start_time = time.time()
        total_loss = 0
        hybrid_model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            batch_pred = hybrid_model(batch_x).reshape(-1, 1)
            loss = loss_function(batch_pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(X_tensor)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{REDUCED_EPOCHS}: Loss = {avg_loss:.4f} (Time: {epoch_time:.2f}s)")
    print("\nTraining complete!")
    if SAVE_MODEL and model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(hybrid_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    return hybrid_model, loss_history

# =============== EVALUATION FUNCTIONS ===============
def evaluate_model(model, X_tensor, y_tensor, norm_params):
    print("\nEvaluating model performance on full dataset...")
    model.eval()
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False) 
    all_preds = []
    all_actual = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_pred = model(batch_x)
            all_preds.append(batch_pred.cpu())
            all_actual.append(batch_y.cpu())
    y_pred_list = torch.cat(all_preds).numpy().flatten()
    y_actual = torch.cat(all_actual).numpy().flatten()
    mse = np.mean((y_pred_list - y_actual) ** 2)
    mae = mean_absolute_error(y_actual, y_pred_list)
    r2 = r2_score(y_actual, y_pred_list)
    accuracy = np.mean(np.abs(y_pred_list - y_actual) < 0.1) * 100
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Accuracy (within normalized ±0.1 error): {accuracy:.2f}%")
    sample_size = min(5, len(y_actual))
    print(f"\nSample predictions (first {sample_size}):")
    print("Predicted (normalized):", y_pred_list[:sample_size])
    print("Actual (normalized):", y_actual[:sample_size])
    y_pred_denorm = y_pred_list * norm_params['y_std'] + norm_params['y_mean']
    y_actual_denorm = y_actual * norm_params['y_std'] + norm_params['y_mean']
    print("\nAfter denormalization:")
    print("Predicted:", y_pred_denorm[:sample_size])
    print("Actual:", y_actual_denorm[:sample_size])
    return {
        'mse': mse, 'mae': mae, 'r2': r2, 'accuracy': accuracy,
        'predictions': y_pred_list, 'actual': y_actual
    }

def plot_results(metrics, loss_history, save_path=None):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plot_sample_size = min(5000, len(metrics['actual']))
    indices = np.random.choice(len(metrics['actual']), plot_sample_size, replace=False)
    plt.scatter(metrics['actual'][indices], metrics['predictions'][indices], alpha=0.3)
    min_val = min(min(metrics['actual']), min(metrics['predictions']))
    max_val = max(max(metrics['actual']), max(metrics['predictions']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Actual vs Predicted (Normalized Sample)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.close()

# =============== MODEL SAVE/LOAD (Unused in main(), but kept for structure) ===============
def save_model_and_params(model, norm_params, model_path, params_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(norm_params, params_path)
    print(f"Model saved to {model_path}")
    print(f"Parameters saved to {params_path}")

def load_model(model_path, input_dim):
    print(f"Loading model from {model_path}...")
    model = HybridModel(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_norm_params(params_path):
    print(f"Loading normalization parameters from {params_path}...")
    return joblib.load(params_path)

# ================= MAIN EXECUTION =================

def main():
    """Main execution function"""
    global MODEL_INPUT_DIM
    start_time = time.time()
    
    # File paths
    data_path = "data/EV_Dataset.csv"
    model_dir = "models"
    model_path = os.path.join(model_dir, "ev_sales_hybrid_model_simple.pth")
    norm_params_path = os.path.join(model_dir, "normalization_params.pkl")
    plot_path = os.path.join(model_dir, "training_results.png")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please download the EV_Dataset.csv and place it in the 'data' directory.")
        return
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Pass the DataFrame to the processing function
    X_tensor, y_tensor, norm_params = preprocess_data(df)
    
    # CRITICAL: SET GLOBAL MODEL INPUT DIMENSION
    MODEL_INPUT_DIM = X_tensor.shape[1]
    input_dim = MODEL_INPUT_DIM
    norm_params['input_dim'] = MODEL_INPUT_DIM
    # Save the normalization parameters and feature names immediately after preprocessing
    joblib.dump(norm_params, norm_params_path)
    print(f"Normalization parameters saved to {norm_params_path}")

    model, loss_history = train_model(X_tensor, y_tensor, input_dim, model_path)
    
    metrics = evaluate_model(model, X_tensor, y_tensor, norm_params)
    
    plot_results(metrics, loss_history, plot_path)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    print("\nTo use this model for predictions:")
    print(f"1. Load the model: model = load_model('{model_path}', {input_dim})")
    print(f"2. Load params: norm_params = joblib.load('{norm_params_path}')")
    print(f"3. Use MODEL_INPUT_DIM = {MODEL_INPUT_DIM}")

if __name__ == "__main__":
    main()