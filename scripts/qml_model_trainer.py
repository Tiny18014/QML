#!/usr/bin/env python3
"""
QML Model Trainer (Quantum-Hybrid)
Trains a PyTorch-PennyLane Hybrid Model for EV Sales Prediction.
Exports functions for Inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# --- Paths ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_SAVE_PATH = MODELS_DIR / "ev_sales_hybrid_model_simple.pth"
PARAMS_SAVE_PATH = MODELS_DIR / "normalization_params.pkl"

# --- Config ---
torch_dtype = torch.float32
device = torch.device("cpu") # Force CPU for stability
MODEL_INPUT_DIM = 0 

# --- Quantum Layer Setup ---
try:
    import pennylane as qml
    HAS_QML = True
    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    class QuantumLayer(nn.Module):
        def __init__(self):
            super().__init__()
            weight_shapes = {"weights": (3, n_qubits)}
            self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        def forward(self, x):
            return self.q_layer(x)
            
except ImportError:
    print("⚠️ PennyLane not found. Falling back to Classical Simulation.")
    HAS_QML = False
    class QuantumLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.sim_layer = nn.Linear(4, 4)
        def forward(self, x):
            return torch.tanh(self.sim_layer(x))

# --- Hybrid Model Class ---
class HybridModel(nn.Module):
    def __init__(self, input_dim):
        super(HybridModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 4)
        self.q_layer = QuantumLayer()
        self.fc3 = nn.Linear(4, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))
        x = self.q_layer(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# --- Helper Functions ---

def feature_engineering(df):
    """Shared feature engineering logic."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_week'] = df['Date'].dt.dayofweek
    
    # Lags (Fill NaNs with 0 to prevent dropping rows in inference)
    for lag in [1, 3, 7]:
        df[f'lag_{lag}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].shift(lag)
    
    df = df.fillna(0)
    
    # One-Hot Encoding
    if 'Vehicle_Category' in df.columns:
        df = pd.get_dummies(df, columns=['Vehicle_Category'], prefix='cat')
    if 'State' in df.columns:
        df = pd.get_dummies(df, columns=['State'], prefix='state')
        
    return df

def preprocess_data(df, training=False):
    """
    TRAINING MODE: Calculates and returns normalization parameters.
    """
    df = feature_engineering(df)
    
    exclude_cols = ['Date', 'EV_Sales_Quantity', 'Month_Name', 'Vehicle_Class']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].astype(float).values
    y = df['EV_Sales_Quantity'].astype(float).values
    
    y_mean = y.mean()
    y_std = y.std() + 1e-6
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-6
    
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    X_tensor = torch.tensor(X_norm, dtype=torch_dtype)
    y_tensor = torch.tensor(y_norm, dtype=torch_dtype).view(-1, 1)
    
    params = {
        'feature_names': feature_cols,
        'input_dim': len(feature_cols),
        'y_mean': y_mean, 'y_std': y_std,
        'X_mean': X_mean, 'X_std': X_std
    }
    
    return X_tensor, y_tensor, params

def prepare_data_for_inference(df, params):
    """
    INFERENCE MODE: Uses saved parameters to normalize data.
    CRITICAL FIX: Ensures columns match exactly and uses saved Mean/Std.
    """
    # 1. Create features
    df_feat = feature_engineering(df)
    
    # 2. Align columns (Ensure exact match with training)
    expected_cols = params['feature_names']
    
    # Add missing columns with 0
    for col in expected_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0
            
    # Select and Sort columns to match training order EXACTLY
    X_df = df_feat[expected_cols].fillna(0)
    X = X_df.astype(float).values
    
    # 3. Normalize using SAVED stats
    X_mean = params['X_mean']
    X_std = params['X_std']
    
    # Safety check for shape mismatch
    if X.shape[1] != len(X_mean):
        print(f"⚠️ Dimension Mismatch: Input {X.shape[1]} vs Params {len(X_mean)}. Truncating/Padding.")
        if X.shape[1] > len(X_mean):
            X = X[:, :len(X_mean)]
        else:
            padding = np.zeros((X.shape[0], len(X_mean) - X.shape[1]))
            X = np.hstack([X, padding])

    X_norm = (X - X_mean) / X_std
    return torch.tensor(X_norm, dtype=torch_dtype)

def load_model(path, input_dim):
    model = HybridModel(input_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def load_norm_params(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# --- Training Loop ---
def train_qml_model():
    print("🧠 Training QML (Quantum-Hybrid) Model...")
    
    df = pd.read_csv(DATA_PATH)
    if 'Vehicle_Class' in df.columns:
        df.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
        
    X_train, y_train, params = preprocess_data(df, training=True)
    input_dim = params['input_dim']
    
    print(f"   Input Dimension: {input_dim}")
    
    model = HybridModel(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 10 
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(PARAMS_SAVE_PATH, 'wb') as f:
        pickle.dump(params, f)
        
    print(f"✅ QML Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_qml_model()