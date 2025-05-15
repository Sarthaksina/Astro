#!/usr/bin/env python
# Simple Optuna Hyperparameter Tuning Example

"""
A simplified example of hyperparameter tuning with Optuna for the Cosmic Market Oracle.
This script demonstrates the core functionality without requiring MLflow.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import optuna
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Create a simple dataset for demonstration
def create_synthetic_data(n_samples=1000, sequence_length=30, n_features=10):
    """Create synthetic time series data for demonstration."""
    # Generate random features
    X = np.random.randn(n_samples, sequence_length, n_features)
    
    # Generate target values (simple function of the features)
    y = np.sum(X[:, -1, :3], axis=1, keepdims=True) * 0.5
    y += np.random.randn(*y.shape) * 0.1  # Add some noise
    
    # Split into train, validation, and test sets
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Simple LSTM model for demonstration
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to optimize
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256, step=32)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )
    
    # Create model
    model = SimpleLSTM(
        input_dim=X_train.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=y_train.shape[1],
        dropout=dropout
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    n_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        # Report to Optuna
        trial.report(val_loss, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Track best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss

# Main function
def main():
    print("Starting Optuna hyperparameter optimization...")
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=20)
    
    # Print results
    print("Optimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value:.6f}")
    print("Best hyperparameters:")
    for param, value in study.best_trial.params.items():
        print(f"    {param}: {value}")
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    
    # Create output directory
    results_dir = Path("results/optuna")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(results_dir / "optimization_history.png")
    
    # Save results as JSON
    results = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "all_trials": [
            {
                "number": trial.number,
                "value": trial.value if trial.value is not None else None,
                "params": trial.params,
                "state": trial.state.name
            }
            for trial in study.trials
        ]
    }
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_dir}")

# Generate synthetic data
print("Generating synthetic data...")
X_train, y_train, X_val, y_val, X_test, y_test = create_synthetic_data()
print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")

if __name__ == "__main__":
    main()
