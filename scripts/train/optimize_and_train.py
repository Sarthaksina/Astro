#!/usr/bin/env python
# Cosmic Market Oracle - Hyperparameter Optimization and Training

"""
Script to run hyperparameter optimization with Optuna and train a model with the optimized parameters.

This script:
1. Loads market and astrological data
2. Performs hyperparameter optimization using Optuna
3. Trains the best model with the optimized parameters
4. Evaluates the model on test data
5. Saves the model and results
"""

import os
import sys
import logging
import argparse
import yaml

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import mlflow
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.hyperparameter_tuning import OptunaHyperparameterTuner
from src.data_acquisition.market_data import fetch_historical_data
from src.data_acquisition.astrological_data import fetch_planetary_positions
# from src.features.feature_generator import generate_features # This seems to be an incorrect import path or file
# Assuming it should be:
from src.feature_engineering.feature_generator import generate_features
from src.utils.logger import get_logger # Changed
from src.utils.config import load_config
from src.models.model_factory import create_model_from_config, get_optimizer, get_scheduler
from src.evaluation.metrics import calculate_metrics, plot_predictions

# Configure logging
logger = get_logger("optimize_and_train") # Changed

def prepare_data(config):
    """
    Prepare data for training and evaluation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader, test_loader
    """
    logger.info("Preparing data...")
    
    # Data settings
    data_config = config['data']
    start_date = data_config.get('start_date', '2010-01-01')
    end_date = data_config.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    sequence_length = data_config.get('sequence_length', 90)
    prediction_horizon = data_config.get('prediction_horizon', 5)
    batch_size = data_config.get('batch_size', 64)
    
    # Fetch market data
    logger.info(f"Fetching market data from {start_date} to {end_date}...")
    market_data = fetch_historical_data(
        start_date=start_date,
        end_date=end_date,
        symbols=data_config.get('symbols', ['SPY']),
        interval=data_config.get('interval', '1d')
    )
    
    # Fetch astrological data
    logger.info("Fetching planetary positions...")
    planetary_data = fetch_planetary_positions(
        start_date=start_date,
        end_date=end_date,
        planets=config['features']['astrological'].get('planets', None)
    )
    
    # Generate features
    logger.info("Generating features...")
    features, targets = generate_features(
        market_data=market_data,
        planetary_data=planetary_data,
        config=config
    )
    
    # Split data
    train_split = data_config.get('train_split', 0.7)
    val_split = data_config.get('val_split', 0.15)
    
    n_samples = len(features)
    train_idx = int(n_samples * train_split)
    val_idx = train_idx + int(n_samples * val_split)
    
    X_train = features[:train_idx]
    y_train = targets[:train_idx]
    X_val = features[train_idx:val_idx]
    y_val = targets[train_idx:val_idx]
    X_test = features[val_idx:]
    y_test = targets[val_idx:]
    
    # Create sequences
    def create_sequences(X, y, seq_length):
        sequences_X, sequences_y = [], []
        for i in range(len(X) - seq_length - prediction_horizon + 1):
            sequences_X.append(X[i:i+seq_length])
            sequences_y.append(y[i+seq_length+prediction_horizon-1])
        return np.array(sequences_X), np.array(sequences_y)
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq)
    X_val_tensor = torch.FloatTensor(X_val_seq)
    y_val_tensor = torch.FloatTensor(y_val_seq)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.FloatTensor(y_test_seq)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logger.info(f"Data preparation complete. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def run_hyperparameter_optimization(config, train_loader, val_loader, test_loader):
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        config: Configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        
    Returns:
        Best parameters and optimization results
    """
    logger.info("Starting hyperparameter optimization...")
    
    # Hyperparameter tuning settings
    hp_config = config['hyperparameter_tuning']
    n_trials = hp_config.get('n_trials', 100)
    timeout = hp_config.get('timeout', 86400)  # 24 hours
    study_name = hp_config.get('study_name', 'cosmic_market_oracle')
    
    # Create pruner
    if hp_config.get('pruner', 'median') == 'median':
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=1
        )
    elif hp_config.get('pruner', 'median') == 'successive_halving':
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    elif hp_config.get('pruner', 'median') == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = None
    
    # Create sampler
    if hp_config.get('sampler', 'tpe') == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=42)
    elif hp_config.get('sampler', 'tpe') == 'random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    elif hp_config.get('sampler', 'tpe') == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler(seed=42)
    else:
        sampler = None
    
    # Set up MLflow tracking
    mlflow_config = config.get('mlflow', {})
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", mlflow_config.get('tracking_uri', 'http://localhost:5000')))
    mlflow.set_experiment(mlflow_config.get('experiment_name', 'cosmic_market_oracle'))
    
    # Create tuner
    tuner = OptunaHyperparameterTuner(
        config_path=None,  # We're passing the config directly
        data_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_trials=n_trials,
        timeout=timeout,
        study_name=study_name,
        pruner=pruner,
        sampler=sampler,
        direction=hp_config.get('direction', 'maximize')
    )
    
    # Run optimization
    results = tuner.optimize()
    
    logger.info(f"Hyperparameter optimization complete. Best parameters: {results['best_params']}")
    
    return results['best_params'], results

def train_model_with_best_params(config, best_params, train_loader, val_loader, test_loader):
    """
    Train a model with the best parameters from hyperparameter optimization.
    
    Args:
        config: Configuration dictionary
        best_params: Best parameters from hyperparameter optimization
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        
    Returns:
        Trained model and evaluation metrics
    """
    logger.info("Training model with best parameters...")
    
    # Update config with best parameters
    for key, value in best_params.items():
        if key == 'model_type':
            config['model']['type'] = value
        elif key == 'learning_rate':
            config['training']['learning_rate'] = value
        elif key == 'batch_size':
            config['data']['batch_size'] = value
        elif key == 'dropout':
            config['model']['dropout'] = value
        elif key == 'weight_decay':
            config['training']['weight_decay'] = value
        elif key == 'optimizer':
            config['training']['optimizer'] = value
        elif key == 'scheduler':
            config['training']['scheduler'] = value
        else:
            # Model-specific parameters
            model_type = best_params.get('model_type', config['model']['type'])
            if model_type == 'lstm':
                if key in ['hidden_size', 'num_layers', 'bidirectional', 'attention_heads']:
                    config['lstm_config'][key] = value
            elif model_type == 'tcn':
                if key in ['num_channels', 'kernel_size', 'num_levels']:
                    config['tcn_config'][key] = value
            elif model_type == 'transformer' or model_type == 'advanced_transformer':
                if key in ['num_heads', 'num_layers', 'dim_feedforward']:
                    config['advanced_transformer_config'][key] = value
            elif model_type == 'gnn':
                if key in ['gnn_layers', 'gnn_hidden_dim', 'aggregation']:
                    config['gnn_config'][key] = value
            elif model_type == 'hybrid':
                if key in ['cnn_filters', 'cnn_kernel_size', 'transformer_heads', 'transformer_layers']:
                    config['hybrid_config'][key] = value
    
    # Create model
    model_config = {
        'model_type': best_params.get('model_type', config['model']['type']),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lstm_config': config.get('lstm_config', {}),
        'tcn_config': config.get('tcn_config', {}),
        'advanced_transformer_config': config.get('advanced_transformer_config', {}),
        'gnn_config': config.get('gnn_config', {}),
        'hybrid_config': config.get('hybrid_config', {}),
        'neural_ode_config': config.get('neural_ode_config', {}),
        'ensemble_config': config.get('ensemble_config', {})
    }
    
    model = create_model_from_config(model_config)
    
    # Training settings
    training_config = config['training']
    num_epochs = training_config.get('epochs', 100)
    learning_rate = best_params.get('learning_rate', training_config.get('learning_rate', 0.001))
    weight_decay = best_params.get('weight_decay', training_config.get('weight_decay', 0.0001))
    optimizer_type = best_params.get('optimizer', training_config.get('optimizer', 'adamw'))
    scheduler_type = best_params.get('scheduler', training_config.get('scheduler', 'cosine'))
    
    # Create optimizer
    optimizer_config = {
        'type': optimizer_type,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }
    optimizer = get_optimizer(model, {'optimizer_config': optimizer_config})
    
    # Create scheduler
    scheduler_config = {
        'type': scheduler_type,
        'T_max': num_epochs,
        'step_size': num_epochs // 4,
        'patience': 10,
        'factor': 0.5
    }
    scheduler = get_scheduler(optimizer, {'scheduler_config': scheduler_config})
    
    # Loss function
    criterion = nn.MSELoss()
    
    # MLflow tracking
    mlflow_config = config.get('mlflow', {})
    run_name = mlflow_config.get('run_name', f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(best_params)
        mlflow.log_param('num_epochs', num_epochs)
        
        # Training loop
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
            
            val_loss /= len(val_loader)
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, optuna.pruners.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss
            }, step=epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                
                predictions.extend(output.cpu().numpy())
                actuals.extend(target.cpu().numpy())
        
        test_loss /= len(test_loader)
        
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(actuals),
            np.array(predictions)
        )
        
        # Log metrics
        mlflow.log_metric('test_loss', test_loss)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Plot predictions
        fig = plot_predictions(
            np.array(actuals),
            np.array(predictions),
            title=f"Model: {best_params.get('model_type', config['model']['type'])}"
        )
        
        # Save figure
        fig_path = Path("results/figures")
        fig_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path / f"predictions_{run_name}.png")
        
        # Log figure
        mlflow.log_artifact(str(fig_path / f"predictions_{run_name}.png"))
        
        # Save model
        model_path = Path("models")
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path / f"best_model_{run_name}.pt")
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        # Log as ONNX
        if config.get('inference', {}).get('use_onnx', True):
            dummy_input = next(iter(test_loader))[0][0:1].to(device)
            mlflow.onnx.log_model(model, "onnx_model", input_example=dummy_input)
        
        logger.info(f"Model training complete. Test Loss: {test_loss:.6f}")
        logger.info(f"Metrics: {metrics}")
    
    return model, metrics

def main():
    """Main function to run hyperparameter optimization and training."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization and training for Cosmic Market Oracle")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(config)
    
    # Run hyperparameter optimization
    best_params, opt_results = run_hyperparameter_optimization(
        config, train_loader, val_loader, test_loader
    )
    
    # Train model with best parameters
    model, metrics = train_model_with_best_params(
        config, best_params, train_loader, val_loader, test_loader
    )
    
    # Save results
    results = {
        'best_params': best_params,
        'metrics': metrics,
        'optimization_results': {
            'n_trials': opt_results['n_trials'],
            'completed_trials': opt_results['completed_trials'],
            'pruned_trials': opt_results['pruned_trials'],
            'best_score': opt_results['best_score']
        }
    }
    
    results_path = Path("results")
    results_path.mkdir(parents=True, exist_ok=True)
    
    with open(results_path / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Optimization and training complete!")

if __name__ == "__main__":
    main()
