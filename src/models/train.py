# Cosmic Market Oracle - Model Training Module

"""
This module handles the training of machine learning models for market prediction
based on astrological and market data. It supports different model architectures
and training configurations.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch

# Import project modules
from src.models.model_factory import create_model_from_config, get_optimizer, get_scheduler
from src.data_processing.integrator import integrate_market_astro_data
from src.feature_engineering.feature_generator import generate_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles the training of market prediction models."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the model trainer with configuration.
        
        Args:
            config_path: Path to the model configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Set up MLflow tracking
        mlflow_tracking_uri = self.config.get('mlflow', {}).get('tracking_uri', 'mlruns')
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.experiment_name = self.config.get('mlflow', {}).get('experiment_name', 'cosmic_market_oracle')
        mlflow.set_experiment(self.experiment_name)
        
        # Create output directories
        self.output_dir = Path(self.config.get('output_dir', 'models'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ModelTrainer with device: {self.device}")
    
    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Load model configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'model_type': 'lstm',
            'data': {
                'start_date': '1900-01-01',
                'end_date': '2023-01-01',
                'symbol': '^DJI',
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'sequence_length': 60,
                'target_horizon': 20,
                'features': 'all'
            },
            'training': {
                'batch_size': 64,
                'epochs': 100,
                'early_stopping_patience': 10,
                'learning_rate': 0.001,
                'weight_decay': 0.0001
            },
            'lstm_config': {
                'input_dim': 64,
                'hidden_dim': 128,
                'num_layers': 2,
                'output_dim': 1,
                'dropout': 0.2
            },
            'optimizer_config': {
                'type': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 0.0001
            },
            'scheduler_config': {
                'type': 'plateau',
                'patience': 5,
                'factor': 0.5
            },
            'mlflow': {
                'tracking_uri': 'mlruns',
                'experiment_name': 'cosmic_market_oracle',
                'log_artifacts': True
            }
        }
        
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        user_config = json.load(f)
                    
                    # Update default config with user config
                    self._deep_update(default_config, user_config)
                    logger.info(f"Loaded configuration from {config_path}")
                except Exception as e:
                    logger.error(f"Error loading config from {config_path}: {str(e)}")
                    logger.warning("Using default configuration")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Deep update a nested dictionary with another dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with values to update in base_dict
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data for model training, validation, and testing.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        data_config = self.config.get('data', {})
        
        # Load and integrate market and astrological data
        start_date = data_config.get('start_date', '1900-01-01')
        end_date = data_config.get('end_date', '2023-01-01')
        symbol = data_config.get('symbol', '^DJI')
        
        logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
        
        # In a real implementation, this would load actual data
        # For now, we'll create synthetic data for demonstration
        from src.data_acquisition.market_data import get_complete_dji_history
        
        # Get market data
        market_data = get_complete_dji_history(start_date=start_date, end_date=end_date)
        
        # Integrate with astrological data
        integrated_data = integrate_market_astro_data(market_data)
        
        # Generate features
        features_df = generate_features(integrated_data)
        
        # Prepare sequences
        sequence_length = data_config.get('sequence_length', 60)
        target_horizon = data_config.get('target_horizon', 20)
        
        # Create sequences and targets
        sequences, targets = self._create_sequences(features_df, sequence_length, target_horizon)
        
        # Split data
        train_ratio = data_config.get('train_ratio', 0.7)
        val_ratio = data_config.get('val_ratio', 0.15)
        test_ratio = data_config.get('test_ratio', 0.15)
        
        # Calculate split indices
        n_samples = len(sequences)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        # Split data
        X_train = sequences[:train_size]
        y_train = targets[:train_size]
        
        X_val = sequences[train_size:train_size+val_size]
        y_val = targets[train_size:train_size+val_size]
        
        X_test = sequences[train_size+val_size:]
        y_test = targets[train_size+val_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        scaler.fit(X_train_reshaped)
        
        # Apply scaling
        X_train_scaled = self._scale_sequences(X_train, scaler)
        X_val_scaled = self._scale_sequences(X_val, scaler)
        X_test_scaled = self._scale_sequences(X_test, scaler)
        
        # Create PyTorch datasets and dataloaders
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        
        test_dataset = TensorDataset(
            torch.tensor(X_test_scaled, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        
        # Create dataloaders
        batch_size = self.config.get('training', {}).get('batch_size', 64)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        logger.info(f"Prepared data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples")
        
        return train_loader, val_loader, test_loader
    
    def _create_sequences(self, df: pd.DataFrame, sequence_length: int, target_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            df: DataFrame with features
            sequence_length: Length of input sequences
            target_horizon: How far in the future to predict
            
        Returns:
            Tuple of (sequences, targets)
        """
        data = df.values
        target_col_idx = df.columns.get_loc('close')  # Assuming 'close' is the target column
        
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length - target_horizon + 1):
            # Extract sequence
            seq = data[i:i+sequence_length]
            sequences.append(seq)
            
            # Extract target (future price change)
            current_price = data[i+sequence_length-1, target_col_idx]
            future_price = data[i+sequence_length+target_horizon-1, target_col_idx]
            price_change = (future_price / current_price) - 1.0
            targets.append(price_change)
        
        return np.array(sequences), np.array(targets)
    
    def _scale_sequences(self, sequences: np.ndarray, scaler: StandardScaler) -> np.ndarray:
        """
        Scale sequences using a fitted scaler.
        
        Args:
            sequences: Input sequences
            scaler: Fitted StandardScaler
            
        Returns:
            Scaled sequences
        """
        # Reshape to 2D for scaling
        orig_shape = sequences.shape
        sequences_reshaped = sequences.reshape(sequences.shape[0], -1)
        
        # Scale
        sequences_scaled = scaler.transform(sequences_reshaped)
        
        # Reshape back to original shape
        return sequences_scaled.reshape(orig_shape)
    
    def train_model(self) -> nn.Module:
        """
        Train the model using the configured parameters.
        
        Returns:
            Trained model
        """
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data()
        
        # Create model
        model = create_model_from_config(self.config)
        model = model.to(self.device)
        
        # Get optimizer and scheduler
        optimizer = get_optimizer(model, self.config)
        scheduler = get_scheduler(optimizer, self.config)
        
        # Set up loss function
        criterion = nn.MSELoss()
        
        # Training parameters
        epochs = self.config.get('training', {}).get('epochs', 100)
        early_stopping_patience = self.config.get('training', {}).get('early_stopping_patience', 10)
        
        # Initialize tracking variables
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{self.config['model_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                'model_type': self.config['model_type'],
                'sequence_length': self.config['data']['sequence_length'],
                'target_horizon': self.config['data']['target_horizon'],
                'batch_size': self.config['training']['batch_size'],
                'learning_rate': self.config['optimizer_config']['learning_rate'],
                'epochs': epochs
            })
            
            # Training loop
            logger.info(f"Starting training for {epochs} epochs")
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * inputs.size(0)
                
                train_loss /= len(train_loader.dataset)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        # Forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs.squeeze(), targets)
                        
                        val_loss += loss.item() * inputs.size(0)
                
                val_loss /= len(val_loader.dataset)
                
                # Update learning rate scheduler if using validation loss
                if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                elif scheduler is not None:
                    scheduler.step()
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, step=epoch)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                    
                    # Save best model checkpoint
                    model_path = self.output_dir / f"{self.config['model_type']}_best.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'config': self.config
                    }, model_path)
                    
                    mlflow.log_artifact(str(model_path))
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Load best model for evaluation
            model.load_state_dict(best_model_state)
            
            # Evaluate on test set
            model.eval()
            test_loss = 0.0
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets)
                    
                    test_loss += loss.item() * inputs.size(0)
                    
                    # Store predictions and actuals
                    predictions.extend(outputs.squeeze().cpu().numpy())
                    actuals.extend(targets.cpu().numpy())
            
            test_loss /= len(test_loader.dataset)
            
            # Calculate additional metrics
            mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            
            # Log test metrics
            mlflow.log_metrics({
                'test_loss': test_loss,
                'test_mse': mse,
                'test_rmse': rmse,
                'test_mae': mae
            })
            
            logger.info(f"Test Loss: {test_loss:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
            
            # Save final model
            final_model_path = self.output_dir / f"{self.config['model_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': self.config,
                'test_metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae
                }
            }, final_model_path)
            
            mlflow.log_artifact(str(final_model_path))
            
            # Log model to MLflow
            mlflow.pytorch.log_model(model, "model")
        
        return model


def main():
    """
    Main function to run model training from command line.
    """
    parser = argparse.ArgumentParser(description="Train market prediction model")
    parser.add_argument("--config", type=str, help="Path to model configuration JSON file")
    args = parser.parse_args()
    
    trainer = ModelTrainer(args.config)
    model = trainer.train_model()
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
