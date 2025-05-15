# Cosmic Market Oracle - Tests for Model Training Module

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.models.train import (
    ModelTrainer,
    prepare_data_loaders,
    train_model,
    save_model,
    log_metrics
)


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    # Create a DataFrame with 100 samples of mock training data
    np.random.seed(42)  # For reproducibility
    
    # Features (20 features)
    X = np.random.normal(0, 1, (100, 20))
    
    # Target (next day's return)
    y = np.random.normal(0, 0.01, 100)
    
    # Create a DataFrame
    feature_cols = [f'feature_{i}' for i in range(20)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    
    # Add dates as index
    dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
    df.index = dates
    
    return df


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model for testing."""
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(20, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    return MockModel()


class TestModelTraining:
    """Tests for model training functions."""
    
    def test_prepare_data_loaders(self, sample_training_data):
        """Test preparation of data loaders from DataFrame."""
        df = sample_training_data.copy()
        
        # Call the function
        train_loader, val_loader, test_loader = prepare_data_loaders(
            df, target_column='target', train_ratio=0.7, val_ratio=0.15, batch_size=16
        )
        
        # Verify the data loaders are correctly created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Check the sizes
        total_samples = len(df)
        expected_train_samples = int(total_samples * 0.7)
        expected_val_samples = int(total_samples * 0.15)
        expected_test_samples = total_samples - expected_train_samples - expected_val_samples
        
        # Count samples in each loader
        train_samples = sum(len(batch[0]) for batch in train_loader)
        val_samples = sum(len(batch[0]) for batch in val_loader)
        test_samples = sum(len(batch[0]) for batch in test_loader)
        
        # Allow for rounding differences
        assert abs(train_samples - expected_train_samples) <= 1
        assert abs(val_samples - expected_val_samples) <= 1
        assert abs(test_samples - expected_test_samples) <= 1
        
        # Check that the data types are correct
        for x, y in train_loader:
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            assert x.dtype == torch.float32
            assert y.dtype == torch.float32
            assert x.shape[1] == 20  # 20 features
            assert y.shape[1] == 1   # 1 target
            break
    
    @patch('mlflow.log_metric')
    @patch('mlflow.log_params')
    def test_log_metrics(self, mock_log_params, mock_log_metric):
        """Test logging of metrics to MLflow."""
        # Sample metrics and parameters
        metrics = {
            'loss': 0.05,
            'val_loss': 0.06,
            'r2_score': 0.75,
            'mse': 0.002
        }
        
        params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        }
        
        # Call the function
        log_metrics(metrics, params)
        
        # Verify that MLflow functions were called correctly
        mock_log_params.assert_called_once_with(params)
        
        # Verify each metric was logged
        assert mock_log_metric.call_count == len(metrics)
        for key, value in metrics.items():
            mock_log_metric.assert_any_call(key, value)
    
    @patch('torch.save')
    def test_save_model(self, mock_save, mock_model):
        """Test saving of trained model."""
        # Call the function
        model_path = 'test_model.pt'
        save_model(mock_model, model_path)
        
        # Verify that torch.save was called correctly
        mock_save.assert_called_once()
        args, kwargs = mock_save.call_args
        assert args[0] == mock_model.state_dict()
        assert args[1] == model_path
    
    @patch('src.models.train.log_metrics')
    @patch('src.models.train.save_model')
    def test_train_model(self, mock_save_model, mock_log_metrics, mock_model, sample_training_data):
        """Test the main model training function."""
        df = sample_training_data.copy()
        
        # Prepare data loaders
        train_loader, val_loader, _ = prepare_data_loaders(
            df, target_column='target', train_ratio=0.7, val_ratio=0.15, batch_size=16
        )
        
        # Mock the model's training behavior
        mock_model.train = MagicMock(return_value=None)
        
        # Call the function with minimal epochs for testing
        trained_model, metrics = train_model(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=0.001,
            epochs=2,  # Use small number for testing
            patience=5,
            model_path='test_model.pt'
        )
        
        # Verify that the model was trained
        assert trained_model is not None
        assert metrics is not None
        assert isinstance(metrics, dict)
        
        # Verify that save_model was called
        mock_save_model.assert_called_once()
        
        # Verify that log_metrics was called
        mock_log_metrics.assert_called_once()
    
    def test_model_trainer_class(self, mock_model, sample_training_data):
        """Test the ModelTrainer class."""
        df = sample_training_data.copy()
        
        # Initialize the trainer
        trainer = ModelTrainer(
            model=mock_model,
            data=df,
            target_column='target',
            learning_rate=0.001,
            batch_size=16,
            epochs=2  # Use small number for testing
        )
        
        # Verify the trainer is correctly initialized
        assert trainer.model is mock_model
        assert trainer.learning_rate == 0.001
        assert trainer.batch_size == 16
        assert trainer.epochs == 2
        
        # Mock the train_model function to avoid actual training
        with patch('src.models.train.train_model') as mock_train:
            mock_train.return_value = (mock_model, {'loss': 0.05, 'val_loss': 0.06})
            
            # Call the train method
            trained_model, metrics = trainer.train()
            
            # Verify that train_model was called with correct arguments
            mock_train.assert_called_once()
            assert trained_model is mock_model
            assert metrics['loss'] == 0.05
            assert metrics['val_loss'] == 0.06
