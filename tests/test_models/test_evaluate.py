# Cosmic Market Oracle - Tests for Model Evaluation Module

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from src.models.evaluate import (
    ModelEvaluator,
    calculate_metrics,
    plot_predictions,
    calculate_directional_accuracy,
    evaluate_model
)


@pytest.fixture
def sample_test_data():
    """Create sample test data for model evaluation."""
    # Create a DataFrame with 50 samples of mock test data
    np.random.seed(42)  # For reproducibility
    
    # Features (20 features)
    X = np.random.normal(0, 1, (50, 20))
    
    # Target (next day's return)
    y = np.random.normal(0, 0.01, 50)
    
    # Create a DataFrame
    feature_cols = [f'feature_{i}' for i in range(20)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    
    # Add dates as index
    dates = pd.date_range(start='2023-01-01', periods=50, freq='B')
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
    
    model = MockModel()
    # Initialize with random weights for reproducible predictions
    torch.manual_seed(42)
    for param in model.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.01)
    
    return model


@pytest.fixture
def sample_predictions():
    """Create sample predictions and actual values for testing."""
    np.random.seed(42)  # For reproducibility
    
    # Actual values
    y_true = np.random.normal(0, 0.01, 50)
    
    # Predictions (correlated with actual values but with some error)
    y_pred = y_true * 0.8 + np.random.normal(0, 0.005, 50)
    
    # Create a DataFrame with dates
    dates = pd.date_range(start='2023-01-01', periods=50, freq='B')
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred
    }, index=dates)
    
    return df


class TestModelEvaluation:
    """Tests for model evaluation functions."""
    
    def test_calculate_metrics(self, sample_predictions):
        """Test calculation of evaluation metrics."""
        df = sample_predictions.copy()
        
        # Call the function
        metrics = calculate_metrics(df['actual'], df['predicted'])
        
        # Verify the metrics dictionary has the expected keys
        expected_keys = ['mse', 'rmse', 'mae', 'r2', 'explained_variance', 'directional_accuracy']
        for key in expected_keys:
            assert key in metrics, f"Metric {key} missing from results"
        
        # Verify the metrics are in reasonable ranges
        assert 0 <= metrics['mse'] < 0.001, "MSE should be small for correlated data"
        assert 0 <= metrics['rmse'] < 0.03, "RMSE should be small for correlated data"
        assert 0 <= metrics['mae'] < 0.02, "MAE should be small for correlated data"
        assert 0 <= metrics['r2'] <= 1, "R² should be between 0 and 1"
        assert 0 <= metrics['explained_variance'] <= 1, "Explained variance should be between 0 and 1"
        assert 0 <= metrics['directional_accuracy'] <= 1, "Directional accuracy should be between 0 and 1"
        
        # For this specific test data, R² should be positive since predictions are correlated
        assert metrics['r2'] > 0, "R² should be positive for correlated predictions"
    
    def test_calculate_directional_accuracy(self, sample_predictions):
        """Test calculation of directional accuracy."""
        df = sample_predictions.copy()
        
        # Call the function
        accuracy = calculate_directional_accuracy(df['actual'], df['predicted'])
        
        # Verify the result is a float between 0 and 1
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        
        # Test with perfect directional prediction
        perfect_pred = np.sign(df['actual']) * np.abs(np.random.normal(0, 0.01, len(df)))
        perfect_accuracy = calculate_directional_accuracy(df['actual'], perfect_pred)
        assert perfect_accuracy == 1.0
        
        # Test with opposite directional prediction
        opposite_pred = -np.sign(df['actual']) * np.abs(np.random.normal(0, 0.01, len(df)))
        opposite_accuracy = calculate_directional_accuracy(df['actual'], opposite_pred)
        assert opposite_accuracy == 0.0
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_plot_predictions(self, mock_figure, mock_savefig, sample_predictions):
        """Test plotting of predictions vs actual values."""
        df = sample_predictions.copy()
        
        # Call the function
        plot_predictions(df['actual'], df['predicted'], df.index, 'test_plot.png')
        
        # Verify that matplotlib functions were called
        mock_figure.assert_called_once()
        mock_savefig.assert_called_once_with('test_plot.png', dpi=300, bbox_inches='tight')
    
    @patch('src.models.evaluate.calculate_metrics')
    @patch('src.models.evaluate.plot_predictions')
    def test_evaluate_model(self, mock_plot, mock_metrics, mock_model, sample_test_data):
        """Test the main model evaluation function."""
        df = sample_test_data.copy()
        
        # Mock the metrics calculation
        mock_metrics.return_value = {
            'mse': 0.0001,
            'rmse': 0.01,
            'mae': 0.008,
            'r2': 0.75,
            'explained_variance': 0.76,
            'directional_accuracy': 0.68
        }
        
        # Prepare test data
        X = torch.tensor(df.drop('target', axis=1).values, dtype=torch.float32)
        y = torch.tensor(df['target'].values.reshape(-1, 1), dtype=torch.float32)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size=16
        )
        
        # Call the function
        metrics, predictions_df = evaluate_model(
            model=mock_model,
            test_loader=test_loader,
            dates=df.index,
            output_dir='./test_output'
        )
        
        # Verify that metrics calculation was called
        mock_metrics.assert_called_once()
        
        # Verify that plot_predictions was called
        mock_plot.assert_called_once()
        
        # Verify the returned objects
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert predictions_df is not None
        assert isinstance(predictions_df, pd.DataFrame)
        assert len(predictions_df) == len(df)
    
    def test_model_evaluator_class(self, mock_model, sample_test_data):
        """Test the ModelEvaluator class."""
        df = sample_test_data.copy()
        
        # Initialize the evaluator
        evaluator = ModelEvaluator(
            model=mock_model,
            test_data=df,
            target_column='target',
            batch_size=16,
            output_dir='./test_output'
        )
        
        # Verify the evaluator is correctly initialized
        assert evaluator.model is mock_model
        assert evaluator.batch_size == 16
        assert evaluator.output_dir == './test_output'
        
        # Mock the evaluate_model function to avoid actual evaluation
        with patch('src.models.evaluate.evaluate_model') as mock_evaluate:
            mock_metrics = {
                'mse': 0.0001,
                'rmse': 0.01,
                'mae': 0.008,
                'r2': 0.75,
                'explained_variance': 0.76,
                'directional_accuracy': 0.68
            }
            mock_predictions = pd.DataFrame({
                'actual': df['target'].values,
                'predicted': df['target'].values * 0.8 + np.random.normal(0, 0.005, len(df))
            }, index=df.index)
            mock_evaluate.return_value = (mock_metrics, mock_predictions)
            
            # Call the evaluate method
            metrics, predictions_df = evaluator.evaluate()
            
            # Verify that evaluate_model was called with correct arguments
            mock_evaluate.assert_called_once()
            assert metrics == mock_metrics
            assert predictions_df is mock_predictions
