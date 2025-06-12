# Cosmic Market Oracle - Model Evaluation Module

"""
This module handles the evaluation of trained models on test data and provides
detailed performance metrics and visualizations for market prediction models.
"""

import argparse
import json
# import logging # Removed
import os
from datetime import datetime
from src.utils.logger import get_logger # Added
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch

# Import project modules
from src.models.model_factory import create_model_from_config

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Removed
logger = get_logger(__name__) # Changed


class ModelEvaluator:
    """Evaluates trained market prediction models."""
    
    def __init__(self, model_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the saved model file (.pth)
            output_dir: Directory to save evaluation results
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and configuration
        self.checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.config = self.checkpoint.get('config', {})
        
        # Set device
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Create model
        self.model = self._load_model()
        
        logger.info(f"Initialized ModelEvaluator for model: {self.model_path.name}")
    
    def _load_model(self) -> nn.Module:
        """
        Load the model from the checkpoint.
        
        Returns:
            Loaded model
        """
        model = create_model_from_config(self.config)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode
        
        return model
    
    def evaluate_on_test_data(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader with test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Store predictions and actuals
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # Calculate directional accuracy
        direction_correct = np.sum((predictions > 0) == (actuals > 0))
        direction_accuracy = direction_correct / len(predictions)
        
        # Calculate additional metrics for market prediction
        # Positive/negative prediction accuracy
        pos_actuals = actuals > 0
        neg_actuals = actuals <= 0
        pos_predictions = predictions > 0
        neg_predictions = predictions <= 0
        
        true_positives = np.sum(pos_actuals & pos_predictions)
        false_positives = np.sum(neg_actuals & pos_predictions)
        true_negatives = np.sum(neg_actuals & neg_predictions)
        false_negatives = np.sum(pos_actuals & neg_predictions)
        
        # Precision, recall, F1 for positive class
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store results
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"Evaluation metrics: MSE={mse:.6f}, RMSE={rmse:.6f}, Direction Accuracy={direction_accuracy:.4f}")
        
        # Save metrics to file
        metrics_file = self.output_dir / f"{self.model_path.stem}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Store predictions and actuals for visualization
        self.predictions = predictions
        self.actuals = actuals
        
        return metrics
    
    def create_visualizations(self) -> None:
        """
        Create and save visualizations of model performance.
        """
        if not hasattr(self, 'predictions') or not hasattr(self, 'actuals'):
            logger.error("No predictions available. Run evaluate_on_test_data first.")
            return
        
        # Set up the figure style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Predictions vs Actuals Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.actuals, self.predictions, alpha=0.5)
        plt.plot([-0.5, 0.5], [-0.5, 0.5], 'r--')  # Diagonal line for perfect predictions
        plt.xlabel('Actual Returns')
        plt.ylabel('Predicted Returns')
        plt.title('Predicted vs Actual Returns')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{self.model_path.stem}_scatter.png", dpi=300)
        plt.close()
        
        # 2. Prediction Error Distribution
        errors = self.predictions - self.actuals
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{self.model_path.stem}_error_dist.png", dpi=300)
        plt.close()
        
        # 3. Cumulative Returns Comparison
        plt.figure(figsize=(12, 6))
        cumulative_actual = np.cumsum(self.actuals)
        cumulative_pred = np.cumsum(self.predictions)
        
        plt.plot(cumulative_actual, label='Actual Returns')
        plt.plot(cumulative_pred, label='Predicted Returns')
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Returns')
        plt.title('Cumulative Returns: Actual vs Predicted')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{self.model_path.stem}_cumulative.png", dpi=300)
        plt.close()
        
        # 4. Directional Accuracy by Magnitude
        plt.figure(figsize=(10, 6))
        
        # Group actual returns by magnitude
        abs_returns = np.abs(self.actuals)
        percentiles = np.percentile(abs_returns, np.arange(0, 101, 10))
        accuracy_by_magnitude = []
        
        for i in range(len(percentiles) - 1):
            mask = (abs_returns >= percentiles[i]) & (abs_returns < percentiles[i+1])
            if np.sum(mask) > 0:
                correct = np.sum((self.predictions[mask] > 0) == (self.actuals[mask] > 0))
                accuracy = correct / np.sum(mask)
                accuracy_by_magnitude.append((percentiles[i], accuracy))
        
        if accuracy_by_magnitude:
            magnitudes, accuracies = zip(*accuracy_by_magnitude)
            plt.bar(range(len(magnitudes)), accuracies, width=0.7)
            plt.xticks(range(len(magnitudes)), [f"{m:.3f}" for m in magnitudes], rotation=45)
            plt.xlabel('Return Magnitude (Absolute Value)')
            plt.ylabel('Directional Accuracy')
            plt.title('Directional Accuracy by Return Magnitude')
            plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{self.model_path.stem}_dir_accuracy.png", dpi=300)
        plt.close()
        
        logger.info(f"Saved visualizations to {self.output_dir}")
    
    def evaluate_trading_strategy(self, initial_capital: float = 10000.0) -> Dict[str, float]:
        """
        Evaluate a simple trading strategy based on model predictions.
        
        Args:
            initial_capital: Initial capital for the strategy
            
        Returns:
            Dictionary with strategy performance metrics
        """
        if not hasattr(self, 'predictions') or not hasattr(self, 'actuals'):
            logger.error("No predictions available. Run evaluate_on_test_data first.")
            return {}
        
        # Simple strategy: Long when prediction > 0, Cash when prediction <= 0
        positions = np.where(self.predictions > 0, 1, 0)  # 1 for long, 0 for cash
        
        # Calculate strategy returns
        strategy_returns = positions * self.actuals
        buy_hold_returns = self.actuals  # Buy and hold strategy
        
        # Calculate cumulative returns
        strategy_cumulative = (1 + strategy_returns).cumprod() * initial_capital
        buy_hold_cumulative = (1 + buy_hold_returns).cumprod() * initial_capital
        
        # Calculate performance metrics
        strategy_total_return = (strategy_cumulative[-1] / initial_capital) - 1
        buy_hold_total_return = (buy_hold_cumulative[-1] / initial_capital) - 1
        
        # Calculate annualized returns (assuming 252 trading days per year)
        n_days = len(strategy_returns)
        years = n_days / 252
        strategy_annual_return = (1 + strategy_total_return) ** (1 / years) - 1
        buy_hold_annual_return = (1 + buy_hold_total_return) ** (1 / years) - 1
        
        # Calculate maximum drawdown
        strategy_peak = np.maximum.accumulate(strategy_cumulative)
        strategy_drawdown = (strategy_cumulative - strategy_peak) / strategy_peak
        strategy_max_drawdown = np.min(strategy_drawdown)
        
        buy_hold_peak = np.maximum.accumulate(buy_hold_cumulative)
        buy_hold_drawdown = (buy_hold_cumulative - buy_hold_peak) / buy_hold_peak
        buy_hold_max_drawdown = np.min(buy_hold_drawdown)
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        strategy_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        buy_hold_sharpe = np.mean(buy_hold_returns) / np.std(buy_hold_returns) * np.sqrt(252)
        
        # Store results
        strategy_metrics = {
            'strategy_total_return': strategy_total_return,
            'buy_hold_total_return': buy_hold_total_return,
            'strategy_annual_return': strategy_annual_return,
            'buy_hold_annual_return': buy_hold_annual_return,
            'strategy_max_drawdown': strategy_max_drawdown,
            'buy_hold_max_drawdown': buy_hold_max_drawdown,
            'strategy_sharpe': strategy_sharpe,
            'buy_hold_sharpe': buy_hold_sharpe,
            'outperformance': strategy_total_return - buy_hold_total_return
        }
        
        logger.info(f"Strategy Total Return: {strategy_total_return:.4f}, Buy & Hold: {buy_hold_total_return:.4f}")
        logger.info(f"Strategy Sharpe Ratio: {strategy_sharpe:.4f}, Buy & Hold: {buy_hold_sharpe:.4f}")
        
        # Save metrics to file
        strategy_file = self.output_dir / f"{self.model_path.stem}_strategy.json"
        with open(strategy_file, 'w') as f:
            json.dump(strategy_metrics, f, indent=4)
        
        # Create strategy performance visualization
        plt.figure(figsize=(12, 6))
        plt.plot(strategy_cumulative, label='Model Strategy')
        plt.plot(buy_hold_cumulative, label='Buy & Hold')
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.title('Strategy Performance Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{self.model_path.stem}_strategy.png", dpi=300)
        plt.close()
        
        return strategy_metrics


def main():
    """
    Main function to run model evaluation from command line.
    """
    parser = argparse.ArgumentParser(description="Evaluate market prediction model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file (.pth)")
    parser.add_argument("--data", type=str, help="Path to test data file (.csv)")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Directory to save evaluation results")
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model, args.output)
    
    # If test data is provided, load and evaluate
    if args.data:
        # Load test data
        # This would be implemented based on the specific data format
        pass
    else:
        logger.warning("No test data provided. Using synthetic data for demonstration.")
        
        # Create synthetic test data for demonstration
        n_samples = 1000
        n_features = 64
        seq_length = 60
        
        # Generate random features and targets
        X = np.random.randn(n_samples, seq_length, n_features)
        y = np.random.randn(n_samples)
        
        # Create DataLoader
        test_dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        # Evaluate model
        metrics = evaluator.evaluate_on_test_data(test_loader)
        evaluator.create_visualizations()
        evaluator.evaluate_trading_strategy()
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
