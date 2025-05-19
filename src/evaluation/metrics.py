#!/usr/bin/env python
# Cosmic Market Oracle - Evaluation Metrics

"""
Evaluation metrics for the Cosmic Market Oracle.

This module provides specialized metrics for evaluating prediction models,
including regime-specific metrics, turning point metrics, and robustness metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, accuracy_score
)
import logging

# Configure logging
logger = logging.getLogger(__name__)


class PredictionMetrics:
    """
    Base class for prediction metrics.
    
    This class provides common metrics for evaluating prediction models,
    including regression metrics (RMSE, MAE, R²) and classification metrics
    (accuracy, precision, recall, F1).
    """
    
    def __init__(self):
        """Initialize the prediction metrics."""
        self.metrics = {}
        self.last_predictions = None
        self.last_actuals = None
        
    def calculate_regression_metrics(self, predictions: np.ndarray, 
                                   actuals: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            predictions: Predicted values
            actuals: Actual values
            
        Returns:
            Dictionary of regression metrics
        """
        # Store for later use
        self.last_predictions = predictions
        self.last_actuals = actuals
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # Calculate directional accuracy
        direction_pred = np.sign(np.diff(predictions))
        direction_actual = np.sign(np.diff(actuals))
        directional_accuracy = np.mean(direction_pred == direction_actual)
        
        # Calculate mean absolute percentage error
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
        
        # Store metrics
        self.metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "directional_accuracy": directional_accuracy,
            "mape": mape
        }
        
        return self.metrics
        
    def calculate_classification_metrics(self, predictions: np.ndarray, 
                                       actuals: np.ndarray,
                                       average: str = "weighted") -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            predictions: Predicted classes
            actuals: Actual classes
            average: Averaging method for multiclass metrics
            
        Returns:
            Dictionary of classification metrics
        """
        # Store for later use
        self.last_predictions = predictions
        self.last_actuals = actuals
        
        # Calculate metrics
        accuracy = accuracy_score(actuals, predictions)
        precision = precision_score(actuals, predictions, average=average, zero_division=0)
        recall = recall_score(actuals, predictions, average=average, zero_division=0)
        f1 = f1_score(actuals, predictions, average=average, zero_division=0)
        
        # Store metrics
        self.metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        return self.metrics
        
    def calculate_profit_metrics(self, predictions: np.ndarray, 
                               actuals: np.ndarray,
                               prices: np.ndarray,
                               transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Calculate profit-based metrics.
        
        Args:
            predictions: Predicted values (signals: 1 for buy, -1 for sell, 0 for hold)
            actuals: Actual returns
            prices: Asset prices
            transaction_cost: Transaction cost as a fraction of price
            
        Returns:
            Dictionary of profit metrics
        """
        # Store for later use
        self.last_predictions = predictions
        self.last_actuals = actuals
        
        # Calculate returns from signals
        signal_returns = np.zeros_like(actuals)
        positions = predictions[:-1]  # Position for each period
        
        # Calculate returns from positions
        for i in range(1, len(actuals)):
            # Return is position * actual return - transaction cost if position changed
            position_change = abs(positions[i-1] - positions[i-2]) if i > 1 else abs(positions[i-1])
            cost = position_change * transaction_cost
            signal_returns[i] = positions[i-1] * actuals[i] - cost
            
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + signal_returns) - 1
        
        # Calculate metrics
        total_return = cumulative_returns[-1]
        
        # Calculate Sharpe ratio (annualized)
        returns_std = np.std(signal_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = (np.mean(signal_returns) * 252) / returns_std if returns_std > 0 else 0
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(cumulative_returns + 1)
        drawdown = (cumulative_returns + 1) / peak - 1
        max_drawdown = np.min(drawdown)
        
        # Calculate win rate
        winning_trades = np.sum(signal_returns > 0)
        total_trades = np.sum(np.abs(np.diff(positions)) > 0) + (abs(positions[0]) > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profits = np.sum(signal_returns[signal_returns > 0])
        gross_losses = np.abs(np.sum(signal_returns[signal_returns < 0]))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Store metrics
        self.metrics = {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades
        }
        
        return self.metrics
        
    def get_metrics(self) -> Dict[str, float]:
        """
        Get the calculated metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
        
    def get_summary(self) -> str:
        """
        Get a summary of the metrics as a string.
        
        Returns:
            Summary string
        """
        if not self.metrics:
            return "No metrics calculated yet."
            
        summary = "Prediction Metrics Summary:\n"
        
        for name, value in self.metrics.items():
            summary += f"  {name}: {value:.4f}\n"
            
        return summary


class RegimeSpecificMetrics(PredictionMetrics):
    """
    Regime-specific metrics for evaluating prediction models.
    
    This class extends PredictionMetrics to provide metrics that are specific
    to different market regimes (bull, bear, sideways, volatile).
    """
    
    def __init__(self, regime_classifier: Optional[Callable] = None):
        """
        Initialize the regime-specific metrics.
        
        Args:
            regime_classifier: Function to classify regimes from market data
        """
        super().__init__()
        self.regime_classifier = regime_classifier
        self.regime_metrics = {}
        
    def set_regime_classifier(self, classifier: Callable):
        """
        Set the regime classifier function.
        
        Args:
            classifier: Function to classify regimes from market data
        """
        self.regime_classifier = classifier
        
    def calculate_regime_metrics(self, predictions: np.ndarray, 
                               actuals: np.ndarray,
                               market_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for different market regimes.
        
        Args:
            predictions: Predicted values
            actuals: Actual values
            market_data: Market data for regime classification
            
        Returns:
            Dictionary of regime-specific metrics
        """
        if self.regime_classifier is None:
            raise ValueError("Regime classifier not set. Use set_regime_classifier() first.")
            
        # Classify regimes
        regimes = self.regime_classifier(market_data)
        
        # Calculate metrics for each regime
        unique_regimes = np.unique(regimes)
        regime_metrics = {}
        
        for regime in unique_regimes:
            # Get indices for this regime
            indices = np.where(regimes == regime)[0]
            
            if len(indices) > 10:  # Ensure enough data points
                # Extract data for this regime
                regime_predictions = predictions[indices]
                regime_actuals = actuals[indices]
                
                # Calculate metrics
                metrics = self.calculate_regression_metrics(regime_predictions, regime_actuals)
                
                # Store metrics for this regime
                regime_metrics[regime] = metrics
                
        # Store regime metrics
        self.regime_metrics = regime_metrics
        
        return regime_metrics
        
    def get_regime_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get the calculated regime-specific metrics.
        
        Returns:
            Dictionary of regime-specific metrics
        """
        return self.regime_metrics
        
    def get_regime_summary(self) -> str:
        """
        Get a summary of the regime-specific metrics as a string.
        
        Returns:
            Summary string
        """
        if not self.regime_metrics:
            return "No regime-specific metrics calculated yet."
            
        summary = "Regime-Specific Metrics Summary:\n"
        
        for regime, metrics in self.regime_metrics.items():
            summary += f"\nRegime: {regime}\n"
            
            for name, value in metrics.items():
                summary += f"  {name}: {value:.4f}\n"
                
        return summary


class TurningPointMetrics(PredictionMetrics):
    """
    Metrics for evaluating prediction models at market turning points.
    
    This class extends PredictionMetrics to provide specialized metrics for
    evaluating how well a model predicts market turning points.
    """
    
    def __init__(self, window_size: int = 5, threshold: float = 0.02):
        """
        Initialize the turning point metrics.
        
        Args:
            window_size: Window size for turning point detection
            threshold: Threshold for turning point significance
        """
        super().__init__()
        self.window_size = window_size
        self.threshold = threshold
        self.turning_point_metrics = {}
        self.turning_points = None
        
    def detect_turning_points(self, prices: np.ndarray) -> np.ndarray:
        """
        Detect turning points in price series.
        
        Args:
            prices: Price series
            
        Returns:
            Boolean array indicating turning points
        """
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Initialize turning points array
        turning_points = np.zeros(len(prices), dtype=bool)
        
        # Detect local maxima and minima
        for i in range(self.window_size, len(prices) - self.window_size):
            # Check if this is a local maximum
            if (prices[i] == np.max(prices[i-self.window_size:i+self.window_size+1]) and
                np.abs(prices[i] / prices[i-self.window_size] - 1) >= self.threshold):
                turning_points[i] = True
                
            # Check if this is a local minimum
            elif (prices[i] == np.min(prices[i-self.window_size:i+self.window_size+1]) and
                  np.abs(prices[i] / prices[i-self.window_size] - 1) >= self.threshold):
                turning_points[i] = True
                
        return turning_points
        
    def calculate_turning_point_metrics(self, predictions: np.ndarray, 
                                      actuals: np.ndarray,
                                      prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics for market turning points.
        
        Args:
            predictions: Predicted values
            actuals: Actual values
            prices: Price series
            
        Returns:
            Dictionary of turning point metrics
        """
        # Detect turning points
        turning_points = self.detect_turning_points(prices)
        self.turning_points = turning_points
        
        # Check if there are any turning points
        if np.sum(turning_points) == 0:
            logger.warning("No turning points detected.")
            self.turning_point_metrics = {
                "turning_point_count": 0,
                "turning_point_rmse": np.nan,
                "turning_point_directional_accuracy": np.nan,
                "turning_point_lead_time": np.nan
            }
            return self.turning_point_metrics
            
        # Calculate metrics at turning points
        tp_indices = np.where(turning_points)[0]
        
        # Ensure we have valid indices
        valid_indices = [i for i in tp_indices if i < len(predictions) and i < len(actuals)]
        
        if not valid_indices:
            logger.warning("No valid turning points for evaluation.")
            self.turning_point_metrics = {
                "turning_point_count": 0,
                "turning_point_rmse": np.nan,
                "turning_point_directional_accuracy": np.nan,
                "turning_point_lead_time": np.nan
            }
            return self.turning_point_metrics
            
        # Extract predictions and actuals at turning points
        tp_predictions = predictions[valid_indices]
        tp_actuals = actuals[valid_indices]
        
        # Calculate RMSE at turning points
        tp_rmse = np.sqrt(mean_squared_error(tp_actuals, tp_predictions))
        
        # Calculate directional accuracy at turning points
        tp_direction_pred = np.sign(np.diff(predictions))[valid_indices[:-1]]
        tp_direction_actual = np.sign(np.diff(actuals))[valid_indices[:-1]]
        tp_directional_accuracy = np.mean(tp_direction_pred == tp_direction_actual)
        
        # Calculate lead time (how many periods before turning point the model predicted it)
        lead_times = []
        
        for tp_idx in valid_indices:
            # Look back up to 10 periods
            for i in range(1, min(11, tp_idx + 1)):
                # Check if prediction direction changed i periods before turning point
                if tp_idx - i >= 1 and np.sign(predictions[tp_idx - i] - predictions[tp_idx - i - 1]) != np.sign(predictions[tp_idx - i + 1] - predictions[tp_idx - i]):
                    lead_times.append(i)
                    break
                    
        # Calculate average lead time
        avg_lead_time = np.mean(lead_times) if lead_times else 0
        
        # Store metrics
        self.turning_point_metrics = {
            "turning_point_count": len(valid_indices),
            "turning_point_rmse": tp_rmse,
            "turning_point_directional_accuracy": tp_directional_accuracy,
            "turning_point_lead_time": avg_lead_time
        }
        
        return self.turning_point_metrics
        
    def get_turning_point_metrics(self) -> Dict[str, float]:
        """
        Get the calculated turning point metrics.
        
        Returns:
            Dictionary of turning point metrics
        """
        return self.turning_point_metrics
        
    def get_turning_points(self) -> Optional[np.ndarray]:
        """
        Get the detected turning points.
        
        Returns:
            Boolean array indicating turning points
        """
        return self.turning_points


class RobustnessMetrics(PredictionMetrics):
    """
    Metrics for evaluating model robustness.
    
    This class extends PredictionMetrics to provide metrics for evaluating
    how robust a model is to different market conditions and data perturbations.
    """
    
    def __init__(self):
        """Initialize the robustness metrics."""
        super().__init__()
        self.robustness_metrics = {}
        self.perturbation_results = {}
        
    def calculate_perturbation_robustness(self, model: Any, 
                                        X: np.ndarray, 
                                        y: np.ndarray,
                                        perturbation_levels: List[float] = [0.01, 0.05, 0.1],
                                        n_samples: int = 10) -> Dict[str, List[float]]:
        """
        Calculate robustness to input data perturbations.
        
        Args:
            model: Model with predict method
            X: Input features
            y: Target values
            perturbation_levels: Levels of perturbation to test
            n_samples: Number of samples for each perturbation level
            
        Returns:
            Dictionary of robustness metrics
        """
        # Initialize results
        results = {
            "perturbation_level": perturbation_levels,
            "rmse": [],
            "directional_accuracy": []
        }
        
        # Get baseline predictions
        baseline_pred = model.predict(X)
        baseline_rmse = np.sqrt(mean_squared_error(y, baseline_pred))
        
        # Test each perturbation level
        for level in perturbation_levels:
            level_rmse = []
            level_dir_acc = []
            
            for _ in range(n_samples):
                # Create perturbed data
                noise = np.random.normal(0, level, X.shape)
                X_perturbed = X + noise * X
                
                # Get predictions on perturbed data
                perturbed_pred = model.predict(X_perturbed)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y, perturbed_pred))
                
                # Calculate directional accuracy
                direction_pred = np.sign(np.diff(perturbed_pred))
                direction_actual = np.sign(np.diff(y))
                dir_acc = np.mean(direction_pred == direction_actual)
                
                level_rmse.append(rmse)
                level_dir_acc.append(dir_acc)
                
            # Store average results for this level
            results["rmse"].append(np.mean(level_rmse) / baseline_rmse)
            results["directional_accuracy"].append(np.mean(level_dir_acc))
            
        # Store results
        self.perturbation_results = results
        
        # Calculate overall robustness metrics
        rmse_robustness = 1.0 - np.mean([(r - 1.0) for r in results["rmse"]])
        dir_acc_robustness = np.mean(results["directional_accuracy"])
        
        self.robustness_metrics = {
            "rmse_robustness": rmse_robustness,
            "directional_accuracy_robustness": dir_acc_robustness,
            "overall_robustness": (rmse_robustness + dir_acc_robustness) / 2
        }
        
        return self.robustness_metrics
        
    def calculate_regime_robustness(self, regime_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate robustness across different market regimes.
        
        Args:
            regime_metrics: Dictionary of regime-specific metrics
            
        Returns:
            Dictionary of regime robustness metrics
        """
        if not regime_metrics:
            logger.warning("No regime metrics provided.")
            return {}
            
        # Extract metrics across regimes
        regime_rmse = [metrics["rmse"] for metrics in regime_metrics.values()]
        regime_dir_acc = [metrics["directional_accuracy"] for metrics in regime_metrics.values()]
        
        # Calculate coefficient of variation (lower is better for robustness)
        rmse_cv = np.std(regime_rmse) / np.mean(regime_rmse) if np.mean(regime_rmse) > 0 else float('inf')
        dir_acc_cv = np.std(regime_dir_acc) / np.mean(regime_dir_acc) if np.mean(regime_dir_acc) > 0 else float('inf')
        
        # Calculate min-max ratio (higher is better for robustness)
        rmse_min_max = np.min(regime_rmse) / np.max(regime_rmse) if np.max(regime_rmse) > 0 else 0
        dir_acc_min_max = np.min(regime_dir_acc) / np.max(regime_dir_acc) if np.max(regime_dir_acc) > 0 else 0
        
        # Calculate overall regime robustness
        rmse_robustness = 1.0 - rmse_cv
        dir_acc_robustness = 1.0 - dir_acc_cv
        
        # Store metrics
        regime_robustness = {
            "rmse_cv": rmse_cv,
            "directional_accuracy_cv": dir_acc_cv,
            "rmse_min_max_ratio": rmse_min_max,
            "directional_accuracy_min_max_ratio": dir_acc_min_max,
            "rmse_robustness": rmse_robustness,
            "directional_accuracy_robustness": dir_acc_robustness,
            "overall_regime_robustness": (rmse_robustness + dir_acc_robustness) / 2
        }
        
        # Update robustness metrics
        self.robustness_metrics.update(regime_robustness)
        
        return regime_robustness
        
    def get_robustness_metrics(self) -> Dict[str, float]:
        """
        Get the calculated robustness metrics.
        
        Returns:
            Dictionary of robustness metrics
        """
        return self.robustness_metrics
        
    def get_perturbation_results(self) -> Dict[str, List[float]]:
        """
        Get the perturbation test results.
        
        Returns:
            Dictionary of perturbation test results
        """
        return self.perturbation_results
