#!/usr/bin/env python
# Cosmic Market Oracle - Benchmark Comparison Framework

"""
Benchmark comparison framework for the Cosmic Market Oracle.

This module provides tools for comparing model performance against various
benchmarks, including statistical baselines and traditional trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.evaluation.metrics import PredictionMetrics

# Configure logging
logger = logging.getLogger(__name__)


class BaselineBenchmarks:
    """
    Statistical baseline benchmarks for prediction models.
    
    This class provides various statistical baselines for comparison,
    including naive forecasts, moving averages, and random walks.
    """
    
    def __init__(self):
        """Initialize the baseline benchmarks."""
        self.benchmarks = {}
        self.predictions = {}
        
    def generate_naive_forecast(self, data: pd.Series, lag: int = 1) -> np.ndarray:
        """
        Generate naive forecast (lag-n forecast).
        
        Args:
            data: Time series data
            lag: Lag value
            
        Returns:
            Naive forecast array
        """
        # Shift data by lag periods
        forecast = np.roll(data.values, lag)
        
        # Fill initial values
        forecast[:lag] = data.values[0]
        
        # Store benchmark
        name = f"naive_lag{lag}"
        self.benchmarks[name] = {
            "name": f"Naive Forecast (Lag {lag})",
            "type": "naive",
            "params": {"lag": lag}
        }
        self.predictions[name] = forecast
        
        return forecast
        
    def generate_moving_average(self, data: pd.Series, window: int = 5) -> np.ndarray:
        """
        Generate moving average forecast.
        
        Args:
            data: Time series data
            window: Window size
            
        Returns:
            Moving average forecast array
        """
        # Calculate moving average
        ma = data.rolling(window=window, min_periods=1).mean().values
        
        # Store benchmark
        name = f"ma{window}"
        self.benchmarks[name] = {
            "name": f"Moving Average (Window {window})",
            "type": "moving_average",
            "params": {"window": window}
        }
        self.predictions[name] = ma
        
        return ma
        
    def generate_exponential_smoothing(self, data: pd.Series, alpha: float = 0.3) -> np.ndarray:
        """
        Generate exponential smoothing forecast.
        
        Args:
            data: Time series data
            alpha: Smoothing factor
            
        Returns:
            Exponential smoothing forecast array
        """
        # Initialize forecast array
        values = data.values
        n = len(values)
        forecast = np.zeros(n)
        
        # Set initial value
        forecast[0] = values[0]
        
        # Calculate exponential smoothing
        for i in range(1, n):
            forecast[i] = alpha * values[i-1] + (1 - alpha) * forecast[i-1]
            
        # Store benchmark
        name = f"exp_smooth{alpha}"
        self.benchmarks[name] = {
            "name": f"Exponential Smoothing (Alpha {alpha})",
            "type": "exponential_smoothing",
            "params": {"alpha": alpha}
        }
        self.predictions[name] = forecast
        
        return forecast
        
    def generate_random_walk(self, data: pd.Series, drift: bool = False) -> np.ndarray:
        """
        Generate random walk forecast.
        
        Args:
            data: Time series data
            drift: Whether to include drift
            
        Returns:
            Random walk forecast array
        """
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Calculate drift term
        drift_term = 0
        if drift:
            drift_term = returns.mean()
            
        # Initialize forecast array
        values = data.values
        n = len(values)
        forecast = np.zeros(n)
        
        # Set initial value
        forecast[0] = values[0]
        
        # Calculate random walk with drift
        for i in range(1, n):
            forecast[i] = values[i-1] * (1 + drift_term)
            
        # Store benchmark
        name = "random_walk_drift" if drift else "random_walk"
        self.benchmarks[name] = {
            "name": "Random Walk with Drift" if drift else "Random Walk",
            "type": "random_walk",
            "params": {"drift": drift}
        }
        self.predictions[name] = forecast
        
        return forecast
        
    def generate_seasonal_naive(self, data: pd.Series, season_length: int = 5) -> np.ndarray:
        """
        Generate seasonal naive forecast.
        
        Args:
            data: Time series data
            season_length: Length of seasonality
            
        Returns:
            Seasonal naive forecast array
        """
        # Initialize forecast array
        values = data.values
        n = len(values)
        forecast = np.zeros(n)
        
        # Fill initial values
        forecast[:season_length] = values[:season_length]
        
        # Calculate seasonal naive forecast
        for i in range(season_length, n):
            forecast[i] = values[i - season_length]
            
        # Store benchmark
        name = f"seasonal_naive{season_length}"
        self.benchmarks[name] = {
            "name": f"Seasonal Naive (Season {season_length})",
            "type": "seasonal_naive",
            "params": {"season_length": season_length}
        }
        self.predictions[name] = forecast
        
        return forecast
        
    def generate_all_benchmarks(self, data: pd.Series) -> Dict[str, np.ndarray]:
        """
        Generate all benchmark forecasts.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary of benchmark forecasts
        """
        # Generate benchmarks
        self.generate_naive_forecast(data, lag=1)
        self.generate_moving_average(data, window=5)
        self.generate_moving_average(data, window=10)
        self.generate_exponential_smoothing(data, alpha=0.3)
        self.generate_random_walk(data, drift=False)
        self.generate_random_walk(data, drift=True)
        self.generate_seasonal_naive(data, season_length=5)
        
        return self.predictions
        
    def evaluate_benchmarks(self, actuals: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate benchmark forecasts.
        
        Args:
            actuals: Actual values
            
        Returns:
            Dictionary of benchmark evaluation results
        """
        # Initialize metrics calculator
        metrics_calculator = PredictionMetrics()
        
        # Initialize results
        results = {}
        
        # Evaluate each benchmark
        for name, predictions in self.predictions.items():
            # Calculate metrics
            metrics = metrics_calculator.calculate_regression_metrics(
                predictions, actuals.values
            )
            
            # Store results
            results[name] = {
                "name": self.benchmarks[name]["name"],
                "type": self.benchmarks[name]["type"],
                "params": self.benchmarks[name]["params"],
                "metrics": metrics
            }
            
        return results
        
    def plot_benchmark_comparison(self, actuals: pd.Series, 
                                metric_name: str = "rmse",
                                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot benchmark comparison.
        
        Args:
            actuals: Actual values
            metric_name: Metric name for comparison
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Evaluate benchmarks
        results = self.evaluate_benchmarks(actuals)
        
        # Extract metric values
        names = []
        values = []
        
        for name, result in results.items():
            names.append(result["name"])
            values.append(result["metrics"][metric_name])
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bar chart
        ax.bar(names, values)
        
        # Add labels and title
        ax.set_xlabel("Benchmark")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Benchmark Comparison ({metric_name})")
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


class TradingStrategyBenchmarks:
    """
    Trading strategy benchmarks for prediction models.
    
    This class provides various trading strategy benchmarks for comparison,
    including buy-and-hold, moving average crossover, and mean reversion.
    """
    
    def __init__(self):
        """Initialize the trading strategy benchmarks."""
        self.benchmarks = {}
        self.signals = {}
        self.returns = {}
        
    def generate_buy_hold(self, prices: pd.Series) -> np.ndarray:
        """
        Generate buy-and-hold strategy signals.
        
        Args:
            prices: Price series
            
        Returns:
            Strategy signals array (1 for long position)
        """
        # Always hold long position
        signals = np.ones(len(prices))
        
        # Store benchmark
        name = "buy_hold"
        self.benchmarks[name] = {
            "name": "Buy and Hold",
            "type": "buy_hold",
            "params": {}
        }
        self.signals[name] = signals
        
        return signals
        
    def generate_ma_crossover(self, prices: pd.Series, 
                            short_window: int = 5, 
                            long_window: int = 20) -> np.ndarray:
        """
        Generate moving average crossover strategy signals.
        
        Args:
            prices: Price series
            short_window: Short moving average window
            long_window: Long moving average window
            
        Returns:
            Strategy signals array (1 for long, -1 for short, 0 for no position)
        """
        # Calculate moving averages
        short_ma = prices.rolling(window=short_window, min_periods=1).mean()
        long_ma = prices.rolling(window=long_window, min_periods=1).mean()
        
        # Generate signals
        signals = np.zeros(len(prices))
        
        # Long when short MA > long MA
        signals[short_ma > long_ma] = 1
        
        # Short when short MA < long MA
        signals[short_ma < long_ma] = -1
        
        # No signal for initial period
        signals[:long_window] = 0
        
        # Store benchmark
        name = f"ma_cross_{short_window}_{long_window}"
        self.benchmarks[name] = {
            "name": f"MA Crossover ({short_window}/{long_window})",
            "type": "ma_crossover",
            "params": {"short_window": short_window, "long_window": long_window}
        }
        self.signals[name] = signals
        
        return signals
        
    def generate_mean_reversion(self, prices: pd.Series, 
                              window: int = 20,
                              threshold: float = 1.0) -> np.ndarray:
        """
        Generate mean reversion strategy signals.
        
        Args:
            prices: Price series
            window: Lookback window
            threshold: Z-score threshold
            
        Returns:
            Strategy signals array (1 for long, -1 for short, 0 for no position)
        """
        # Calculate rolling mean and std
        rolling_mean = prices.rolling(window=window, min_periods=1).mean()
        rolling_std = prices.rolling(window=window, min_periods=1).std()
        
        # Calculate z-scores
        z_scores = (prices - rolling_mean) / rolling_std
        
        # Generate signals
        signals = np.zeros(len(prices))
        
        # Long when z-score < -threshold (oversold)
        signals[z_scores < -threshold] = 1
        
        # Short when z-score > threshold (overbought)
        signals[z_scores > threshold] = -1
        
        # No signal for initial period
        signals[:window] = 0
        
        # Store benchmark
        name = f"mean_rev_{window}_{threshold}"
        self.benchmarks[name] = {
            "name": f"Mean Reversion (Window {window}, Threshold {threshold})",
            "type": "mean_reversion",
            "params": {"window": window, "threshold": threshold}
        }
        self.signals[name] = signals
        
        return signals
        
    def generate_momentum(self, prices: pd.Series, 
                        window: int = 20,
                        threshold: float = 0.02) -> np.ndarray:
        """
        Generate momentum strategy signals.
        
        Args:
            prices: Price series
            window: Lookback window
            threshold: Return threshold
            
        Returns:
            Strategy signals array (1 for long, -1 for short, 0 for no position)
        """
        # Calculate momentum (return over window)
        momentum = prices.pct_change(window)
        
        # Generate signals
        signals = np.zeros(len(prices))
        
        # Long when momentum > threshold
        signals[momentum > threshold] = 1
        
        # Short when momentum < -threshold
        signals[momentum < -threshold] = -1
        
        # No signal for initial period
        signals[:window] = 0
        
        # Store benchmark
        name = f"momentum_{window}_{threshold}"
        self.benchmarks[name] = {
            "name": f"Momentum (Window {window}, Threshold {threshold})",
            "type": "momentum",
            "params": {"window": window, "threshold": threshold}
        }
        self.signals[name] = signals
        
        return signals
        
    def generate_all_benchmarks(self, prices: pd.Series) -> Dict[str, np.ndarray]:
        """
        Generate all benchmark strategy signals.
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary of benchmark signals
        """
        # Generate benchmarks
        self.generate_buy_hold(prices)
        self.generate_ma_crossover(prices, short_window=5, long_window=20)
        self.generate_ma_crossover(prices, short_window=10, long_window=50)
        self.generate_mean_reversion(prices, window=20, threshold=1.0)
        self.generate_momentum(prices, window=20, threshold=0.02)
        
        return self.signals
        
    def calculate_returns(self, prices: pd.Series, 
                        transaction_cost: float = 0.001) -> Dict[str, pd.Series]:
        """
        Calculate returns for all benchmark strategies.
        
        Args:
            prices: Price series
            transaction_cost: Transaction cost as fraction of price
            
        Returns:
            Dictionary of strategy returns
        """
        # Calculate price returns
        price_returns = prices.pct_change().fillna(0)
        
        # Calculate returns for each strategy
        for name, signals in self.signals.items():
            # Calculate position changes
            position_changes = np.diff(signals, prepend=0)
            
            # Calculate transaction costs
            costs = np.abs(position_changes) * transaction_cost
            
            # Calculate strategy returns
            strategy_returns = signals[:-1] * price_returns.values[1:] - costs[1:]
            
            # Store returns
            self.returns[name] = pd.Series(
                strategy_returns, 
                index=price_returns.index[1:]
            )
            
        return self.returns
        
    def evaluate_strategies(self, prices: pd.Series,
                          transaction_cost: float = 0.001) -> Dict[str, Dict[str, float]]:
        """
        Evaluate benchmark strategies.
        
        Args:
            prices: Price series
            transaction_cost: Transaction cost as fraction of price
            
        Returns:
            Dictionary of strategy evaluation results
        """
        # Calculate returns
        if not self.returns:
            self.calculate_returns(prices, transaction_cost)
            
        # Initialize results
        results = {}
        
        # Evaluate each strategy
        for name, returns in self.returns.items():
            # Calculate metrics
            total_return = (1 + returns).prod() - 1
            annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Calculate maximum drawdown
            cum_returns = (1 + returns).cumprod()
            peak = cum_returns.expanding().max()
            drawdown = (cum_returns / peak - 1)
            max_drawdown = drawdown.min()
            
            # Calculate win rate
            win_rate = (returns > 0).mean()
            
            # Store results
            results[name] = {
                "name": self.benchmarks[name]["name"],
                "type": self.benchmarks[name]["type"],
                "params": self.benchmarks[name]["params"],
                "metrics": {
                    "total_return": total_return,
                    "annual_return": annual_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate
                }
            }
            
        return results
        
    def plot_equity_curves(self, prices: pd.Series,
                         transaction_cost: float = 0.001,
                         figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot equity curves for benchmark strategies.
        
        Args:
            prices: Price series
            transaction_cost: Transaction cost as fraction of price
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Calculate returns
        if not self.returns:
            self.calculate_returns(prices, transaction_cost)
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot equity curves
        for name, returns in self.returns.items():
            equity = (1 + returns).cumprod()
            ax.plot(equity.index, equity, label=self.benchmarks[name]["name"])
            
        # Add labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (Starting at 1.0)")
        ax.set_title("Benchmark Strategy Equity Curves")
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


class BenchmarkComparer:
    """
    Comparer for model performance against benchmarks.
    
    This class provides tools for comparing model performance against various
    benchmarks, including statistical baselines and trading strategies.
    """
    
    def __init__(self):
        """Initialize the benchmark comparer."""
        self.baseline_benchmarks = BaselineBenchmarks()
        self.strategy_benchmarks = TradingStrategyBenchmarks()
        self.model_predictions = None
        self.model_signals = None
        self.model_returns = None
        
    def set_model_predictions(self, predictions: np.ndarray):
        """
        Set model predictions for comparison.
        
        Args:
            predictions: Model predictions
        """
        self.model_predictions = predictions
        
    def set_model_signals(self, signals: np.ndarray):
        """
        Set model trading signals for comparison.
        
        Args:
            signals: Model trading signals
        """
        self.model_signals = signals
        
    def compare_predictions(self, actuals: pd.Series, 
                          metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare model predictions against baseline benchmarks.
        
        Args:
            actuals: Actual values
            metrics: List of metrics for comparison
            
        Returns:
            Dictionary of comparison results
        """
        if self.model_predictions is None:
            logger.warning("Model predictions not set. Use set_model_predictions() first.")
            return {}
            
        # Default metrics if not specified
        if metrics is None:
            metrics = ["rmse", "mae", "directional_accuracy"]
            
        # Generate baseline benchmarks
        self.baseline_benchmarks.generate_all_benchmarks(actuals)
        
        # Evaluate baseline benchmarks
        benchmark_results = self.baseline_benchmarks.evaluate_benchmarks(actuals)
        
        # Evaluate model
        metrics_calculator = PredictionMetrics()
        model_metrics = metrics_calculator.calculate_regression_metrics(
            self.model_predictions, actuals.values
        )
        
        # Initialize comparison results
        comparison = {
            "model": {
                "name": "Model",
                "metrics": {metric: model_metrics[metric] for metric in metrics}
            }
        }
        
        # Add benchmark results
        for name, result in benchmark_results.items():
            comparison[name] = {
                "name": result["name"],
                "metrics": {metric: result["metrics"][metric] for metric in metrics}
            }
            
        return comparison
        
    def compare_strategies(self, prices: pd.Series,
                         transaction_cost: float = 0.001,
                         metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare model trading strategy against benchmark strategies.
        
        Args:
            prices: Price series
            transaction_cost: Transaction cost as fraction of price
            metrics: List of metrics for comparison
            
        Returns:
            Dictionary of comparison results
        """
        if self.model_signals is None:
            logger.warning("Model signals not set. Use set_model_signals() first.")
            return {}
            
        # Default metrics if not specified
        if metrics is None:
            metrics = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
            
        # Generate strategy benchmarks
        self.strategy_benchmarks.generate_all_benchmarks(prices)
        
        # Evaluate strategy benchmarks
        benchmark_results = self.strategy_benchmarks.evaluate_strategies(
            prices, transaction_cost
        )
        
        # Calculate model returns
        price_returns = prices.pct_change().fillna(0)
        position_changes = np.diff(self.model_signals, prepend=0)
        costs = np.abs(position_changes) * transaction_cost
        model_returns = self.model_signals[:-1] * price_returns.values[1:] - costs[1:]
        self.model_returns = pd.Series(model_returns, index=price_returns.index[1:])
        
        # Calculate model metrics
        total_return = (1 + self.model_returns).prod() - 1
        annual_return = (1 + self.model_returns).prod() ** (252 / len(self.model_returns)) - 1
        volatility = self.model_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cum_returns = (1 + self.model_returns).cumprod()
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns / peak - 1)
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        win_rate = (self.model_returns > 0).mean()
        
        # Initialize comparison results
        comparison = {
            "model": {
                "name": "Model Strategy",
                "metrics": {
                    "total_return": total_return,
                    "annual_return": annual_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate
                }
            }
        }
        
        # Add benchmark results
        for name, result in benchmark_results.items():
            comparison[name] = {
                "name": result["name"],
                "metrics": {metric: result["metrics"][metric] for metric in metrics}
            }
            
        return comparison
        
    def plot_prediction_comparison(self, actuals: pd.Series,
                                 metric: str = "rmse",
                                 figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot prediction comparison against benchmarks.
        
        Args:
            actuals: Actual values
            metric: Metric for comparison
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Compare predictions
        comparison = self.compare_predictions(actuals, [metric])
        
        # Extract metric values
        names = []
        values = []
        
        for name, result in comparison.items():
            names.append(result["name"])
            values.append(result["metrics"][metric])
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bar chart
        bars = ax.bar(names, values)
        
        # Highlight model bar
        bars[0].set_color("green")
        
        # Add labels and title
        ax.set_xlabel("Model / Benchmark")
        ax.set_ylabel(metric)
        ax.set_title(f"Prediction Comparison ({metric})")
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
        
    def plot_strategy_comparison(self, prices: pd.Series,
                               metric: str = "sharpe_ratio",
                               transaction_cost: float = 0.001,
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot strategy comparison against benchmarks.
        
        Args:
            prices: Price series
            metric: Metric for comparison
            transaction_cost: Transaction cost as fraction of price
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Compare strategies
        comparison = self.compare_strategies(prices, transaction_cost, [metric])
        
        # Extract metric values
        names = []
        values = []
        
        for name, result in comparison.items():
            names.append(result["name"])
            values.append(result["metrics"][metric])
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bar chart
        bars = ax.bar(names, values)
        
        # Highlight model bar
        bars[0].set_color("green")
        
        # Add labels and title
        ax.set_xlabel("Strategy")
        ax.set_ylabel(metric)
        ax.set_title(f"Strategy Comparison ({metric})")
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
        
    def plot_equity_curve_comparison(self, prices: pd.Series,
                                   transaction_cost: float = 0.001,
                                   figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot equity curve comparison against benchmark strategies.
        
        Args:
            prices: Price series
            transaction_cost: Transaction cost as fraction of price
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Compare strategies
        self.compare_strategies(prices, transaction_cost)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot model equity curve
        model_equity = (1 + self.model_returns).cumprod()
        ax.plot(model_equity.index, model_equity, label="Model Strategy", 
              linewidth=2, color="green")
        
        # Plot benchmark equity curves
        for name, returns in self.strategy_benchmarks.returns.items():
            equity = (1 + returns).cumprod()
            ax.plot(equity.index, equity, label=self.strategy_benchmarks.benchmarks[name]["name"],
                  alpha=0.7)
            
        # Add labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (Starting at 1.0)")
        ax.set_title("Equity Curve Comparison")
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
