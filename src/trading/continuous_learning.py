"""
Continuous Learning and Evolution Framework for the Cosmic Market Oracle.

This module provides tools for:
1. Automated model retraining based on performance triggers
2. Online learning for incremental model updates
3. Concept drift detection for market regime changes
4. Model versioning and management
5. A/B testing framework for strategy comparison
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from src.trading.strategy_framework import BaseStrategy
from src.trading.backtest import BacktestEngine


class ModelVersionManager:
    """Manages versioning for trading strategy models."""
    
    def __init__(self, base_dir: str = "models"):
        """
        Initialize the model version manager.
        
        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.version_log_path = os.path.join(base_dir, "version_log.json")
        self.version_log = self._load_version_log()
    
    def _load_version_log(self) -> Dict:
        """Load the version log from disk."""
        if os.path.exists(self.version_log_path):
            with open(self.version_log_path, 'r') as f:
                return json.load(f)
        return {"models": [], "current_version": None}
    
    def _save_version_log(self):
        """Save the version log to disk."""
        with open(self.version_log_path, 'w') as f:
            json.dump(self.version_log, f, indent=2)
    
    def save_model(self, model: BaseStrategy, metrics: Dict, 
                  description: str = "") -> str:
        """
        Save a model with its metadata.
        
        Args:
            model: The strategy model to save
            metrics: Performance metrics for the model
            description: Optional description of the model
            
        Returns:
            Version ID of the saved model
        """
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model.name.replace(' ', '_')}_{timestamp}"
        
        # Create model directory
        model_dir = os.path.join(self.base_dir, version_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            "version_id": version_id,
            "model_name": model.name,
            "timestamp": timestamp,
            "metrics": metrics,
            "description": description,
            "parameters": model.__dict__
        }
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update version log
        self.version_log["models"].append({
            "version_id": version_id,
            "model_name": model.name,
            "timestamp": timestamp,
            "metrics_summary": {k: metrics[k] for k in 
                              ["sharpe_ratio", "total_return", "max_drawdown"] 
                              if k in metrics},
            "description": description
        })
        
        self.version_log["current_version"] = version_id
        self._save_version_log()
        
        return version_id
    
    def load_model(self, version_id: Optional[str] = None) -> Tuple[BaseStrategy, Dict]:
        """
        Load a model and its metadata.
        
        Args:
            version_id: Version ID to load, or None for the current version
            
        Returns:
            Tuple of (model, metadata)
        """
        if version_id is None:
            version_id = self.version_log["current_version"]
            
        if version_id is None:
            raise ValueError("No models have been saved yet")
            
        model_dir = os.path.join(self.base_dir, version_id)
        model_path = os.path.join(model_dir, "model.joblib")
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise ValueError(f"Model version {version_id} not found")
            
        model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return model, metadata
    
    def list_versions(self, model_name: Optional[str] = None) -> List[Dict]:
        """
        List available model versions.
        
        Args:
            model_name: Optional filter by model name
            
        Returns:
            List of version metadata
        """
        versions = self.version_log["models"]
        
        if model_name:
            versions = [v for v in versions if v["model_name"] == model_name]
            
        return versions
    
    def set_current_version(self, version_id: str):
        """Set the current active model version."""
        versions = [v["version_id"] for v in self.version_log["models"]]
        
        if version_id not in versions:
            raise ValueError(f"Model version {version_id} not found")
            
        self.version_log["current_version"] = version_id
        self._save_version_log()


class ConceptDriftDetector:
    """Detects concept drift in market data to identify regime changes."""
    
    def __init__(self, window_size: int = 60, threshold: float = 0.05):
        """
        Initialize the concept drift detector.
        
        Args:
            window_size: Size of the reference and detection windows
            threshold: P-value threshold for drift detection
        """
        self.window_size = window_size
        self.threshold = threshold
        self.reference_data = None
        self.drift_history = []
    
    def update_reference(self, data: pd.DataFrame):
        """Update the reference data window."""
        self.reference_data = data.iloc[-self.window_size:].copy()
    
    def detect_drift(self, new_data: pd.DataFrame) -> bool:
        """
        Detect if concept drift has occurred.
        
        Args:
            new_data: New market data to check for drift
            
        Returns:
            True if drift is detected, False otherwise
        """
        if self.reference_data is None:
            self.update_reference(new_data)
            return False
        
        # Ensure we have enough data
        if len(new_data) < self.window_size:
            return False
        
        # Get the detection window
        detection_window = new_data.iloc[-self.window_size:].copy()
        
        # Calculate drift for each numeric column
        drift_detected = False
        drift_details = {}
        
        for col in self.reference_data.select_dtypes(include=[np.number]).columns:
            ref_values = self.reference_data[col].dropna().values
            new_values = detection_window[col].dropna().values
            
            if len(ref_values) < 10 or len(new_values) < 10:
                continue
                
            # Use Kolmogorov-Smirnov test to detect distribution changes
            ks_stat, p_value = ks_2samp(ref_values, new_values)
            
            drift_details[col] = {
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "drift_detected": p_value < self.threshold
            }
            
            if p_value < self.threshold:
                drift_detected = True
        
        # Record drift event if detected
        if drift_detected:
            self.drift_history.append({
                "timestamp": new_data.index[-1],
                "details": drift_details
            })
            
            # Update reference after drift
            self.update_reference(new_data)
            
        return drift_detected
    
    def plot_drift_history(self, save_path: Optional[str] = None):
        """
        Plot the history of detected drift events.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.drift_history:
            print("No drift events detected yet")
            return
            
        timestamps = [event["timestamp"] for event in self.drift_history]
        
        plt.figure(figsize=(12, 6))
        plt.scatter(timestamps, [1] * len(timestamps), marker='v', s=100, color='red')
        plt.yticks([])
        plt.title("Market Regime Change Detection")
        plt.xlabel("Date")
        plt.ylabel("Regime Changes")
        
        for i, ts in enumerate(timestamps):
            plt.axvline(ts, color='lightgray', linestyle='--', alpha=0.7)
            
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class OnlineLearner:
    """Provides online learning capabilities for incremental model updates."""
    
    def __init__(self, strategy: BaseStrategy, 
                learning_rate: float = 0.1,
                version_manager: Optional[ModelVersionManager] = None):
        """
        Initialize the online learner.
        
        Args:
            strategy: The strategy to update
            learning_rate: Learning rate for parameter updates
            version_manager: Optional version manager for model tracking
        """
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.version_manager = version_manager
        self.update_history = []
    
    def update_model(self, new_data: pd.DataFrame, 
                   planetary_data: pd.DataFrame,
                   actual_returns: pd.Series) -> Dict:
        """
        Update the model based on new data.
        
        Args:
            new_data: New market data
            planetary_data: New planetary data
            actual_returns: Actual market returns
            
        Returns:
            Update metrics
        """
        # Generate signals using current model
        signals = self.strategy.generate_signals(new_data, planetary_data)
        
        # Calculate prediction error
        predicted_returns = signals["signal"] * signals["strength"]
        error = actual_returns - predicted_returns
        
        # Update model parameters based on error
        self._update_parameters(error)
        
        # Calculate update metrics
        mse = (error ** 2).mean()
        mae = error.abs().mean()
        
        update_info = {
            "timestamp": datetime.now(),
            "data_range": (new_data.index[0], new_data.index[-1]),
            "mse": mse,
            "mae": mae,
            "learning_rate": self.learning_rate
        }
        
        self.update_history.append(update_info)
        
        # Save updated model if version manager is available
        if self.version_manager:
            version_id = self.version_manager.save_model(
                self.strategy,
                {"mse": mse, "mae": mae},
                f"Online update at {datetime.now()}"
            )
            update_info["version_id"] = version_id
            
        return update_info
    
    def _update_parameters(self, error: pd.Series):
        """
        Update model parameters based on prediction error.
        
        Args:
            error: Prediction error series
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated
        # online learning algorithms
        
        # Example: Adjust signal threshold based on error
        if hasattr(self.strategy, "min_signal_strength"):
            avg_error = error.abs().mean()
            
            # If error is high, increase the threshold to be more selective
            if avg_error > 0.1:
                self.strategy.min_signal_strength = min(
                    0.9, 
                    self.strategy.min_signal_strength + self.learning_rate * avg_error
                )
            # If error is low, decrease the threshold to capture more signals
            else:
                self.strategy.min_signal_strength = max(
                    0.1,
                    self.strategy.min_signal_strength - self.learning_rate * (0.1 - avg_error)
                )


class PerformanceMonitor:
    """Monitors strategy performance and triggers retraining when needed."""
    
    def __init__(self, 
               strategy: BaseStrategy,
               backtest_engine: BacktestEngine,
               retrain_callback: Callable,
               market_data: pd.DataFrame,
               planetary_data: pd.DataFrame,
               window_size: int = 30,
               sharpe_threshold: float = 0.5,
               drawdown_threshold: float = 0.1,
               check_frequency: int = 7):
        """
        Initialize the performance monitor.
        
        Args:
            strategy: The strategy to monitor
            backtest_engine: Backtest engine for evaluation
            retrain_callback: Function to call when retraining is needed
            market_data: Historical market data
            planetary_data: Historical planetary data
            window_size: Window size for performance evaluation
            sharpe_threshold: Minimum acceptable Sharpe ratio
            drawdown_threshold: Maximum acceptable drawdown
            check_frequency: How often to check performance (in days)
        """
        self.strategy = strategy
        self.backtest_engine = backtest_engine
        self.retrain_callback = retrain_callback
        self.market_data = market_data
        self.planetary_data = planetary_data
        self.window_size = window_size
        self.sharpe_threshold = sharpe_threshold
        self.drawdown_threshold = drawdown_threshold
        self.check_frequency = check_frequency
        self.last_check = None
        self.performance_history = []
    
    def check_performance(self, current_date: datetime) -> Dict:
        """
        Check strategy performance and trigger retraining if needed.
        
        Args:
            current_date: Current date for the check
            
        Returns:
            Performance metrics and retraining decision
        """
        # Skip if we checked recently
        if (self.last_check and 
            (current_date - self.last_check).days < self.check_frequency):
            return {"status": "skipped", "reason": "Too soon since last check"}
            
        self.last_check = current_date
        
        # Get recent data for evaluation
        end_idx = self.market_data.index.get_indexer([current_date], method='pad')[0]
        start_idx = max(0, end_idx - self.window_size)
        
        recent_market_data = self.market_data.iloc[start_idx:end_idx+1]
        recent_planetary_data = self.planetary_data.iloc[start_idx:end_idx+1]
        
        # Run backtest on recent data
        results = self.backtest_engine.run_backtest(
            self.strategy,
            recent_market_data,
            recent_planetary_data,
            "PERFORMANCE_CHECK"
        )
        
        metrics = results["metrics"]
        
        # Record performance
        performance_record = {
            "date": current_date,
            "metrics": metrics,
            "window_size": self.window_size
        }
        
        self.performance_history.append(performance_record)
        
        # Check if retraining is needed
        retrain_needed = False
        retrain_reason = []
        
        if metrics["sharpe_ratio"] < self.sharpe_threshold:
            retrain_needed = True
            retrain_reason.append(f"Sharpe ratio ({metrics['sharpe_ratio']:.2f}) below threshold ({self.sharpe_threshold})")
            
        if abs(metrics["max_drawdown"]) > self.drawdown_threshold:
            retrain_needed = True
            retrain_reason.append(f"Drawdown ({metrics['max_drawdown']:.2f}) exceeds threshold ({self.drawdown_threshold})")
            
        # Trigger retraining if needed
        if retrain_needed:
            self.retrain_callback(
                self.strategy,
                self.market_data,
                self.planetary_data,
                metrics
            )
            
        return {
            "status": "retrain" if retrain_needed else "ok",
            "reason": ", ".join(retrain_reason) if retrain_needed else "Performance acceptable",
            "metrics": metrics
        }
    
    def plot_performance_history(self, metric: str = "sharpe_ratio", 
                              save_path: Optional[str] = None):
        """
        Plot the history of performance metrics.
        
        Args:
            metric: The metric to plot
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        dates = [h['date'] for h in self.performance_history]
        values = [h['metrics'][metric] for h in self.performance_history]
        
        plt.plot(dates, values)
        plt.title(f"Performance History - {metric}")
            plt.axhline(-self.drawdown_threshold, color='red', linestyle='--',
                      label=f"Threshold ({-self.drawdown_threshold})")
            
        plt.title(f"Strategy {metric.replace('_', ' ').title()} Over Time")
        plt.xlabel("Date")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class ABTestingFramework:
    """Framework for A/B testing of trading strategies."""
    
    def __init__(self, 
               backtest_engine: BacktestEngine,
               market_data: pd.DataFrame,
               planetary_data: pd.DataFrame,
               test_duration: int = 30,
               confidence_level: float = 0.95):
        """
        Initialize the A/B testing framework.
        
        Args:
            backtest_engine: Backtest engine for evaluation
            market_data: Historical market data
            planetary_data: Historical planetary data
            test_duration: Duration of A/B tests in days
            confidence_level: Statistical confidence level
        """
        self.backtest_engine = backtest_engine
        self.market_data = market_data
        self.planetary_data = planetary_data
        self.test_duration = test_duration
        self.confidence_level = confidence_level
        self.test_history = []
    
    def run_ab_test(self, 
                  strategy_a: BaseStrategy, 
                  strategy_b: BaseStrategy,
                  start_date: Optional[datetime] = None,
                  metrics: List[str] = ["sharpe_ratio", "total_return", "max_drawdown"]) -> Dict:
        """
        Run an A/B test comparing two strategies.
        
        Args:
            strategy_a: First strategy (control)
            strategy_b: Second strategy (variant)
            start_date: Start date for the test
            metrics: Metrics to compare
            
        Returns:
            Test results and statistical significance
        """
        if start_date is None:
            start_date = self.market_data.index[-self.test_duration-1]
            
        end_date = start_date + timedelta(days=self.test_duration)
        
        # Get test data
        mask = (self.market_data.index >= start_date) & (self.market_data.index <= end_date)
        test_market_data = self.market_data[mask]
        test_planetary_data = self.planetary_data[mask]
        
        # Run backtests
        results_a = self.backtest_engine.run_backtest(
            strategy_a,
            test_market_data,
            test_planetary_data,
            "A"
        )
        
        results_b = self.backtest_engine.run_backtest(
            strategy_b,
            test_market_data,
            test_planetary_data,
            "B"
        )
        
        # Compare results
        comparison = {}
        for metric in metrics:
            value_a = results_a["metrics"].get(metric, 0)
            value_b = results_b["metrics"].get(metric, 0)
            
            comparison[metric] = {
                "A": value_a,
                "B": value_b,
                "difference": value_b - value_a,
                "percent_change": ((value_b - value_a) / abs(value_a) * 100) if value_a != 0 else float('inf')
            }
            
        # Perform statistical tests on returns
        returns_a = results_a["results"]["returns"]
        returns_b = results_b["results"]["returns"]
        
        t_stat, p_value = self._calculate_significance(returns_a, returns_b)
        
        is_significant = p_value < (1 - self.confidence_level)
        winner = "B" if comparison["sharpe_ratio"]["difference"] > 0 else "A"
        
        # Record test results
        test_record = {
            "start_date": start_date,
            "end_date": end_date,
            "strategy_a": strategy_a.name,
            "strategy_b": strategy_b.name,
            "comparison": comparison,
            "statistical_tests": {
                "t_statistic": t_stat,
                "p_value": p_value,
                "is_significant": is_significant
            },
            "winner": winner if is_significant else "Inconclusive"
        }
        
        self.test_history.append(test_record)
        
        return test_record
    
    def _calculate_significance(self, returns_a: pd.Series, 
                             returns_b: pd.Series) -> Tuple[float, float]:
        """
        Calculate statistical significance of returns difference.
        
        Args:
            returns_a: Returns from strategy A
            returns_b: Returns from strategy B
            
        Returns:
            Tuple of (t-statistic, p-value)
        """
        from scipy import stats
        
        # Calculate t-test for independent samples
        t_stat, p_value = stats.ttest_ind(
            returns_a.dropna(), 
            returns_b.dropna(),
            equal_var=False  # Welch's t-test
        )
        
        return t_stat, p_value
    
    def plot_test_results(self, test_index: int = -1, 
                        save_path: Optional[str] = None):
        """
        Plot the results of an A/B test.
        
        Args:
            test_index: Index of the test to plot
            save_path: Optional path to save the plot
        """
        if not self.test_history:
            print("No test history available")
            return
            
        test = self.test_history[test_index]
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot equity curves
        axs[0].set_title(f"Equity Curves: {test['strategy_a']} vs {test['strategy_b']}")
        
        # Get the results from the backtest engine
        results_a = self.backtest_engine.results.get(test['strategy_a'])
        results_b = self.backtest_engine.results.get(test['strategy_b'])
        
        if results_a and results_b:
            axs[0].plot(results_a["equity"], label="Strategy A")
            axs[0].plot(results_b["equity"], label="Strategy B")
            axs[0].legend()
            axs[0].set_xlabel("Date")
            axs[0].set_ylabel("Equity")
            
        # Plot metric comparison
        metrics = list(test["comparison"].keys())
        values_a = [test["comparison"][m]["A"] for m in metrics]
        values_b = [test["comparison"][m]["B"] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axs[1].bar(x - width/2, values_a, width, label="Strategy A")
        axs[1].bar(x + width/2, values_b, width, label="Strategy B")
        
        axs[1].set_title("Performance Metrics Comparison")
        axs[1].set_xticks(x)
        axs[1].set_xticklabels([m.replace("_", " ").title() for m in metrics])
        axs[1].legend()
        
        # Add significance annotation
        is_significant = test["statistical_tests"]["is_significant"]
        winner = test["winner"]
        
        if is_significant:
            axs[1].annotate(
                f"Significant difference! Winner: Strategy {winner}",
                xy=(0.5, 0.95),
                xycoords='axes fraction',
                ha='center',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
            )
        else:
            axs[1].annotate(
                "No significant difference",
                xy=(0.5, 0.95),
                xycoords='axes fraction',
                ha='center',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="gray", alpha=0.3)
            )
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()