#!/usr/bin/env python
# Cosmic Market Oracle - Validation Framework

"""
Validation framework for the Cosmic Market Oracle.

This module provides tools for validating prediction models using various
methodologies, including walk-forward validation, cross-market validation,
and statistical significance testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import copy

from src.evaluation.metrics import PredictionMetrics

# Configure logging
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for time series prediction models.
    
    This class implements walk-forward validation, which respects temporal
    causality by training on historical data and validating on future data
    in a sequential manner.
    """
    
    def __init__(self, n_splits: int = 5, 
                train_size: int = 252,  # 1 year of daily data
                test_size: int = 63,    # 3 months of daily data
                step_size: int = 21,    # 1 month of daily data
                metrics: Optional[PredictionMetrics] = None):
        """
        Initialize the walk-forward validator.
        
        Args:
            n_splits: Number of validation splits
            train_size: Size of the training window
            test_size: Size of the testing window
            step_size: Step size between splits
            metrics: Metrics calculator
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.metrics = metrics or PredictionMetrics()
        
        self.results = []
        
    def validate(self, model_factory: Callable, 
                X: pd.DataFrame, 
                y: pd.Series,
                fit_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform walk-forward validation.
        
        Args:
            model_factory: Function that returns a new model instance
            X: Feature data
            y: Target data
            fit_params: Parameters for model fitting
            
        Returns:
            Dictionary of validation results
        """
        if fit_params is None:
            fit_params = {}
            
        # Initialize results
        self.results = []
        all_predictions = np.zeros_like(y)
        all_predictions[:] = np.nan
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=self.n_splits, 
                              test_size=self.test_size,
                              gap=0)
        
        # Track the start of each fold for reporting
        fold_starts = []
        
        # Perform walk-forward validation
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Ensure train_idx is limited to train_size
            if len(train_idx) > self.train_size:
                train_idx = train_idx[-self.train_size:]
                
            # Get train/test data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Create and fit model
            model = model_factory()
            model.fit(X_train, y_train, **fit_params)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Store predictions
            all_predictions[test_idx] = predictions
            
            # Calculate metrics
            metrics = self.metrics.calculate_regression_metrics(predictions, y_test)
            
            # Store results for this fold
            fold_result = {
                "fold": i + 1,
                "train_start": X.index[train_idx[0]],
                "train_end": X.index[train_idx[-1]],
                "test_start": X.index[test_idx[0]],
                "test_end": X.index[test_idx[-1]],
                "metrics": metrics,
                "n_train_samples": len(train_idx),
                "n_test_samples": len(test_idx)
            }
            
            self.results.append(fold_result)
            fold_starts.append(X.index[test_idx[0]])
            
            logger.info(f"Fold {i+1}: Train {fold_result['train_start']} to {fold_result['train_end']}, "
                      f"Test {fold_result['test_start']} to {fold_result['test_end']}, "
                      f"RMSE: {metrics['rmse']:.4f}, Dir. Acc: {metrics['directional_accuracy']:.4f}")
            
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics()
        
        # Create final results
        validation_results = {
            "method": "walk_forward",
            "n_splits": self.n_splits,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "step_size": self.step_size,
            "fold_results": self.results,
            "aggregate_metrics": aggregate_metrics,
            "predictions": all_predictions,
            "fold_starts": fold_starts
        }
        
        return validation_results
        
    def _calculate_aggregate_metrics(self) -> Dict[str, float]:
        """
        Calculate aggregate metrics across all folds.
        
        Returns:
            Dictionary of aggregate metrics
        """
        if not self.results:
            return {}
            
        # Extract metrics from all folds
        all_metrics = {}
        
        for metric_name in self.results[0]["metrics"]:
            all_metrics[metric_name] = [fold["metrics"][metric_name] for fold in self.results]
            
        # Calculate mean and std for each metric
        aggregate_metrics = {}
        
        for metric_name, values in all_metrics.items():
            aggregate_metrics[f"{metric_name}_mean"] = np.mean(values)
            aggregate_metrics[f"{metric_name}_std"] = np.std(values)
            aggregate_metrics[f"{metric_name}_min"] = np.min(values)
            aggregate_metrics[f"{metric_name}_max"] = np.max(values)
            
        return aggregate_metrics


class CrossMarketValidator:
    """
    Cross-market validation for prediction models.
    
    This class implements cross-market validation, which trains on one market
    and validates on another market to assess generalization capabilities.
    """
    
    def __init__(self, metrics: Optional[PredictionMetrics] = None):
        """
        Initialize the cross-market validator.
        
        Args:
            metrics: Metrics calculator
        """
        self.metrics = metrics or PredictionMetrics()
        self.results = []
        
    def validate(self, model_factory: Callable, 
                markets: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                fit_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform cross-market validation.
        
        Args:
            model_factory: Function that returns a new model instance
            markets: Dictionary of (X, y) tuples for each market
            fit_params: Parameters for model fitting
            
        Returns:
            Dictionary of validation results
        """
        if fit_params is None:
            fit_params = {}
            
        # Initialize results
        self.results = []
        
        # Get market names
        market_names = list(markets.keys())
        
        # Perform cross-market validation
        for train_market in market_names:
            for test_market in market_names:
                if train_market == test_market:
                    continue
                    
                # Get train/test data
                X_train, y_train = markets[train_market]
                X_test, y_test = markets[test_market]
                
                # Create and fit model
                model = model_factory()
                model.fit(X_train, y_train, **fit_params)
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Calculate metrics
                metrics = self.metrics.calculate_regression_metrics(predictions, y_test)
                
                # Store results for this pair
                pair_result = {
                    "train_market": train_market,
                    "test_market": test_market,
                    "metrics": metrics,
                    "n_train_samples": len(X_train),
                    "n_test_samples": len(X_test)
                }
                
                self.results.append(pair_result)
                
                logger.info(f"Train: {train_market}, Test: {test_market}, "
                          f"RMSE: {metrics['rmse']:.4f}, Dir. Acc: {metrics['directional_accuracy']:.4f}")
                
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics()
        
        # Create final results
        validation_results = {
            "method": "cross_market",
            "markets": market_names,
            "pair_results": self.results,
            "aggregate_metrics": aggregate_metrics
        }
        
        return validation_results
        
    def _calculate_aggregate_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate aggregate metrics across all market pairs.
        
        Returns:
            Dictionary of aggregate metrics by market
        """
        if not self.results:
            return {}
            
        # Group results by test market
        market_results = {}
        
        for result in self.results:
            test_market = result["test_market"]
            
            if test_market not in market_results:
                market_results[test_market] = []
                
            market_results[test_market].append(result["metrics"])
            
        # Calculate aggregate metrics for each market
        aggregate_metrics = {}
        
        for market, metrics_list in market_results.items():
            market_metrics = {}
            
            for metric_name in metrics_list[0]:
                values = [metrics[metric_name] for metrics in metrics_list]
                market_metrics[f"{metric_name}_mean"] = np.mean(values)
                market_metrics[f"{metric_name}_std"] = np.std(values)
                
            aggregate_metrics[market] = market_metrics
            
        return aggregate_metrics


class TemporalValidator:
    """
    Temporal validation for prediction models.
    
    This class implements temporal validation, which assesses model performance
    across different time periods to identify temporal dependencies.
    """
    
    def __init__(self, period_length: int = 252,  # 1 year of daily data
                metrics: Optional[PredictionMetrics] = None):
        """
        Initialize the temporal validator.
        
        Args:
            period_length: Length of each time period
            metrics: Metrics calculator
        """
        self.period_length = period_length
        self.metrics = metrics or PredictionMetrics()
        self.results = []
        
    def validate(self, model: Any, 
                X: pd.DataFrame, 
                y: pd.Series) -> Dict[str, Any]:
        """
        Perform temporal validation.
        
        Args:
            model: Trained model
            X: Feature data
            y: Target data
            
        Returns:
            Dictionary of validation results
        """
        # Initialize results
        self.results = []
        
        # Calculate number of periods
        n_samples = len(X)
        n_periods = max(1, n_samples // self.period_length)
        
        # Make predictions for the entire dataset
        predictions = model.predict(X)
        
        # Validate by period
        for i in range(n_periods):
            # Calculate period indices
            start_idx = i * self.period_length
            end_idx = min(n_samples, (i + 1) * self.period_length)
            
            # Get period data
            X_period = X.iloc[start_idx:end_idx]
            y_period = y.iloc[start_idx:end_idx]
            pred_period = predictions[start_idx:end_idx]
            
            # Calculate metrics
            metrics = self.metrics.calculate_regression_metrics(pred_period, y_period)
            
            # Store results for this period
            period_result = {
                "period": i + 1,
                "start_date": X.index[start_idx],
                "end_date": X.index[end_idx - 1] if end_idx > start_idx else X.index[start_idx],
                "metrics": metrics,
                "n_samples": end_idx - start_idx
            }
            
            self.results.append(period_result)
            
            logger.info(f"Period {i+1}: {period_result['start_date']} to {period_result['end_date']}, "
                      f"RMSE: {metrics['rmse']:.4f}, Dir. Acc: {metrics['directional_accuracy']:.4f}")
            
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics()
        
        # Create final results
        validation_results = {
            "method": "temporal",
            "period_length": self.period_length,
            "n_periods": n_periods,
            "period_results": self.results,
            "aggregate_metrics": aggregate_metrics
        }
        
        return validation_results
        
    def _calculate_aggregate_metrics(self) -> Dict[str, float]:
        """
        Calculate aggregate metrics across all periods.
        
        Returns:
            Dictionary of aggregate metrics
        """
        if not self.results:
            return {}
            
        # Extract metrics from all periods
        all_metrics = {}
        
        for metric_name in self.results[0]["metrics"]:
            all_metrics[metric_name] = [period["metrics"][metric_name] for period in self.results]
            
        # Calculate mean and std for each metric
        aggregate_metrics = {}
        
        for metric_name, values in all_metrics.items():
            aggregate_metrics[f"{metric_name}_mean"] = np.mean(values)
            aggregate_metrics[f"{metric_name}_std"] = np.std(values)
            aggregate_metrics[f"{metric_name}_min"] = np.min(values)
            aggregate_metrics[f"{metric_name}_max"] = np.max(values)
            
        # Calculate temporal stability (lower std is better)
        for metric_name, values in all_metrics.items():
            if np.mean(values) != 0:
                stability = 1.0 - (np.std(values) / np.mean(values))
                aggregate_metrics[f"{metric_name}_stability"] = max(0, stability)
            else:
                aggregate_metrics[f"{metric_name}_stability"] = 0.0
                
        return aggregate_metrics


class StatisticalSignificanceTester:
    """
    Statistical significance testing for prediction models.
    
    This class implements statistical tests to assess whether a model's
    performance is significantly better than baseline or competing models.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the statistical significance tester.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
        self.test_results = {}
        
    def test_prediction_significance(self, predictions: np.ndarray, 
                                   actuals: np.ndarray,
                                   baseline_predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Test statistical significance of predictions.
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            baseline_predictions: Baseline model predictions (optional)
            
        Returns:
            Dictionary of test results
        """
        # Calculate errors
        errors = actuals - predictions
        
        # Calculate baseline errors if provided
        if baseline_predictions is not None:
            baseline_errors = actuals - baseline_predictions
            
            # Paired t-test for error difference
            t_stat, p_value = stats.ttest_rel(np.abs(errors), np.abs(baseline_errors))
            
            # Calculate mean error reduction
            mean_error_reduction = np.mean(np.abs(baseline_errors)) - np.mean(np.abs(errors))
            percent_improvement = mean_error_reduction / np.mean(np.abs(baseline_errors)) * 100
            
            # Store results
            self.test_results["error_comparison"] = {
                "test": "paired_t_test",
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < self.alpha,
                "mean_error_reduction": mean_error_reduction,
                "percent_improvement": percent_improvement
            }
            
        # Test for bias (mean error should be zero)
        mean_error = np.mean(errors)
        t_stat, p_value = stats.ttest_1samp(errors, 0)
        
        self.test_results["bias_test"] = {
            "test": "one_sample_t_test",
            "mean_error": mean_error,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_bias": p_value < self.alpha
        }
        
        # Test for autocorrelation in errors (errors should be random)
        acf = self._autocorrelation(errors, lag=1)
        z_score = acf * np.sqrt(len(errors))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        self.test_results["autocorrelation_test"] = {
            "test": "autocorrelation",
            "lag_1_autocorrelation": acf,
            "z_score": z_score,
            "p_value": p_value,
            "significant_autocorrelation": p_value < self.alpha
        }
        
        # Test directional accuracy (better than random)
        direction_pred = np.sign(np.diff(predictions))
        direction_actual = np.sign(np.diff(actuals))
        correct_directions = np.sum(direction_pred == direction_actual)
        total_directions = len(direction_pred)
        
        # Binomial test (null hypothesis: p = 0.5, random guessing)
        p_value = stats.binom_test(correct_directions, total_directions, p=0.5, alternative='greater')
        
        self.test_results["directional_accuracy_test"] = {
            "test": "binomial_test",
            "correct_directions": correct_directions,
            "total_directions": total_directions,
            "accuracy": correct_directions / total_directions,
            "p_value": p_value,
            "significant": p_value < self.alpha
        }
        
        return self.test_results
        
    def test_strategy_significance(self, strategy_returns: np.ndarray,
                                 benchmark_returns: np.ndarray) -> Dict[str, Any]:
        """
        Test statistical significance of trading strategy returns.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Dictionary of test results
        """
        # Calculate excess returns
        excess_returns = strategy_returns - benchmark_returns
        
        # t-test for excess returns (null hypothesis: mean excess return = 0)
        t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
        
        # Calculate Sharpe ratio
        strategy_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
        benchmark_sharpe = np.mean(benchmark_returns) / np.std(benchmark_returns) if np.std(benchmark_returns) > 0 else 0
        
        # Store results
        self.test_results["strategy_test"] = {
            "test": "one_sample_t_test",
            "mean_excess_return": np.mean(excess_returns),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "strategy_sharpe": strategy_sharpe,
            "benchmark_sharpe": benchmark_sharpe,
            "sharpe_ratio_difference": strategy_sharpe - benchmark_sharpe
        }
        
        # Test for strategy consistency (runs test)
        positive_runs = np.where(excess_returns > 0, 1, 0)
        runs, p_value = self._runs_test(positive_runs)
        
        self.test_results["consistency_test"] = {
            "test": "runs_test",
            "runs": runs,
            "p_value": p_value,
            "significant_inconsistency": p_value < self.alpha
        }
        
        return self.test_results
        
    def compare_models(self, predictions_list: List[np.ndarray],
                     model_names: List[str],
                     actuals: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple prediction models.
        
        Args:
            predictions_list: List of model predictions
            model_names: List of model names
            actuals: Actual values
            
        Returns:
            Dictionary of comparison results
        """
        n_models = len(predictions_list)
        
        if n_models < 2:
            logger.warning("At least two models are required for comparison.")
            return {}
            
        if len(model_names) != n_models:
            logger.warning("Number of model names must match number of prediction sets.")
            model_names = [f"Model {i+1}" for i in range(n_models)]
            
        # Calculate errors for each model
        errors_list = [actuals - predictions for predictions in predictions_list]
        
        # Initialize comparison results
        comparison_results = {}
        
        # Compare each pair of models
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model_i = model_names[i]
                model_j = model_names[j]
                
                # Paired t-test for error difference
                t_stat, p_value = stats.ttest_rel(np.abs(errors_list[i]), np.abs(errors_list[j]))
                
                # Calculate mean error difference
                mean_error_i = np.mean(np.abs(errors_list[i]))
                mean_error_j = np.mean(np.abs(errors_list[j]))
                error_diff = mean_error_j - mean_error_i
                
                # Store results
                comparison_key = f"{model_i}_vs_{model_j}"
                comparison_results[comparison_key] = {
                    "test": "paired_t_test",
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < self.alpha,
                    "mean_error_diff": error_diff,
                    "better_model": model_i if error_diff > 0 else model_j if error_diff < 0 else "tie"
                }
                
        # Store overall comparison
        self.test_results["model_comparison"] = comparison_results
        
        return comparison_results
        
    def get_test_results(self) -> Dict[str, Any]:
        """
        Get the test results.
        
        Returns:
            Dictionary of test results
        """
        return self.test_results
        
    def _autocorrelation(self, x: np.ndarray, lag: int = 1) -> float:
        """
        Calculate autocorrelation at specified lag.
        
        Args:
            x: Input array
            lag: Lag value
            
        Returns:
            Autocorrelation coefficient
        """
        n = len(x)
        if n <= lag:
            return 0
            
        # Mean-adjust the series
        x_adj = x - np.mean(x)
        
        # Calculate autocorrelation
        numerator = np.sum(x_adj[lag:] * x_adj[:-lag])
        denominator = np.sum(x_adj ** 2)
        
        if denominator == 0:
            return 0
            
        return numerator / denominator
        
    def _runs_test(self, x: np.ndarray) -> Tuple[int, float]:
        """
        Perform runs test for randomness.
        
        Args:
            x: Binary array (0s and 1s)
            
        Returns:
            Tuple of (number of runs, p-value)
        """
        n = len(x)
        n1 = np.sum(x)
        n0 = n - n1
        
        # Count runs
        runs = 1
        for i in range(1, n):
            if x[i] != x[i-1]:
                runs += 1
                
        # Calculate expected runs and standard deviation
        expected_runs = (2 * n1 * n0) / n + 1
        std_runs = np.sqrt((2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1)))
        
        # Calculate z-score and p-value
        if std_runs == 0:
            return runs, 1.0
            
        z = (runs - expected_runs) / std_runs
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return runs, p_value
