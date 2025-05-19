#!/usr/bin/env python
# Cosmic Market Oracle - Tests for Benchmark Comparison Framework

"""
Unit tests for the benchmark comparison framework.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from src.evaluation.benchmark import (
    BaselineBenchmarks,
    TradingStrategyBenchmarks,
    BenchmarkComparer
)


class TestBaselineBenchmarks:
    """Tests for the BaselineBenchmarks class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        # Create date range
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Create sample data with trend and noise
        trend = np.linspace(100, 150, 100)
        noise = np.random.normal(0, 5, 100)
        values = trend + noise
        
        # Create series
        return pd.Series(values, index=dates)
    
    def test_generate_naive_forecast(self, sample_data):
        """Test naive forecast generation."""
        # Create benchmark
        benchmarks = BaselineBenchmarks()
        
        # Generate naive forecast
        forecast = benchmarks.generate_naive_forecast(sample_data, lag=1)
        
        # Check shape
        assert len(forecast) == len(sample_data)
        
        # Check values (shifted by 1)
        np.testing.assert_almost_equal(forecast[1:], sample_data.values[:-1])
        
        # Check benchmark storage
        assert "naive_lag1" in benchmarks.benchmarks
        assert "naive_lag1" in benchmarks.predictions
    
    def test_generate_moving_average(self, sample_data):
        """Test moving average forecast generation."""
        # Create benchmark
        benchmarks = BaselineBenchmarks()
        
        # Generate moving average
        forecast = benchmarks.generate_moving_average(sample_data, window=5)
        
        # Check shape
        assert len(forecast) == len(sample_data)
        
        # Check values (first value should be the same as input)
        assert forecast[0] == sample_data.values[0]
        
        # Check benchmark storage
        assert "ma5" in benchmarks.benchmarks
        assert "ma5" in benchmarks.predictions
    
    def test_generate_exponential_smoothing(self, sample_data):
        """Test exponential smoothing forecast generation."""
        # Create benchmark
        benchmarks = BaselineBenchmarks()
        
        # Generate exponential smoothing
        forecast = benchmarks.generate_exponential_smoothing(sample_data, alpha=0.3)
        
        # Check shape
        assert len(forecast) == len(sample_data)
        
        # Check values (first value should be the same as input)
        assert forecast[0] == sample_data.values[0]
        
        # Check benchmark storage
        assert "exp_smooth0.3" in benchmarks.benchmarks
        assert "exp_smooth0.3" in benchmarks.predictions
    
    def test_generate_random_walk(self, sample_data):
        """Test random walk forecast generation."""
        # Create benchmark
        benchmarks = BaselineBenchmarks()
        
        # Generate random walk
        forecast = benchmarks.generate_random_walk(sample_data, drift=False)
        
        # Check shape
        assert len(forecast) == len(sample_data)
        
        # Check values (first value should be the same as input)
        assert forecast[0] == sample_data.values[0]
        
        # Check benchmark storage
        assert "random_walk" in benchmarks.benchmarks
        assert "random_walk" in benchmarks.predictions
    
    def test_generate_seasonal_naive(self, sample_data):
        """Test seasonal naive forecast generation."""
        # Create benchmark
        benchmarks = BaselineBenchmarks()
        
        # Generate seasonal naive
        forecast = benchmarks.generate_seasonal_naive(sample_data, season_length=5)
        
        # Check shape
        assert len(forecast) == len(sample_data)
        
        # Check values (first 5 values should be the same as input)
        np.testing.assert_array_equal(forecast[:5], sample_data.values[:5])
        
        # Check values (values after season_length should match values from season_length earlier)
        np.testing.assert_array_equal(forecast[5:10], sample_data.values[:5])
        
        # Check benchmark storage
        assert "seasonal_naive5" in benchmarks.benchmarks
        assert "seasonal_naive5" in benchmarks.predictions
    
    def test_generate_all_benchmarks(self, sample_data):
        """Test generation of all benchmarks."""
        # Create benchmark
        benchmarks = BaselineBenchmarks()
        
        # Generate all benchmarks
        predictions = benchmarks.generate_all_benchmarks(sample_data)
        
        # Check number of benchmarks
        assert len(predictions) == 7
        
        # Check benchmark types
        assert "naive_lag1" in predictions
        assert "ma5" in predictions
        assert "ma10" in predictions
        assert "exp_smooth0.3" in predictions
        assert "random_walk" in predictions
        assert "random_walk_drift" in predictions
        assert "seasonal_naive5" in predictions
    
    def test_evaluate_benchmarks(self, sample_data):
        """Test benchmark evaluation."""
        # Create benchmark
        benchmarks = BaselineBenchmarks()
        
        # Generate all benchmarks
        benchmarks.generate_all_benchmarks(sample_data)
        
        # Evaluate benchmarks
        results = benchmarks.evaluate_benchmarks(sample_data)
        
        # Check number of results
        assert len(results) == 7
        
        # Check result structure
        for name, result in results.items():
            assert "name" in result
            assert "type" in result
            assert "params" in result
            assert "metrics" in result
            
            # Check metrics
            assert "rmse" in result["metrics"]
            assert "mae" in result["metrics"]
            assert "mape" in result["metrics"]
            assert "r2" in result["metrics"]
    
    def test_plot_benchmark_comparison(self, sample_data):
        """Test benchmark comparison plot."""
        # Create benchmark
        benchmarks = BaselineBenchmarks()
        
        # Generate all benchmarks
        benchmarks.generate_all_benchmarks(sample_data)
        
        # Plot comparison
        fig = benchmarks.plot_benchmark_comparison(sample_data, metric_name="rmse")
        
        # Check figure type
        assert isinstance(fig, plt.Figure)
        
        # Close figure to avoid memory leaks
        plt.close(fig)


class TestTradingStrategyBenchmarks:
    """Tests for the TradingStrategyBenchmarks class."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data for testing."""
        # Create date range
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Create sample prices with trend, seasonality, and noise
        trend = np.linspace(100, 150, 100)
        seasonality = 10 * np.sin(np.linspace(0, 6 * np.pi, 100))
        noise = np.random.normal(0, 5, 100)
        prices = trend + seasonality + noise
        
        # Create series
        return pd.Series(prices, index=dates)
    
    def test_generate_buy_hold(self, sample_prices):
        """Test buy-and-hold strategy generation."""
        # Create benchmark
        benchmarks = TradingStrategyBenchmarks()
        
        # Generate buy-and-hold
        signals = benchmarks.generate_buy_hold(sample_prices)
        
        # Check shape
        assert len(signals) == len(sample_prices)
        
        # Check values (all should be 1)
        assert np.all(signals == 1)
        
        # Check benchmark storage
        assert "buy_hold" in benchmarks.benchmarks
        assert "buy_hold" in benchmarks.signals
    
    def test_generate_ma_crossover(self, sample_prices):
        """Test moving average crossover strategy generation."""
        # Create benchmark
        benchmarks = TradingStrategyBenchmarks()
        
        # Generate MA crossover
        signals = benchmarks.generate_ma_crossover(sample_prices, short_window=5, long_window=20)
        
        # Check shape
        assert len(signals) == len(sample_prices)
        
        # Check values (first long_window should be 0)
        assert np.all(signals[:20] == 0)
        
        # Check values (rest should be -1, 0, or 1)
        assert np.all((signals[20:] == -1) | (signals[20:] == 0) | (signals[20:] == 1))
        
        # Check benchmark storage
        assert "ma_cross_5_20" in benchmarks.benchmarks
        assert "ma_cross_5_20" in benchmarks.signals
    
    def test_generate_mean_reversion(self, sample_prices):
        """Test mean reversion strategy generation."""
        # Create benchmark
        benchmarks = TradingStrategyBenchmarks()
        
        # Generate mean reversion
        signals = benchmarks.generate_mean_reversion(sample_prices, window=20, threshold=1.0)
        
        # Check shape
        assert len(signals) == len(sample_prices)
        
        # Check values (first window should be 0)
        assert np.all(signals[:20] == 0)
        
        # Check values (rest should be -1, 0, or 1)
        assert np.all((signals[20:] == -1) | (signals[20:] == 0) | (signals[20:] == 1))
        
        # Check benchmark storage
        assert "mean_rev_20_1.0" in benchmarks.benchmarks
        assert "mean_rev_20_1.0" in benchmarks.signals
    
    def test_generate_momentum(self, sample_prices):
        """Test momentum strategy generation."""
        # Create benchmark
        benchmarks = TradingStrategyBenchmarks()
        
        # Generate momentum
        signals = benchmarks.generate_momentum(sample_prices, window=20, threshold=0.02)
        
        # Check shape
        assert len(signals) == len(sample_prices)
        
        # Check values (first window should be 0)
        assert np.all(signals[:20] == 0)
        
        # Check values (rest should be -1, 0, or 1)
        assert np.all((signals[20:] == -1) | (signals[20:] == 0) | (signals[20:] == 1))
        
        # Check benchmark storage
        assert "momentum_20_0.02" in benchmarks.benchmarks
        assert "momentum_20_0.02" in benchmarks.signals
    
    def test_generate_all_benchmarks(self, sample_prices):
        """Test generation of all benchmark strategies."""
        # Create benchmark
        benchmarks = TradingStrategyBenchmarks()
        
        # Generate all benchmarks
        signals = benchmarks.generate_all_benchmarks(sample_prices)
        
        # Check number of benchmarks
        assert len(signals) == 5
        
        # Check benchmark types
        assert "buy_hold" in signals
        assert "ma_cross_5_20" in signals
        assert "ma_cross_10_50" in signals
        assert "mean_rev_20_1.0" in signals
        assert "momentum_20_0.02" in signals
    
    def test_calculate_returns(self, sample_prices):
        """Test calculation of strategy returns."""
        # Create benchmark
        benchmarks = TradingStrategyBenchmarks()
        
        # Generate all benchmarks
        benchmarks.generate_all_benchmarks(sample_prices)
        
        # Calculate returns
        returns = benchmarks.calculate_returns(sample_prices, transaction_cost=0.001)
        
        # Check number of returns
        assert len(returns) == 5
        
        # Check return types
        assert "buy_hold" in returns
        assert "ma_cross_5_20" in returns
        assert "ma_cross_10_50" in returns
        assert "mean_rev_20_1.0" in returns
        assert "momentum_20_0.02" in returns
        
        # Check return shapes
        for name, return_series in returns.items():
            assert len(return_series) == len(sample_prices) - 1
    
    def test_evaluate_strategies(self, sample_prices):
        """Test evaluation of benchmark strategies."""
        # Create benchmark
        benchmarks = TradingStrategyBenchmarks()
        
        # Generate all benchmarks
        benchmarks.generate_all_benchmarks(sample_prices)
        
        # Evaluate strategies
        results = benchmarks.evaluate_strategies(sample_prices, transaction_cost=0.001)
        
        # Check number of results
        assert len(results) == 5
        
        # Check result structure
        for name, result in results.items():
            assert "name" in result
            assert "type" in result
            assert "params" in result
            assert "metrics" in result
            
            # Check metrics
            assert "total_return" in result["metrics"]
            assert "annual_return" in result["metrics"]
            assert "volatility" in result["metrics"]
            assert "sharpe_ratio" in result["metrics"]
            assert "max_drawdown" in result["metrics"]
            assert "win_rate" in result["metrics"]
    
    def test_plot_equity_curves(self, sample_prices):
        """Test equity curve plot."""
        # Create benchmark
        benchmarks = TradingStrategyBenchmarks()
        
        # Generate all benchmarks
        benchmarks.generate_all_benchmarks(sample_prices)
        
        # Plot equity curves
        fig = benchmarks.plot_equity_curves(sample_prices, transaction_cost=0.001)
        
        # Check figure type
        assert isinstance(fig, plt.Figure)
        
        # Close figure to avoid memory leaks
        plt.close(fig)


class TestBenchmarkComparer:
    """Tests for the BenchmarkComparer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create date range
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Create sample prices with trend, seasonality, and noise
        trend = np.linspace(100, 150, 100)
        seasonality = 10 * np.sin(np.linspace(0, 6 * np.pi, 100))
        noise = np.random.normal(0, 5, 100)
        prices = trend + seasonality + noise
        
        # Create series
        prices_series = pd.Series(prices, index=dates)
        
        # Create model predictions (actual + small noise)
        predictions = prices + np.random.normal(0, 2, 100)
        
        # Create model signals (random for testing)
        signals = np.random.choice([-1, 0, 1], size=100, p=[0.3, 0.1, 0.6])
        
        return {
            "prices": prices_series,
            "predictions": predictions,
            "signals": signals
        }
    
    def test_set_model_predictions(self, sample_data):
        """Test setting model predictions."""
        # Create comparer
        comparer = BenchmarkComparer()
        
        # Set model predictions
        comparer.set_model_predictions(sample_data["predictions"])
        
        # Check model predictions
        assert comparer.model_predictions is not None
        assert len(comparer.model_predictions) == 100
    
    def test_set_model_signals(self, sample_data):
        """Test setting model signals."""
        # Create comparer
        comparer = BenchmarkComparer()
        
        # Set model signals
        comparer.set_model_signals(sample_data["signals"])
        
        # Check model signals
        assert comparer.model_signals is not None
        assert len(comparer.model_signals) == 100
    
    def test_compare_predictions(self, sample_data):
        """Test comparison of model predictions against benchmarks."""
        # Create comparer
        comparer = BenchmarkComparer()
        
        # Set model predictions
        comparer.set_model_predictions(sample_data["predictions"])
        
        # Compare predictions
        comparison = comparer.compare_predictions(sample_data["prices"])
        
        # Check comparison structure
        assert "model" in comparison
        assert len(comparison) > 1  # Model + at least one benchmark
        
        # Check model metrics
        assert "metrics" in comparison["model"]
        assert "rmse" in comparison["model"]["metrics"]
        assert "mae" in comparison["model"]["metrics"]
        assert "directional_accuracy" in comparison["model"]["metrics"]
    
    def test_compare_strategies(self, sample_data):
        """Test comparison of model strategy against benchmarks."""
        # Create comparer
        comparer = BenchmarkComparer()
        
        # Set model signals
        comparer.set_model_signals(sample_data["signals"])
        
        # Compare strategies
        comparison = comparer.compare_strategies(sample_data["prices"])
        
        # Check comparison structure
        assert "model" in comparison
        assert len(comparison) > 1  # Model + at least one benchmark
        
        # Check model metrics
        assert "metrics" in comparison["model"]
        assert "total_return" in comparison["model"]["metrics"]
        assert "sharpe_ratio" in comparison["model"]["metrics"]
        assert "max_drawdown" in comparison["model"]["metrics"]
        assert "win_rate" in comparison["model"]["metrics"]
    
    def test_plot_prediction_comparison(self, sample_data):
        """Test prediction comparison plot."""
        # Create comparer
        comparer = BenchmarkComparer()
        
        # Set model predictions
        comparer.set_model_predictions(sample_data["predictions"])
        
        # Plot comparison
        fig = comparer.plot_prediction_comparison(sample_data["prices"], metric="rmse")
        
        # Check figure type
        assert isinstance(fig, plt.Figure)
        
        # Close figure to avoid memory leaks
        plt.close(fig)
    
    def test_plot_strategy_comparison(self, sample_data):
        """Test strategy comparison plot."""
        # Create comparer
        comparer = BenchmarkComparer()
        
        # Set model signals
        comparer.set_model_signals(sample_data["signals"])
        
        # Plot comparison
        fig = comparer.plot_strategy_comparison(sample_data["prices"], metric="sharpe_ratio")
        
        # Check figure type
        assert isinstance(fig, plt.Figure)
        
        # Close figure to avoid memory leaks
        plt.close(fig)
    
    def test_plot_equity_curve_comparison(self, sample_data):
        """Test equity curve comparison plot."""
        # Create comparer
        comparer = BenchmarkComparer()
        
        # Set model signals
        comparer.set_model_signals(sample_data["signals"])
        
        # Plot comparison
        fig = comparer.plot_equity_curve_comparison(sample_data["prices"])
        
        # Check figure type
        assert isinstance(fig, plt.Figure)
        
        # Close figure to avoid memory leaks
        plt.close(fig)
