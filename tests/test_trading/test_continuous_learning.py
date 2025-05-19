"""
Tests for the continuous learning module.

This module contains unit tests for the continuous learning components, including:
- Model version management
- Concept drift detection
- Online learning
- Performance monitoring
- A/B testing framework
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import os
import tempfile
import shutil

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.trading.continuous_learning import (
    ModelVersionManager,
    ConceptDriftDetector,
    OnlineLearner,
    PerformanceMonitor,
    ABTestingFramework
)
from src.trading.strategy_framework import BaseStrategy, VedicAstrologyStrategy
from src.trading.backtest import BacktestEngine


class TestModelVersionManager:
    """Tests for the ModelVersionManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelVersionManager(base_dir=self.temp_dir)
        
        # Create a mock strategy
        self.strategy = MagicMock(spec=BaseStrategy)
        self.strategy.name = "Test Strategy"
        self.strategy.__dict__ = {
            "name": "Test Strategy",
            "min_signal_strength": 0.6,
            "use_yogas": True,
            "use_nakshatras": True
        }
        
        # Sample metrics
        self.metrics = {
            "sharpe_ratio": 1.5,
            "total_return": 0.2,
            "max_drawdown": -0.1,
            "win_rate": 0.6
        }
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_save_model(self):
        """Test saving a model."""
        # Save the model
        version_id = self.manager.save_model(
            self.strategy, self.metrics, "Test description")
        
        # Check if files were created
        model_dir = os.path.join(self.temp_dir, version_id)
        assert os.path.exists(model_dir)
        assert os.path.exists(os.path.join(model_dir, "model.joblib"))
        assert os.path.exists(os.path.join(model_dir, "metadata.json"))
        
        # Check if version log was updated
        assert os.path.exists(self.manager.version_log_path)
        assert len(self.manager.version_log["models"]) == 1
        assert self.manager.version_log["current_version"] == version_id
    
    @patch("joblib.load")
    def test_load_model(self, mock_load):
        """Test loading a model."""
        # Mock joblib.load to return our strategy
        mock_load.return_value = self.strategy
        
        # Save the model first
        version_id = self.manager.save_model(
            self.strategy, self.metrics, "Test description")
        
        # Load the model
        model, metadata = self.manager.load_model(version_id)
        
        # Check if the correct model was loaded
        assert model == self.strategy
        assert metadata["version_id"] == version_id
        assert metadata["model_name"] == "Test Strategy"
        assert "metrics" in metadata
        assert metadata["metrics"]["sharpe_ratio"] == 1.5
    
    def test_list_versions(self):
        """Test listing model versions."""
        # Save multiple models
        version_id1 = self.manager.save_model(
            self.strategy, self.metrics, "First version")
        
        self.strategy.name = "Another Strategy"
        version_id2 = self.manager.save_model(
            self.strategy, self.metrics, "Second version")
        
        # List all versions
        versions = self.manager.list_versions()
        assert len(versions) == 2
        
        # List versions by name
        versions = self.manager.list_versions("Test Strategy")
        assert len(versions) == 1
        assert versions[0]["version_id"] == version_id1
        
        versions = self.manager.list_versions("Another Strategy")
        assert len(versions) == 1
        assert versions[0]["version_id"] == version_id2
    
    def test_set_current_version(self):
        """Test setting the current version."""
        # Save multiple models
        version_id1 = self.manager.save_model(
            self.strategy, self.metrics, "First version")
        
        self.strategy.name = "Another Strategy"
        version_id2 = self.manager.save_model(
            self.strategy, self.metrics, "Second version")
        
        # Current version should be the last one saved
        assert self.manager.version_log["current_version"] == version_id2
        
        # Set current version to the first one
        self.manager.set_current_version(version_id1)
        assert self.manager.version_log["current_version"] == version_id1
        
        # Try to set an invalid version
        with pytest.raises(ValueError):
            self.manager.set_current_version("invalid_version")


class TestConceptDriftDetector:
    """Tests for the ConceptDriftDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ConceptDriftDetector(window_size=30, threshold=0.05)
        
        # Create sample data with no drift
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(100)]
        np.random.seed(42)
        self.no_drift_data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(5, 2, 100)
        }, index=dates)
        
        # Create sample data with drift
        self.drift_data = pd.DataFrame({
            "feature1": np.concatenate([
                np.random.normal(0, 1, 50),
                np.random.normal(2, 1, 50)  # Mean shift
            ]),
            "feature2": np.concatenate([
                np.random.normal(5, 2, 50),
                np.random.normal(5, 4, 50)  # Variance shift
            ])
        }, index=dates)
    
    def test_update_reference(self):
        """Test updating reference data."""
        self.detector.update_reference(self.no_drift_data)
        
        assert self.detector.reference_data is not None
        assert len(self.detector.reference_data) == 30
        assert "feature1" in self.detector.reference_data.columns
        assert "feature2" in self.detector.reference_data.columns
    
    def test_detect_no_drift(self):
        """Test detecting no drift."""
        # Initialize with first half
        self.detector.update_reference(self.no_drift_data.iloc[:50])
        
        # Check second half
        drift_detected = self.detector.detect_drift(self.no_drift_data.iloc[50:])
        
        assert drift_detected is False
        assert len(self.detector.drift_history) == 0
    
    def test_detect_drift(self):
        """Test detecting drift."""
        # Initialize with first half
        self.detector.update_reference(self.drift_data.iloc[:50])
        
        # Check second half
        drift_detected = self.detector.detect_drift(self.drift_data.iloc[50:])
        
        assert drift_detected is True
        assert len(self.detector.drift_history) == 1
        assert "timestamp" in self.detector.drift_history[0]
        assert "details" in self.detector.drift_history[0]
    
    @patch("matplotlib.pyplot.savefig")
    def test_plot_drift_history(self, mock_savefig):
        """Test plotting drift history."""
        # Add some drift events
        self.detector.drift_history = [
            {"timestamp": datetime(2022, 1, 15), "details": {}},
            {"timestamp": datetime(2022, 2, 10), "details": {}}
        ]
        
        # Plot with save path
        self.detector.plot_drift_history("test_drift.png")
        
        # Check if plot was saved
        mock_savefig.assert_called_once_with("test_drift.png")


class TestOnlineLearner:
    """Tests for the OnlineLearner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock strategy
        self.strategy = MagicMock(spec=VedicAstrologyStrategy)
        self.strategy.name = "Test Strategy"
        self.strategy.min_signal_strength = 0.6
        
        # Create a mock version manager
        self.version_manager = MagicMock(spec=ModelVersionManager)
        self.version_manager.save_model.return_value = "test_version_id"
        
        # Create the learner
        self.learner = OnlineLearner(
            self.strategy, learning_rate=0.1, version_manager=self.version_manager)
        
        # Create sample data
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        self.market_data = pd.DataFrame({
            "Open": [100 + i for i in range(10)],
            "High": [105 + i for i in range(10)],
            "Low": [95 + i for i in range(10)],
            "Close": [102 + i for i in range(10)],
            "Volume": [1000000 for _ in range(10)]
        }, index=dates)
        
        self.planetary_data = pd.DataFrame({
            "sun_longitude": [i * 10 for i in range(10)],
            "moon_longitude": [i * 12 for i in range(10)]
        }, index=dates)
        
        # Create sample signals
        signals = pd.DataFrame({
            "signal": [0, 1, 0, -1, 0, 1, 0, -1, 0, 0],
            "strength": [0.0, 0.8, 0.0, 0.7, 0.0, 0.9, 0.0, 0.6, 0.0, 0.0],
            "reason": ["", "Buy signal", "", "Sell signal", "", "Buy signal", "", "Sell signal", "", ""]
        }, index=dates)
        
        self.strategy.generate_signals.return_value = signals
        
        # Create sample returns
        self.actual_returns = pd.Series([0.01, 0.02, -0.01, -0.02, 0.005, 0.015, -0.005, -0.015, 0.0, 0.0], index=dates)
    
    def test_update_model(self):
        """Test updating the model."""
        # Update the model
        update_info = self.learner.update_model(
            self.market_data, self.planetary_data, self.actual_returns)
        
        # Check if strategy methods were called
        self.strategy.generate_signals.assert_called_once()
        
        # Check if version manager was used
        self.version_manager.save_model.assert_called_once()
        
        # Check update info
        assert "timestamp" in update_info
        assert "data_range" in update_info
        assert "mse" in update_info
        assert "mae" in update_info
        assert "learning_rate" in update_info
        assert "version_id" in update_info
        assert update_info["version_id"] == "test_version_id"
        
        # Check if update history was recorded
        assert len(self.learner.update_history) == 1
    
    def test_update_parameters(self):
        """Test updating model parameters."""
        # Create error series with high error
        high_error = pd.Series([0.2] * 10, index=self.market_data.index)
        
        # Update parameters
        self.learner._update_parameters(high_error)
        
        # Check if min_signal_strength was increased
        assert self.strategy.min_signal_strength > 0.6
        
        # Create error series with low error
        low_error = pd.Series([0.01] * 10, index=self.market_data.index)
        
        # Reset min_signal_strength
        self.strategy.min_signal_strength = 0.6
        
        # Update parameters
        self.learner._update_parameters(low_error)
        
        # Check if min_signal_strength was decreased
        assert self.strategy.min_signal_strength < 0.6


class TestPerformanceMonitor:
    """Tests for the PerformanceMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock strategy
        self.strategy = MagicMock(spec=VedicAstrologyStrategy)
        self.strategy.name = "Test Strategy"
        
        # Create a mock backtest engine
        self.backtest_engine = MagicMock(spec=BacktestEngine)
        
        # Create a mock retrain callback
        self.retrain_callback = MagicMock()
        
        # Create sample data
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(100)]
        self.market_data = pd.DataFrame({
            "Open": [100 + i * 0.1 for i in range(100)],
            "High": [105 + i * 0.1 for i in range(100)],
            "Low": [95 + i * 0.1 for i in range(100)],
            "Close": [102 + i * 0.1 for i in range(100)],
            "Volume": [1000000 for _ in range(100)]
        }, index=dates)
        
        self.planetary_data = pd.DataFrame({
            "sun_longitude": [i * 3.6 for i in range(100)],
            "moon_longitude": [i * 12 for i in range(100)]
        }, index=dates)
        
        # Create the monitor
        self.monitor = PerformanceMonitor(
            self.strategy,
            self.backtest_engine,
            self.retrain_callback,
            self.market_data,
            self.planetary_data,
            window_size=30,
            sharpe_threshold=0.5,
            drawdown_threshold=0.1,
            check_frequency=7
        )
        
        # Mock backtest results
        self.backtest_engine.run_backtest.return_value = {
            "metrics": {
                "sharpe_ratio": 0.3,  # Below threshold
                "total_return": 0.05,
                "max_drawdown": -0.15,  # Exceeds threshold
                "win_rate": 0.55
            },
            "results": pd.DataFrame({
                "equity": [100000 + i * 100 for i in range(30)]
            }, index=dates[:30])
        }
    
    def test_check_performance_triggers_retrain(self):
        """Test performance check triggering retraining."""
        # Check performance
        check_result = self.monitor.check_performance(datetime(2022, 2, 1))
        
        # Check if backtest was run
        self.backtest_engine.run_backtest.assert_called_once()
        
        # Check if retrain callback was called
        self.retrain_callback.assert_called_once()
        
        # Check result
        assert check_result["status"] == "retrain"
        assert "Sharpe ratio" in check_result["reason"]
        assert "Drawdown" in check_result["reason"]
        
        # Check if performance history was recorded
        assert len(self.monitor.performance_history) == 1
    
    def test_check_performance_no_retrain(self):
        """Test performance check not triggering retraining."""
        # Change mock results to be above thresholds
        self.backtest_engine.run_backtest.return_value["metrics"]["sharpe_ratio"] = 0.7
        self.backtest_engine.run_backtest.return_value["metrics"]["max_drawdown"] = -0.05
        
        # Check performance
        check_result = self.monitor.check_performance(datetime(2022, 2, 1))
        
        # Check if backtest was run
        self.backtest_engine.run_backtest.assert_called_once()
        
        # Check if retrain callback was not called
        self.retrain_callback.assert_not_called()
        
        # Check result
        assert check_result["status"] == "ok"
        assert check_result["reason"] == "Performance acceptable"
    
    def test_check_performance_skip(self):
        """Test skipping performance check."""
        # Check performance first time
        self.monitor.check_performance(datetime(2022, 2, 1))
        
        # Reset mocks
        self.backtest_engine.run_backtest.reset_mock()
        self.retrain_callback.reset_mock()
        
        # Check performance again too soon
        check_result = self.monitor.check_performance(datetime(2022, 2, 2))
        
        # Check if backtest was not run
        self.backtest_engine.run_backtest.assert_not_called()
        
        # Check if retrain callback was not called
        self.retrain_callback.assert_not_called()
        
        # Check result
        assert check_result["status"] == "skipped"
        assert check_result["reason"] == "Too soon since last check"
    
    @patch("matplotlib.pyplot.savefig")
    def test_plot_performance_history(self, mock_savefig):
        """Test plotting performance history."""
        # Add performance history
        self.monitor.performance_history = [
            {
                "date": datetime(2022, 1, 15),
                "metrics": {"sharpe_ratio": 0.3, "max_drawdown": -0.15},
                "window_size": 30
            },
            {
                "date": datetime(2022, 1, 22),
                "metrics": {"sharpe_ratio": 0.4, "max_drawdown": -0.12},
                "window_size": 30
            },
            {
                "date": datetime(2022, 1, 29),
                "metrics": {"sharpe_ratio": 0.6, "max_drawdown": -0.08},
                "window_size": 30
            }
        ]
        
        # Plot sharpe ratio
        self.monitor.plot_performance_history("sharpe_ratio", "test_sharpe.png")
        
        # Check if plot was saved
        mock_savefig.assert_called_once_with("test_sharpe.png")
        
        # Reset mock
        mock_savefig.reset_mock()
        
        # Plot max drawdown
        self.monitor.plot_performance_history("max_drawdown", "test_drawdown.png")
        
        # Check if plot was saved
        mock_savefig.assert_called_once_with("test_drawdown.png")


class TestABTestingFramework:
    """Tests for the ABTestingFramework class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock backtest engine
        self.backtest_engine = MagicMock(spec=BacktestEngine)
        
        # Create sample data
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(100)]
        self.market_data = pd.DataFrame({
            "Open": [100 + i * 0.1 for i in range(100)],
            "High": [105 + i * 0.1 for i in range(100)],
            "Low": [95 + i * 0.1 for i in range(100)],
            "Close": [102 + i * 0.1 for i in range(100)],
            "Volume": [1000000 for _ in range(100)]
        }, index=dates)
        
        self.planetary_data = pd.DataFrame({
            "sun_longitude": [i * 3.6 for i in range(100)],
            "moon_longitude": [i * 12 for i in range(100)]
        }, index=dates)
        
        # Create the framework
        self.framework = ABTestingFramework(
            self.backtest_engine,
            self.market_data,
            self.planetary_data,
            test_duration=30,
            confidence_level=0.95
        )
        
        # Create mock strategies
        self.strategy_a = MagicMock(spec=VedicAstrologyStrategy)
        self.strategy_a.name = "Strategy A"
        
        self.strategy_b = MagicMock(spec=VedicAstrologyStrategy)
        self.strategy_b.name = "Strategy B"
        
        # Mock backtest results
        self.backtest_engine.run_backtest.side_effect = [
            # Strategy A results
            {
                "metrics": {
                    "sharpe_ratio": 1.0,
                    "total_return": 0.1,
                    "max_drawdown": -0.05
                },
                "results": pd.DataFrame({
                    "returns": np.random.normal(0.001, 0.01, 30),
                    "equity": [100000 + i * 100 for i in range(30)]
                }, index=dates[:30])
            },
            # Strategy B results
            {
                "metrics": {
                    "sharpe_ratio": 1.5,
                    "total_return": 0.15,
                    "max_drawdown": -0.07
                },
                "results": pd.DataFrame({
                    "returns": np.random.normal(0.002, 0.01, 30),
                    "equity": [100000 + i * 150 for i in range(30)]
                }, index=dates[:30])
            }
        ]
        
        # Mock results for plotting
        self.backtest_engine.results = {
            "A": {
                "results": {
                    "equity": pd.Series([100000 + i * 100 for i in range(30)], index=dates[:30])
                }
            },
            "B": {
                "results": {
                    "equity": pd.Series([100000 + i * 150 for i in range(30)], index=dates[:30])
                }
            }
        }
    
    def test_run_ab_test(self):
        """Test running an A/B test."""
        # Run the test
        test_results = self.framework.run_ab_test(
            self.strategy_a, self.strategy_b, datetime(2022, 1, 1))
        
        # Check if backtest was run for both strategies
        assert self.backtest_engine.run_backtest.call_count == 2
        
        # Check test results
        assert "start_date" in test_results
        assert "end_date" in test_results
        assert "strategy_a" in test_results
        assert "strategy_b" in test_results
        assert "comparison" in test_results
        assert "statistical_tests" in test_results
        assert "winner" in test_results
        
        # Check comparison
        assert "sharpe_ratio" in test_results["comparison"]
        assert "total_return" in test_results["comparison"]
        assert "max_drawdown" in test_results["comparison"]
        
        # Check if test history was recorded
        assert len(self.framework.test_history) == 1
    
    def test_calculate_significance(self):
        """Test calculating statistical significance."""
        # Create sample returns
        returns_a = pd.Series(np.random.normal(0.001, 0.01, 30))
        returns_b = pd.Series(np.random.normal(0.002, 0.01, 30))
        
        # Calculate significance
        t_stat, p_value = self.framework._calculate_significance(returns_a, returns_b)
        
        # Check results
        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
    
    @patch("matplotlib.pyplot.savefig")
    def test_plot_test_results(self, mock_savefig):
        """Test plotting test results."""
        # Run a test first
        self.framework.run_ab_test(self.strategy_a, self.strategy_b, datetime(2022, 1, 1))
        
        # Plot results
        self.framework.plot_test_results(0, "test_ab.png")
        
        # Check if plot was saved
        mock_savefig.assert_called_once_with("test_ab.png")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
