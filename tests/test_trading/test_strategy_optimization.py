"""
Tests for the strategy optimization module.

This module contains unit tests for the strategy optimization components, including:
- Base strategy optimizer
- Vedic strategy optimizer
- Multi-objective optimizer
- Scenario analysis
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.trading.strategy_optimization import (
    StrategyOptimizer,
    VedicStrategyOptimizer,
    MultiObjectiveOptimizer,
    ScenarioAnalysis
)
from src.trading.strategy_framework import VedicAstrologyStrategy


class TestStrategyOptimizer:
    """Tests for the base StrategyOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample market data
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(100)]
        self.market_data = pd.DataFrame({
            "Open": [100 + i * 0.1 for i in range(100)],
            "High": [105 + i * 0.1 for i in range(100)],
            "Low": [95 + i * 0.1 for i in range(100)],
            "Close": [102 + i * 0.1 for i in range(100)],
            "Volume": [1000000 for _ in range(100)]
        }, index=dates)
        
        # Create sample planetary data
        self.planetary_data = pd.DataFrame({
            "sun_longitude": [i * 3.6 for i in range(100)],
            "moon_longitude": [i * 12 for i in range(100)],
            "market_trend_primary_trend": ["bullish" if i % 3 == 0 else "bearish" if i % 3 == 1 else "neutral" for i in range(100)],
            "market_trend_strength": [70 + i % 20 for i in range(100)],
            "bullish_yoga_count": [i % 5 for i in range(100)],
            "bearish_yoga_count": [(4 - i) % 5 for i in range(100)],
            "moon_nakshatra_financial": ["bullish" if i % 3 == 0 else "bearish" if i % 3 == 1 else "neutral" for i in range(100)],
            "current_dasha_lord": ["Jupiter" if i % 9 == 0 else "Venus" if i % 9 == 1 else "Mercury" if i % 9 == 2 else
                                  "Saturn" if i % 9 == 3 else "Mars" if i % 9 == 4 else "Rahu" if i % 9 == 5 else
                                  "Moon" if i % 9 == 6 else "Sun" if i % 9 == 7 else "Ketu" for i in range(100)]
        }, index=dates)
        
        # Create optimizer
        self.optimizer = StrategyOptimizer(
            market_data=self.market_data,
            planetary_data=self.planetary_data,
            initial_capital=100000.0,
            commission=0.001
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        assert hasattr(self.optimizer, "market_data")
        assert hasattr(self.optimizer, "planetary_data")
        assert hasattr(self.optimizer, "initial_capital")
        assert hasattr(self.optimizer, "commission")
        assert hasattr(self.optimizer, "backtest_engine")
        assert hasattr(self.optimizer, "best_params")
        assert hasattr(self.optimizer, "best_metrics")
    
    @patch("optuna.create_study")
    def test_optimize(self, mock_create_study):
        """Test strategy optimization."""
        # Mock Optuna study
        mock_study = MagicMock()
        mock_study.best_params = {"param1": 0.5, "param2": True}
        mock_create_study.return_value = mock_study
        
        # Mock strategy class
        mock_strategy_class = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class.return_value = mock_strategy_instance
        
        # Mock backtest engine
        self.optimizer.backtest_engine = MagicMock()
        self.optimizer.backtest_engine.run_backtest.return_value = {
            "metrics": {
                "sharpe_ratio": 1.5,
                "total_return": 0.2,
                "annualized_return": 0.15,
                "win_rate": 0.6
            }
        }
        
        # Define parameter space
        param_space = {
            "param1": (0.0, 1.0, 0.1),
            "param2": [True, False]
        }
        
        # Run optimization
        best_params = self.optimizer.optimize(
            mock_strategy_class, param_space, objective="sharpe_ratio", n_trials=10)
        
        # Check if study was created and optimized
        mock_create_study.assert_called_once()
        mock_study.optimize.assert_called_once()
        
        # Check if best parameters were returned
        assert best_params == {"param1": 0.5, "param2": True}
        assert self.optimizer.best_params == {"param1": 0.5, "param2": True}
        assert self.optimizer.best_metrics == {
            "sharpe_ratio": 1.5,
            "total_return": 0.2,
            "annualized_return": 0.15,
            "win_rate": 0.6
        }
    
    @patch("matplotlib.pyplot.savefig")
    @patch("optuna.visualization.matplotlib.plot_optimization_history")
    @patch("optuna.visualization.matplotlib.plot_param_importances")
    def test_plot_optimization_results(self, mock_plot_importances, mock_plot_history, mock_savefig):
        """Test plotting optimization results."""
        # Mock study
        mock_study = MagicMock()
        
        # Test with save_path
        self.optimizer.plot_optimization_results(mock_study, "test_plot.png")
        
        # Check if visualization functions were called
        mock_plot_history.assert_called_once()
        mock_plot_importances.assert_called_once()
        
        # Check if plots were saved
        assert mock_savefig.call_count == 2
    
    def test_cross_validate(self):
        """Test cross-validation."""
        # Mock strategy class
        mock_strategy_class = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class.return_value = mock_strategy_instance
        
        # Mock backtest engine
        self.optimizer.backtest_engine = MagicMock()
        self.optimizer.backtest_engine.run_backtest.return_value = {
            "metrics": {
                "sharpe_ratio": 1.5,
                "total_return": 0.2,
                "annualized_return": 0.15,
                "max_drawdown": -0.1,
                "win_rate": 0.6
            }
        }
        
        # Run cross-validation
        cv_stats = self.optimizer.cross_validate(
            mock_strategy_class, {"param1": 0.5, "param2": True}, n_splits=5)
        
        # Check if backtest was run for each split
        assert self.optimizer.backtest_engine.run_backtest.call_count == 5
        
        # Check if statistics were calculated
        assert "total_return_mean" in cv_stats
        assert "total_return_std" in cv_stats
        assert "total_return_min" in cv_stats
        assert "total_return_max" in cv_stats
        assert "sharpe_ratio_mean" in cv_stats
        assert "max_drawdown_mean" in cv_stats
        assert "win_rate_mean" in cv_stats


class TestVedicStrategyOptimizer:
    """Tests for the VedicStrategyOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample market data
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(100)]
        self.market_data = pd.DataFrame({
            "Open": [100 + i * 0.1 for i in range(100)],
            "High": [105 + i * 0.1 for i in range(100)],
            "Low": [95 + i * 0.1 for i in range(100)],
            "Close": [102 + i * 0.1 for i in range(100)],
            "Volume": [1000000 for _ in range(100)]
        }, index=dates)
        
        # Create sample planetary data
        self.planetary_data = pd.DataFrame({
            "sun_longitude": [i * 3.6 for i in range(100)],
            "moon_longitude": [i * 12 for i in range(100)],
            "market_trend_primary_trend": ["bullish" if i % 3 == 0 else "bearish" if i % 3 == 1 else "neutral" for i in range(100)],
            "market_trend_strength": [70 + i % 20 for i in range(100)],
            "bullish_yoga_count": [i % 5 for i in range(100)],
            "bearish_yoga_count": [(4 - i) % 5 for i in range(100)],
            "moon_nakshatra_financial": ["bullish" if i % 3 == 0 else "bearish" if i % 3 == 1 else "neutral" for i in range(100)],
            "current_dasha_lord": ["Jupiter" if i % 9 == 0 else "Venus" if i % 9 == 1 else "Mercury" if i % 9 == 2 else
                                  "Saturn" if i % 9 == 3 else "Mars" if i % 9 == 4 else "Rahu" if i % 9 == 5 else
                                  "Moon" if i % 9 == 6 else "Sun" if i % 9 == 7 else "Ketu" for i in range(100)]
        }, index=dates)
        
        # Create optimizer
        self.optimizer = VedicStrategyOptimizer(
            market_data=self.market_data,
            planetary_data=self.planetary_data,
            initial_capital=100000.0,
            commission=0.001
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        assert isinstance(self.optimizer, StrategyOptimizer)
    
    @patch("src.trading.strategy_optimization.StrategyOptimizer.optimize")
    def test_optimize_vedic_strategy(self, mock_optimize):
        """Test optimizing Vedic strategy."""
        # Mock optimize method
        mock_optimize.return_value = {
            "name": "Optimized Vedic Strategy",
            "min_signal_strength": 0.7,
            "use_yogas": True,
            "use_nakshatras": True,
            "use_dashas": False
        }
        
        # Run optimization
        best_params = self.optimizer.optimize_vedic_strategy(n_trials=10)
        
        # Check if optimize was called with correct parameters
        mock_optimize.assert_called_once()
        args, kwargs = mock_optimize.call_args
        assert args[0] == VedicAstrologyStrategy
        assert "min_signal_strength" in args[1]
        assert "use_yogas" in args[1]
        assert "use_nakshatras" in args[1]
        assert "use_dashas" in args[1]
        assert kwargs["objective"] == "sharpe_ratio"
        assert kwargs["n_trials"] == 10
        
        # Check if best parameters were returned
        assert best_params == {
            "name": "Optimized Vedic Strategy",
            "min_signal_strength": 0.7,
            "use_yogas": True,
            "use_nakshatras": True,
            "use_dashas": False
        }
    
    @patch("src.trading.strategy_optimization.BacktestEngine.run_backtest")
    def test_analyze_market_regimes(self, mock_run_backtest):
        """Test analyzing market regimes."""
        # Mock run_backtest method
        mock_run_backtest.return_value = {
            "metrics": {
                "sharpe_ratio": 1.5,
                "total_return": 0.2,
                "annualized_return": 0.15,
                "max_drawdown": -0.1,
                "win_rate": 0.6
            }
        }
        
        # Run analysis
        regime_results = self.optimizer.analyze_market_regimes(
            VedicAstrologyStrategy,
            {"min_signal_strength": 0.7, "use_yogas": True},
            n_regimes=3
        )
        
        # Check if run_backtest was called for each regime
        assert mock_run_backtest.call_count == 3
        
        # Check if results were returned for each regime
        assert "regime_0" in regime_results
        assert "regime_1" in regime_results
        assert "regime_2" in regime_results
        
        # Check if metrics were recorded
        for regime in range(3):
            assert "sharpe_ratio" in regime_results[f"regime_{regime}"]
            assert "total_return" in regime_results[f"regime_{regime}"]
            assert "annualized_return" in regime_results[f"regime_{regime}"]
            assert "max_drawdown" in regime_results[f"regime_{regime}"]
            assert "win_rate" in regime_results[f"regime_{regime}"]
    
    @patch("src.trading.strategy_optimization.StrategyOptimizer.optimize")
    def test_optimize_for_regime(self, mock_optimize):
        """Test optimizing for specific regimes."""
        # Mock optimize method
        mock_optimize.return_value = {
            "name": "Regime-Specific Vedic Strategy",
            "min_signal_strength": 0.7,
            "use_yogas": True,
            "use_nakshatras": True,
            "use_dashas": False
        }
        
        # Create regime data
        regime_data = {
            "bull_market": {
                "market_data": self.market_data.iloc[:30],
                "planetary_data": self.planetary_data.iloc[:30]
            },
            "bear_market": {
                "market_data": self.market_data.iloc[30:60],
                "planetary_data": self.planetary_data.iloc[30:60]
            },
            "sideways_market": {
                "market_data": self.market_data.iloc[60:],
                "planetary_data": self.planetary_data.iloc[60:]
            }
        }
        
        # Run optimization
        regime_params = self.optimizer.optimize_for_regime(regime_data, n_trials=10)
        
        # Check if optimize was called for each regime
        assert mock_optimize.call_count == 3
        
        # Check if parameters were returned for each regime
        assert "bull_market" in regime_params
        assert "bear_market" in regime_params
        assert "sideways_market" in regime_params


class TestMultiObjectiveOptimizer:
    """Tests for the MultiObjectiveOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample market data
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(100)]
        self.market_data = pd.DataFrame({
            "Open": [100 + i * 0.1 for i in range(100)],
            "High": [105 + i * 0.1 for i in range(100)],
            "Low": [95 + i * 0.1 for i in range(100)],
            "Close": [102 + i * 0.1 for i in range(100)],
            "Volume": [1000000 for _ in range(100)]
        }, index=dates)
        
        # Create sample planetary data
        self.planetary_data = pd.DataFrame({
            "sun_longitude": [i * 3.6 for i in range(100)],
            "moon_longitude": [i * 12 for i in range(100)],
            "market_trend_primary_trend": ["bullish" if i % 3 == 0 else "bearish" if i % 3 == 1 else "neutral" for i in range(100)],
            "market_trend_strength": [70 + i % 20 for i in range(100)],
            "bullish_yoga_count": [i % 5 for i in range(100)],
            "bearish_yoga_count": [(4 - i) % 5 for i in range(100)]
        }, index=dates)
        
        # Create optimizer
        self.optimizer = MultiObjectiveOptimizer(
            market_data=self.market_data,
            planetary_data=self.planetary_data,
            initial_capital=100000.0,
            commission=0.001
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        assert isinstance(self.optimizer, StrategyOptimizer)
    
    @patch("optuna.create_study")
    def test_optimize_multi_objective(self, mock_create_study):
        """Test multi-objective optimization."""
        # Mock Optuna study
        mock_study = MagicMock()
        mock_best_trials = [
            MagicMock(params={"param1": 0.5, "param2": True}, values=[0.2, 1.5]),
            MagicMock(params={"param1": 0.7, "param2": False}, values=[0.3, 1.2])
        ]
        mock_study.best_trials = mock_best_trials
        mock_create_study.return_value = mock_study
        
        # Mock strategy class
        mock_strategy_class = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class.return_value = mock_strategy_instance
        
        # Mock backtest engine
        self.optimizer.backtest_engine = MagicMock()
        self.optimizer.backtest_engine.run_backtest.return_value = {
            "metrics": {
                "sharpe_ratio": 1.5,
                "total_return": 0.2
            }
        }
        
        # Define parameter space
        param_space = {
            "param1": (0.0, 1.0, 0.1),
            "param2": [True, False]
        }
        
        # Run optimization
        pareto_solutions = self.optimizer.optimize_multi_objective(
            mock_strategy_class, param_space, 
            objectives=["total_return", "sharpe_ratio"], 
            n_trials=10
        )
        
        # Check if study was created and optimized
        mock_create_study.assert_called_once_with(directions=["maximize", "maximize"])
        mock_study.optimize.assert_called_once()
        
        # Check if Pareto solutions were returned
        assert len(pareto_solutions) == 2
        assert "params" in pareto_solutions[0]
        assert "values" in pareto_solutions[0]
        assert pareto_solutions[0]["params"] == {"param1": 0.5, "param2": True}
        assert pareto_solutions[0]["values"] == [0.2, 1.5]
    
    @patch("matplotlib.pyplot.savefig")
    def test_plot_pareto_front_2d(self, mock_savefig):
        """Test plotting 2D Pareto front."""
        # Create sample Pareto solutions
        pareto_solutions = [
            {"params": {"param1": 0.5}, "values": [0.2, 1.5]},
            {"params": {"param1": 0.6}, "values": [0.25, 1.4]},
            {"params": {"param1": 0.7}, "values": [0.3, 1.2]}
        ]
        
        # Test with save_path
        self.optimizer.plot_pareto_front(
            pareto_solutions, ["total_return", "sharpe_ratio"], "test_pareto.png")
        
        # Check if plot was saved
        mock_savefig.assert_called_once_with("test_pareto.png")
    
    @patch("matplotlib.pyplot.savefig")
    def test_plot_pareto_front_3d(self, mock_savefig):
        """Test plotting 3D Pareto front."""
        # Create sample Pareto solutions
        pareto_solutions = [
            {"params": {"param1": 0.5}, "values": [0.2, 1.5, 0.6]},
            {"params": {"param1": 0.6}, "values": [0.25, 1.4, 0.55]},
            {"params": {"param1": 0.7}, "values": [0.3, 1.2, 0.5]}
        ]
        
        # Test with save_path
        self.optimizer.plot_pareto_front(
            pareto_solutions, ["total_return", "sharpe_ratio", "win_rate"], "test_pareto_3d.png")
        
        # Check if plot was saved
        mock_savefig.assert_called_once_with("test_pareto_3d.png")


class TestScenarioAnalysis:
    """Tests for the ScenarioAnalysis class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample market data
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(100)]
        self.market_data = pd.DataFrame({
            "Open": [100 + i * 0.1 for i in range(100)],
            "High": [105 + i * 0.1 for i in range(100)],
            "Low": [95 + i * 0.1 for i in range(100)],
            "Close": [102 + i * 0.1 for i in range(100)],
            "Volume": [1000000 for _ in range(100)]
        }, index=dates)
        
        # Create sample planetary data
        self.planetary_data = pd.DataFrame({
            "sun_longitude": [i * 3.6 for i in range(100)],
            "moon_longitude": [i * 12 for i in range(100)]
        }, index=dates)
        
        # Create mock strategy
        self.strategy = MagicMock()
        
        # Create scenario analysis
        self.scenario_analysis = ScenarioAnalysis(
            strategy=self.strategy,
            market_data=self.market_data,
            planetary_data=self.planetary_data,
            initial_capital=100000.0,
            commission=0.001
        )
    
    def test_initialization(self):
        """Test scenario analysis initialization."""
        assert hasattr(self.scenario_analysis, "strategy")
        assert hasattr(self.scenario_analysis, "market_data")
        assert hasattr(self.scenario_analysis, "planetary_data")
        assert hasattr(self.scenario_analysis, "initial_capital")
        assert hasattr(self.scenario_analysis, "commission")
        assert hasattr(self.scenario_analysis, "backtest_engine")
    
    @patch("src.trading.strategy_optimization.BacktestEngine.run_backtest")
    def test_run_scenario_analysis(self, mock_run_backtest):
        """Test running scenario analysis."""
        # Mock run_backtest method
        mock_run_backtest.return_value = {
            "metrics": {
                "sharpe_ratio": 1.5,
                "total_return": 0.2,
                "annualized_return": 0.15,
                "max_drawdown": -0.1,
                "win_rate": 0.6
            },
            "results": pd.DataFrame({
                "equity": [100000 + i * 1000 for i in range(100)]
            }, index=self.market_data.index)
        }
        
        # Define scenarios
        scenarios = {
            "base_case": {},
            "high_volatility": {"volatility_factor": 1.5},
            "bull_market": {"trend_factor": 0.2},
            "bear_market": {"trend_factor": -0.2},
            "market_crash": {"crash": True, "crash_magnitude": -0.3}
        }
        
        # Run scenario analysis
        scenario_results = self.scenario_analysis.run_scenario_analysis(scenarios)
        
        # Check if run_backtest was called for each scenario
        assert mock_run_backtest.call_count == 5
        
        # Check if results were returned for each scenario
        for scenario in scenarios:
            assert scenario in scenario_results
            assert "metrics" in scenario_results[scenario]
            assert "equity_curve" in scenario_results[scenario]
    
    def test_modify_market_data(self):
        """Test modifying market data for scenarios."""
        # Test volatility adjustment
        modified_data = self.scenario_analysis._modify_market_data({"volatility_factor": 1.5})
        assert not modified_data.equals(self.market_data)
        
        # Test trend adjustment
        modified_data = self.scenario_analysis._modify_market_data({"trend_factor": 0.2})
        assert not modified_data.equals(self.market_data)
        
        # Test market crash
        modified_data = self.scenario_analysis._modify_market_data({
            "crash": True, 
            "crash_magnitude": -0.3, 
            "crash_day": 50,
            "recovery_days": 30
        })
        assert not modified_data.equals(self.market_data)
        
        # Check if crash was applied
        assert modified_data["Close"].iloc[50] < self.market_data["Close"].iloc[50]
    
    @patch("matplotlib.pyplot.savefig")
    def test_plot_scenario_results(self, mock_savefig):
        """Test plotting scenario results."""
        # Create sample scenario results
        scenario_results = {
            "base_case": {
                "metrics": {
                    "total_return": 0.2,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": -0.1,
                    "win_rate": 0.6
                },
                "equity_curve": pd.Series([100000 + i * 1000 for i in range(100)], index=self.market_data.index)
            },
            "high_volatility": {
                "metrics": {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.0,
                    "max_drawdown": -0.15,
                    "win_rate": 0.55
                },
                "equity_curve": pd.Series([100000 + i * 800 for i in range(100)], index=self.market_data.index)
            }
        }
        
        # Test with save_path
        self.scenario_analysis.plot_scenario_results(scenario_results, "test_scenarios.png")
        
        # Check if plots were saved
        assert mock_savefig.call_count == 2


if __name__ == "__main__":
    pytest.main(["-v", __file__])
