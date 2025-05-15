"""
Tests for the backtesting framework.

This module contains unit tests for the backtesting framework, including:
- BacktestEngine
- BacktestRunner
- Performance metrics calculation
- Report generation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import os
import tempfile

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.trading.backtest import BacktestEngine, BacktestRunner
from src.trading.strategy_framework import VedicAstrologyStrategy


class TestBacktestEngine:
    """Tests for the BacktestEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = BacktestEngine(initial_capital=100000.0, commission=0.001)
        
        # Create a mock strategy
        self.strategy = MagicMock()
        self.strategy.name = "Mock Strategy"
        self.strategy.reset = MagicMock()
        self.strategy.trades = []
        
        # Create mock signals
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        signals = pd.DataFrame({
            "signal": [0, 1, 0, -1, 0, 1, 0, -1, 0, 0],
            "strength": [0.0, 0.8, 0.0, 0.7, 0.0, 0.9, 0.0, 0.6, 0.0, 0.0],
            "reason": ["", "Buy signal", "", "Sell signal", "", "Buy signal", "", "Sell signal", "", ""]
        }, index=dates)
        
        self.strategy.generate_signals = MagicMock(return_value=signals)
        self.strategy.calculate_position_size = MagicMock(side_effect=lambda strength, account: account * 0.1 * strength)
        self.strategy.execute_trade = MagicMock(side_effect=self._mock_execute_trade)
        
        # Create sample market data
        self.market_data = pd.DataFrame({
            "Open": [100 + i for i in range(10)],
            "High": [105 + i for i in range(10)],
            "Low": [95 + i for i in range(10)],
            "Close": [102 + i for i in range(10)],
            "Volume": [1000000 for _ in range(10)]
        }, index=dates)
        
        # Create sample planetary data
        self.planetary_data = pd.DataFrame({
            "sun_longitude": [i * 10 for i in range(10)],
            "moon_longitude": [i * 12 for i in range(10)]
        }, index=dates)
    
    def _mock_execute_trade(self, symbol, direction, quantity, price, timestamp, signal_strength):
        """Mock trade execution."""
        trade = {
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp,
            "signal_strength": signal_strength
        }
        self.strategy.trades.append(trade)
        return trade
    
    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.initial_capital == 100000.0
        assert self.engine.commission == 0.001
        assert self.engine.results == {}
    
    def test_run_backtest(self):
        """Test running a backtest."""
        results = self.engine.run_backtest(
            self.strategy, 
            self.market_data, 
            self.planetary_data,
            "MOCK"
        )
        
        assert "results" in results
        assert "metrics" in results
        assert "trades" in results
        
        # Check if strategy methods were called
        self.strategy.reset.assert_called_once()
        self.strategy.generate_signals.assert_called_once()
        
        # Check if trades were executed
        assert len(self.strategy.trades) > 0
        
        # Check if results were stored
        assert self.strategy.name in self.engine.results
        
        # Check if results DataFrame has expected columns
        results_df = results["results"]
        assert "close" in results_df.columns
        assert "signal" in results_df.columns
        assert "position" in results_df.columns
        assert "cash" in results_df.columns
        assert "equity" in results_df.columns
        assert "returns" in results_df.columns
        assert "cumulative_returns" in results_df.columns
        assert "drawdown" in results_df.columns
        
        # Check if metrics were calculated
        metrics = results["metrics"]
        assert "total_return" in metrics
        assert "annualized_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
    
    def test_compare_strategies(self):
        """Test comparing multiple strategies."""
        # Create a second mock strategy
        strategy2 = MagicMock()
        strategy2.name = "Mock Strategy 2"
        strategy2.reset = MagicMock()
        strategy2.trades = []
        strategy2.generate_signals = MagicMock(return_value=self.strategy.generate_signals())
        strategy2.calculate_position_size = MagicMock(side_effect=lambda strength, account: account * 0.05 * strength)
        strategy2.execute_trade = MagicMock(side_effect=self._mock_execute_trade)
        
        # Run backtests
        self.engine.run_backtest(self.strategy, self.market_data, self.planetary_data, "MOCK")
        self.engine.run_backtest(strategy2, self.market_data, self.planetary_data, "MOCK")
        
        # Compare strategies
        comparison = self.engine.compare_strategies(
            [], 
            self.market_data, 
            self.planetary_data,
            "MOCK"
        )
        
        assert "equity_curves" in comparison
        assert "metrics_comparison" in comparison
        
        # Check if equity curves include both strategies and market
        equity_curves = comparison["equity_curves"]
        assert "Market" in equity_curves.columns
        assert "Mock Strategy" in equity_curves.columns
        assert "Mock Strategy 2" in equity_curves.columns
        
        # Check if metrics comparison includes both strategies and market
        metrics_comparison = comparison["metrics_comparison"]
        assert "Mock Strategy" in metrics_comparison.index
        assert "Mock Strategy 2" in metrics_comparison.index
        assert "Market" in metrics_comparison.index
    
    def test_generate_report(self):
        """Test report generation."""
        # Run a backtest first
        self.engine.run_backtest(self.strategy, self.market_data, self.planetary_data, "MOCK")
        
        # Create a temporary directory for the report
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = self.engine.generate_report(temp_dir)
            
            # Check if report was generated
            assert os.path.exists(report_path)
            assert report_path.endswith(".html")
            
            # Check if report contains strategy name
            with open(report_path, "r") as f:
                report_content = f.read()
                assert self.strategy.name in report_content
                assert "Performance Metrics" in report_content
                assert "Trades" in report_content
    
    @patch("matplotlib.pyplot.savefig")
    def test_plot_results(self, mock_savefig):
        """Test plotting results."""
        # Run a backtest first
        self.engine.run_backtest(self.strategy, self.market_data, self.planetary_data, "MOCK")
        
        # Plot results
        self.engine.plot_results("test_plot.png")
        
        # Check if savefig was called
        mock_savefig.assert_called_once_with("test_plot.png")
    
    @patch("matplotlib.pyplot.savefig")
    def test_plot_drawdowns(self, mock_savefig):
        """Test plotting drawdowns."""
        # Run a backtest first
        self.engine.run_backtest(self.strategy, self.market_data, self.planetary_data, "MOCK")
        
        # Plot drawdowns
        self.engine.plot_drawdowns("test_drawdowns.png")
        
        # Check if savefig was called
        mock_savefig.assert_called_once_with("test_drawdowns.png")


class TestBacktestRunner:
    """Tests for the BacktestRunner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample market data
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        self.market_data = pd.DataFrame({
            "Open": [100 + i for i in range(10)],
            "High": [105 + i for i in range(10)],
            "Low": [95 + i for i in range(10)],
            "Close": [102 + i for i in range(10)],
            "Volume": [1000000 for _ in range(10)]
        }, index=dates)
        
        # Create sample planetary data
        self.planetary_data = pd.DataFrame({
            "sun_longitude": [i * 10 for i in range(10)],
            "moon_longitude": [i * 12 for i in range(10)],
            "market_trend_primary_trend": ["bullish", "bullish", "neutral", "bearish", 
                                          "bearish", "neutral", "bullish", "bullish", 
                                          "bearish", "neutral"],
            "market_trend_strength": [80, 70, 50, 75, 85, 40, 65, 90, 80, 60],
            "bullish_yoga_count": [3, 2, 1, 0, 0, 1, 2, 4, 1, 0],
            "bearish_yoga_count": [1, 1, 2, 3, 4, 2, 1, 0, 3, 2],
            "moon_nakshatra_financial": ["bullish", "neutral", "bearish", "bearish", 
                                        "neutral", "bullish", "bullish", "neutral", 
                                        "bearish", "neutral"],
            "current_dasha_lord": ["Jupiter", "Venus", "Mercury", "Saturn", 
                                  "Mars", "Rahu", "Moon", "Sun", "Ketu", "Venus"]
        }, index=dates)
        
        # Create runner
        self.runner = BacktestRunner(self.market_data, self.planetary_data)
    
    def test_initialization(self):
        """Test runner initialization."""
        assert hasattr(self.runner, "market_data")
        assert hasattr(self.runner, "planetary_data")
        assert hasattr(self.runner, "engine")
    
    @patch.object(BacktestEngine, "run_backtest")
    def test_run_vedic_strategy_backtest(self, mock_run_backtest):
        """Test running a Vedic strategy backtest."""
        # Set up mock
        mock_run_backtest.return_value = {"results": pd.DataFrame(), "metrics": {}, "trades": []}
        
        # Run backtest
        results = self.runner.run_vedic_strategy_backtest(
            symbol="TEST",
            use_yogas=True,
            use_nakshatras=True,
            use_dashas=False,
            min_signal_strength=0.7
        )
        
        # Check if backtest was run
        mock_run_backtest.assert_called_once()
        
        # Check if strategy was configured correctly
        strategy = mock_run_backtest.call_args[0][0]
        assert isinstance(strategy, VedicAstrologyStrategy)
        assert strategy.use_yogas is True
        assert strategy.use_nakshatras is True
        assert strategy.use_dashas is False
        assert strategy.min_signal_strength == 0.7
    
    @patch.object(BacktestEngine, "run_backtest")
    @patch.object(BacktestEngine, "compare_strategies")
    @patch.object(BacktestEngine, "generate_report")
    def test_run_parameter_sweep(self, mock_generate_report, mock_compare_strategies, mock_run_backtest):
        """Test running a parameter sweep."""
        # Set up mocks
        mock_run_backtest.return_value = {"results": pd.DataFrame(), "metrics": {}, "trades": []}
        mock_compare_strategies.return_value = {"equity_curves": pd.DataFrame(), "metrics_comparison": pd.DataFrame()}
        mock_generate_report.return_value = "test_report.html"
        
        # Run parameter sweep
        results = self.runner.run_parameter_sweep("TEST")
        
        # Check if backtests were run for each parameter combination
        assert mock_run_backtest.call_count == 6  # 6 parameter combinations
        
        # Check if strategies were compared
        mock_compare_strategies.assert_called_once()
        
        # Check if report was generated
        mock_generate_report.assert_called_once()
        
        # Check results
        assert "comparison" in results
        assert "report_path" in results
        assert results["report_path"] == "test_report.html"
    
    @patch.object(BacktestEngine, "plot_results")
    @patch.object(BacktestEngine, "plot_drawdowns")
    def test_visualize_results(self, mock_plot_drawdowns, mock_plot_results):
        """Test visualizing results."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Visualize results
            self.runner.visualize_results(temp_dir)
            
            # Check if plots were generated
            mock_plot_results.assert_called_once()
            mock_plot_drawdowns.assert_called_once()
            
            # Check if paths were passed correctly
            assert temp_dir in mock_plot_results.call_args[0][0]
            assert temp_dir in mock_plot_drawdowns.call_args[0][0]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
