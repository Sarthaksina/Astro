"""
Tests for the trading strategy framework.

This module contains unit tests for the trading strategy framework, including:
- Base strategy functionality
- Vedic astrology strategy
- Position sizing
- Risk management
- Performance calculation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.trading.strategy_framework import BaseStrategy, VedicAstrologyStrategy


class TestBaseStrategy:
    """Tests for the BaseStrategy class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a concrete implementation of BaseStrategy for testing
        class ConcreteStrategy(BaseStrategy):
            def generate_signals(self, market_data, planetary_data):
                signals = pd.DataFrame(index=market_data.index)
                signals["signal"] = 0
                signals["strength"] = 0.0
                signals["reason"] = ""
                return signals
        
        self.strategy = ConcreteStrategy("Test Strategy")
        
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
            "reversal_probability": [20 + i for i in range(10)]
        }, index=dates)
    
    def test_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.name == "Test Strategy"
        assert self.strategy.positions == {}
        assert self.strategy.trades == []
        assert isinstance(self.strategy.metrics, dict)
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Test with different signal strengths
        size1 = self.strategy.calculate_position_size(0.5, 100000.0)
        size2 = self.strategy.calculate_position_size(1.0, 100000.0)
        
        assert size1 == 1000.0  # 100000 * 0.02 * 0.5
        assert size2 == 2000.0  # 100000 * 0.02 * 1.0
        
        # Test with different risk levels
        size3 = self.strategy.calculate_position_size(0.5, 100000.0, 0.05)
        assert size3 == 2500.0  # 100000 * 0.05 * 0.5
    
    def test_apply_risk_management(self):
        """Test risk management rules."""
        # Test buy signal
        buy_signal = {"direction": "buy"}
        risk_levels = self.strategy.apply_risk_management(100.0, buy_signal, 
                                                        self.market_data, self.planetary_data)
        
        assert "stop_loss" in risk_levels
        assert "take_profit" in risk_levels
        assert risk_levels["stop_loss"] < 100.0  # Stop loss should be below entry for buy
        assert risk_levels["take_profit"] > 100.0  # Take profit should be above entry for buy
        
        # Test sell signal
        sell_signal = {"direction": "sell"}
        risk_levels = self.strategy.apply_risk_management(100.0, sell_signal, 
                                                         self.market_data, self.planetary_data)
        
        assert risk_levels["stop_loss"] > 100.0  # Stop loss should be above entry for sell
        assert risk_levels["take_profit"] < 100.0  # Take profit should be below entry for sell
    
    def test_execute_trade(self):
        """Test trade execution."""
        # Execute a buy trade
        trade = self.strategy.execute_trade(
            symbol="AAPL",
            direction="buy",
            quantity=10,
            price=150.0,
            timestamp=datetime(2022, 1, 1),
            signal_strength=0.8
        )
        
        assert len(self.strategy.trades) == 1
        assert self.strategy.positions == {"AAPL": 10}
        assert trade["symbol"] == "AAPL"
        assert trade["direction"] == "buy"
        assert trade["quantity"] == 10
        assert trade["price"] == 150.0
        
        # Execute a sell trade
        trade = self.strategy.execute_trade(
            symbol="AAPL",
            direction="sell",
            quantity=5,
            price=160.0,
            timestamp=datetime(2022, 1, 2),
            signal_strength=0.7
        )
        
        assert len(self.strategy.trades) == 2
        assert self.strategy.positions == {"AAPL": 5}  # 10 - 5 = 5
    
    def test_reset(self):
        """Test strategy reset."""
        # Add some trades and positions
        self.strategy.execute_trade(
            symbol="AAPL",
            direction="buy",
            quantity=10,
            price=150.0,
            timestamp=datetime(2022, 1, 1),
            signal_strength=0.8
        )
        
        # Reset strategy
        self.strategy.reset()
        
        assert self.strategy.positions == {}
        assert self.strategy.trades == []
        assert self.strategy.metrics["total_return"] == 0.0


class TestVedicAstrologyStrategy:
    """Tests for the VedicAstrologyStrategy class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = VedicAstrologyStrategy()
        
        # Create sample market data
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        self.market_data = pd.DataFrame({
            "Open": [100 + i for i in range(10)],
            "High": [105 + i for i in range(10)],
            "Low": [95 + i for i in range(10)],
            "Close": [102 + i for i in range(10)],
            "Volume": [1000000 for _ in range(10)]
        }, index=dates)
        
        # Create sample planetary data with Vedic factors
        self.planetary_data = pd.DataFrame({
            "sun_longitude": [i * 10 for i in range(10)],
            "moon_longitude": [i * 12 for i in range(10)],
            "market_trend_primary_trend": ["bullish", "bullish", "neutral", "bearish", 
                                          "bearish", "neutral", "bullish", "bullish", 
                                          "bearish", "neutral"],
            "market_trend_strength": [80, 70, 50, 75, 85, 40, 65, 90, 80, 60],
            "market_trend_reversal_probability": [20, 25, 30, 15, 10, 60, 30, 20, 15, 75],
            "bullish_yoga_count": [3, 2, 1, 0, 0, 1, 2, 4, 1, 0],
            "bearish_yoga_count": [1, 1, 2, 3, 4, 2, 1, 0, 3, 2],
            "moon_nakshatra_financial": ["bullish", "neutral", "bearish", "bearish", 
                                        "neutral", "bullish", "bullish", "neutral", 
                                        "bearish", "neutral"],
            "current_dasha_lord": ["Jupiter", "Venus", "Mercury", "Saturn", 
                                  "Mars", "Rahu", "Moon", "Sun", "Ketu", "Venus"]
        }, index=dates)
    
    def test_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.name == "Vedic Astrology Strategy"
        assert self.strategy.use_yogas is True
        assert self.strategy.use_nakshatras is True
        assert self.strategy.use_dashas is True
        assert self.strategy.min_signal_strength == 0.6
    
    def test_generate_signals(self):
        """Test signal generation."""
        signals = self.strategy.generate_signals(self.market_data, self.planetary_data)
        
        assert isinstance(signals, pd.DataFrame)
        assert "signal" in signals.columns
        assert "strength" in signals.columns
        assert "reason" in signals.columns
        
        # Check if signals are generated
        assert (signals["signal"] != 0).any()
        
        # Check if signal strengths are valid
        assert (signals["strength"] >= 0).all() and (signals["strength"] <= 1).all()
        
        # Check if reasons are provided for signals
        assert all(signals.loc[signals["signal"] != 0, "reason"] != "")
    
    def test_backtest(self):
        """Test strategy backtesting."""
        results = self.strategy.backtest(self.market_data, self.planetary_data)
        
        assert isinstance(results, pd.DataFrame)
        assert "close" in results.columns
        assert "signal" in results.columns
        assert "position" in results.columns
        assert "cash" in results.columns
        assert "equity" in results.columns
        assert "returns" in results.columns
        assert "cumulative_returns" in results.columns
        assert "drawdown" in results.columns
        
        # Check if equity is calculated correctly
        assert (results["equity"] == results["cash"] + results["position"] * results["close"]).all()
        
        # Check if trades are recorded
        assert len(self.strategy.trades) > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
