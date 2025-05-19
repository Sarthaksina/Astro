"""
Tests for the portfolio construction module.

This module contains unit tests for the portfolio construction components, including:
- Base portfolio constructor
- MPT portfolio constructor
- Astrological sector rotation
- Astrological risk parity
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.trading.portfolio_construction import (
    PortfolioConstructor,
    MPTPortfolioConstructor,
    AstrologicalSectorRotation,
    AstrologicalRiskParity,
    create_portfolio_constructor
)


class TestPortfolioConstructor:
    """Tests for the base PortfolioConstructor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.constructor = PortfolioConstructor("Test Constructor")
        
        # Create sample signals
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        self.signals = pd.DataFrame({
            "AAPL": [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            "MSFT": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            "GOOG": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
        }, index=dates)
        
        # Add signal strength
        self.signals.loc["strength"] = [0.8, 0.7, 0.6]
        
        # Create sample market data
        self.market_data = {}
        for symbol in ["AAPL", "MSFT", "GOOG"]:
            self.market_data[symbol] = pd.DataFrame({
                "Open": [100 + i for i in range(10)],
                "High": [105 + i for i in range(10)],
                "Low": [95 + i for i in range(10)],
                "Close": [102 + i for i in range(10)],
                "Volume": [1000000 for _ in range(10)]
            }, index=dates)
    
    def test_initialization(self):
        """Test constructor initialization."""
        assert self.constructor.name == "Test Constructor"
        assert hasattr(self.constructor, "calculator")
    
    def test_allocate_portfolio(self):
        """Test portfolio allocation."""
        # Test with equal allocation
        allocation = self.constructor.allocate_portfolio(
            self.signals, self.market_data, 100000.0, 10)
        
        assert isinstance(allocation, dict)
        assert len(allocation) == 3  # All three assets have positive signals
        
        # Check if allocation sums to total capital
        assert sum(allocation.values()) == pytest.approx(100000.0)
        
        # Check if all assets have equal allocation
        for value in allocation.values():
            assert value == pytest.approx(100000.0 / 3)
        
        # Test with max_positions limit
        allocation = self.constructor.allocate_portfolio(
            self.signals, self.market_data, 100000.0, 2)
        
        assert isinstance(allocation, dict)
        assert len(allocation) == 2  # Limited to 2 assets
        
        # Check if allocation sums to total capital
        assert sum(allocation.values()) == pytest.approx(100000.0)
        
        # Check if all assets have equal allocation
        for value in allocation.values():
            assert value == pytest.approx(100000.0 / 2)


class TestMPTPortfolioConstructor:
    """Tests for the MPTPortfolioConstructor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.constructor = MPTPortfolioConstructor(
            risk_aversion=2.0, min_weight=0.1, max_weight=0.5)
        
        # Create sample signals
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(100)]
        self.signals = pd.DataFrame({
            "AAPL": [1] * 100,
            "MSFT": [1] * 100,
            "GOOG": [1] * 100
        }, index=dates)
        
        # Add signal strength
        self.signals.loc["strength"] = [0.8, 0.7, 0.6]
        
        # Create sample market data with realistic returns
        self.market_data = {}
        np.random.seed(42)  # For reproducible tests
        
        for symbol, vol in [("AAPL", 0.2), ("MSFT", 0.15), ("GOOG", 0.25)]:
            # Generate random returns
            returns = np.random.normal(0.0005, vol / np.sqrt(252), 100)
            
            # Generate prices from returns
            prices = 100 * np.cumprod(1 + returns)
            
            # Create DataFrame
            self.market_data[symbol] = pd.DataFrame({
                "Open": prices * 0.99,
                "High": prices * 1.01,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": [1000000 for _ in range(100)]
            }, index=dates)
    
    def test_initialization(self):
        """Test constructor initialization."""
        assert self.constructor.name == "MPT Portfolio Constructor"
        assert self.constructor.risk_aversion == 2.0
        assert self.constructor.min_weight == 0.1
        assert self.constructor.max_weight == 0.5
    
    def test_allocate_portfolio(self):
        """Test portfolio allocation using MPT."""
        # Test allocation
        allocation = self.constructor.allocate_portfolio(
            self.signals, self.market_data, 100000.0, 10)
        
        assert isinstance(allocation, dict)
        assert len(allocation) == 3  # All three assets have positive signals
        
        # Check if allocation sums to total capital
        assert sum(allocation.values()) == pytest.approx(100000.0, rel=1e-3)
        
        # Check if weights respect constraints
        for value in allocation.values():
            assert value >= 0.1 * 100000.0  # min_weight
            assert value <= 0.5 * 100000.0  # max_weight
    
    def test_optimize_portfolio(self):
        """Test portfolio optimization."""
        # Create sample returns and covariance
        expected_returns = pd.Series({
            "AAPL": 0.1,
            "MSFT": 0.08,
            "GOOG": 0.12
        })
        
        cov_matrix = pd.DataFrame({
            "AAPL": [0.04, 0.02, 0.01],
            "MSFT": [0.02, 0.03, 0.015],
            "GOOG": [0.01, 0.015, 0.05]
        }, index=["AAPL", "MSFT", "GOOG"])
        
        # Optimize portfolio
        weights = self.constructor._optimize_portfolio(expected_returns, cov_matrix)
        
        assert isinstance(weights, dict)
        assert len(weights) == 3
        
        # Check if weights sum to 1
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-3)
        
        # Check if weights respect constraints
        for value in weights.values():
            assert value >= self.constructor.min_weight
            assert value <= self.constructor.max_weight


class TestAstrologicalSectorRotation:
    """Tests for the AstrologicalSectorRotation class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.constructor = AstrologicalSectorRotation()
        
        # Create sample signals
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        self.signals = pd.DataFrame({
            "XLK": [1] * 10,  # Technology
            "XLE": [1] * 10,  # Energy
            "XLF": [1] * 10,  # Financial Services
            "XLV": [1] * 10,  # Healthcare
            "GLD": [1] * 10   # Gold
        }, index=dates)
        
        # Add signal strength
        self.signals.loc["strength"] = [0.8, 0.7, 0.6, 0.5, 0.9]
        
        # Create sample market data
        self.market_data = {}
        for symbol in ["XLK", "XLE", "XLF", "XLV", "GLD"]:
            self.market_data[symbol] = pd.DataFrame({
                "Open": [100 + i for i in range(10)],
                "High": [105 + i for i in range(10)],
                "Low": [95 + i for i in range(10)],
                "Close": [102 + i for i in range(10)],
                "Volume": [1000000 for _ in range(10)]
            }, index=dates)
    
    def test_initialization(self):
        """Test constructor initialization."""
        assert self.constructor.name == "Astrological Sector Rotation"
        assert hasattr(self.constructor, "planet_sector_affinities")
        assert hasattr(self.constructor, "zodiac_sector_affinities")
        assert hasattr(self.constructor, "sector_tickers")
    
    @patch("src.trading.portfolio_construction.PlanetaryCalculator.calculate_planet_positions")
    def test_allocate_portfolio(self, mock_calculate_positions):
        """Test portfolio allocation using sector rotation."""
        # Mock planetary positions
        mock_positions = {
            "Sun": {"longitude": 300.0},  # Capricorn
            "Moon": {"longitude": 45.0},  # Taurus
            "Mercury": {"longitude": 320.0},  # Aquarius
            "Venus": {"longitude": 120.0},  # Leo
            "Jupiter": {"longitude": 240.0}  # Scorpio
        }
        mock_calculate_positions.return_value = mock_positions
        
        # Test allocation
        allocation = self.constructor.allocate_portfolio(
            self.signals, self.market_data, 100000.0, 10, datetime(2022, 1, 1))
        
        assert isinstance(allocation, dict)
        assert len(allocation) > 0
        
        # Check if allocation sums to total capital
        assert sum(allocation.values()) == pytest.approx(100000.0)
    
    def test_determine_favored_sectors(self):
        """Test determining favored sectors."""
        # Sample planetary positions
        positions = {
            "Sun": {"longitude": 300.0},  # Capricorn
            "Moon": {"longitude": 45.0},  # Taurus
            "Mercury": {"longitude": 320.0},  # Aquarius
            "Venus": {"longitude": 120.0},  # Leo
            "Jupiter": {"longitude": 240.0}  # Scorpio
        }
        
        # Get favored sectors
        sectors = self.constructor._determine_favored_sectors(positions)
        
        assert isinstance(sectors, list)
        assert len(sectors) > 0
    
    def test_get_tickers_for_sectors(self):
        """Test getting tickers for sectors."""
        # Sample sectors
        sectors = ["Technology", "Energy", "Financial Services"]
        
        # Get tickers
        tickers = self.constructor._get_tickers_for_sectors(sectors, 5)
        
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert len(tickers) <= 5
        
        # Check if tickers are from the specified sectors
        for ticker in tickers:
            found = False
            for sector in sectors:
                if ticker in self.constructor.sector_tickers[sector]:
                    found = True
                    break
            assert found


class TestAstrologicalRiskParity:
    """Tests for the AstrologicalRiskParity class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.constructor = AstrologicalRiskParity()
        
        # Create sample signals
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(100)]
        self.signals = pd.DataFrame({
            "AAPL": [1] * 100,
            "MSFT": [1] * 100,
            "GOOG": [1] * 100
        }, index=dates)
        
        # Add signal strength
        self.signals.loc["strength"] = [0.8, 0.7, 0.6]
        
        # Create sample market data with realistic returns
        self.market_data = {}
        np.random.seed(42)  # For reproducible tests
        
        for symbol, vol in [("AAPL", 0.2), ("MSFT", 0.15), ("GOOG", 0.25)]:
            # Generate random returns
            returns = np.random.normal(0.0005, vol / np.sqrt(252), 100)
            
            # Generate prices from returns
            prices = 100 * np.cumprod(1 + returns)
            
            # Create DataFrame
            self.market_data[symbol] = pd.DataFrame({
                "Open": prices * 0.99,
                "High": prices * 1.01,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": [1000000 for _ in range(100)]
            }, index=dates)
        
        # Create sample planetary data
        self.planetary_data = pd.DataFrame({
            "market_volatility_factor": np.random.uniform(0.8, 1.2, 100),
            "high_volatility_configuration": [bool(np.random.randint(0, 2)) for _ in range(100)]
        }, index=dates)
    
    def test_initialization(self):
        """Test constructor initialization."""
        assert self.constructor.name == "Astrological Risk Parity"
    
    def test_allocate_portfolio(self):
        """Test portfolio allocation using risk parity."""
        # Test allocation without planetary data
        allocation = self.constructor.allocate_portfolio(
            self.signals, self.market_data, 100000.0, 10)
        
        assert isinstance(allocation, dict)
        assert len(allocation) == 3  # All three assets have positive signals
        
        # Check if allocation sums to total capital
        assert sum(allocation.values()) == pytest.approx(100000.0)
        
        # Test allocation with planetary data
        allocation = self.constructor.allocate_portfolio(
            self.signals, self.market_data, 100000.0, 10, self.planetary_data)
        
        assert isinstance(allocation, dict)
        assert len(allocation) == 3
        
        # Check if allocation sums to total capital
        assert sum(allocation.values()) == pytest.approx(100000.0)
    
    def test_adjust_volatilities(self):
        """Test volatility adjustment."""
        # Sample volatilities
        volatilities = {
            "AAPL": 0.2,
            "MSFT": 0.15,
            "GOOG": 0.25
        }
        
        # Get latest planetary data
        latest_data = self.planetary_data.iloc[-1]
        
        # Adjust volatilities
        adjusted = self.constructor._adjust_volatilities(volatilities, self.planetary_data)
        
        assert isinstance(adjusted, dict)
        assert len(adjusted) == 3
        
        # Check if volatilities are adjusted
        if latest_data["market_volatility_factor"] != 1.0:
            for symbol in volatilities:
                assert adjusted[symbol] != volatilities[symbol]


class TestCreatePortfolioConstructor:
    """Tests for the create_portfolio_constructor function."""
    
    def test_create_portfolio_constructor(self):
        """Test creating different portfolio constructors."""
        # Test basic constructor
        constructor = create_portfolio_constructor("basic")
        assert isinstance(constructor, PortfolioConstructor)
        
        # Test MPT constructor
        constructor = create_portfolio_constructor("mpt")
        assert isinstance(constructor, MPTPortfolioConstructor)
        
        # Test sector rotation constructor
        constructor = create_portfolio_constructor("sector_rotation")
        assert isinstance(constructor, AstrologicalSectorRotation)
        
        # Test risk parity constructor
        constructor = create_portfolio_constructor("risk_parity")
        assert isinstance(constructor, AstrologicalRiskParity)
        
        # Test with invalid type
        constructor = create_portfolio_constructor("invalid")
        assert isinstance(constructor, PortfolioConstructor)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
