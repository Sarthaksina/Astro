"""
Tests for the signal generator module.

This module contains unit tests for the signal generator components, including:
- Base signal generator
- Vedic nakshatra signal generator
- Vedic yoga signal generator
- Vedic dasha signal generator
- Combined signal generator
- Signal filter
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.trading.signal_generator import (
    SignalGenerator, 
    VedicNakshatraSignalGenerator,
    VedicYogaSignalGenerator, 
    VedicDashaSignalGenerator,
    CombinedSignalGenerator,
    SignalFilter
)


class TestSignalGenerator:
    """Tests for the base SignalGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SignalGenerator("Test Generator")
        
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
            "moon_longitude": [i * 12 for i in range(10)]
        }, index=dates)
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.name == "Test Generator"
        assert hasattr(self.generator, "calculator")
        assert hasattr(self.generator, "feature_generator")
    
    def test_generate_signals(self):
        """Test base signal generation."""
        signals = self.generator.generate_signals(self.market_data, self.planetary_data)
        
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == len(self.market_data)
        assert "signal" in signals.columns
        assert "strength" in signals.columns
        assert "reason" in signals.columns
        
        # Base implementation should return no signals
        assert (signals["signal"] == 0).all()


class TestVedicNakshatraSignalGenerator:
    """Tests for the VedicNakshatraSignalGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = VedicNakshatraSignalGenerator()
        
        # Create sample market data
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        self.market_data = pd.DataFrame({
            "Open": [100 + i for i in range(10)],
            "High": [105 + i for i in range(10)],
            "Low": [95 + i for i in range(10)],
            "Close": [102 + i for i in range(10)],
            "Volume": [1000000 for _ in range(10)]
        }, index=dates)
        
        # Create sample planetary data with nakshatra information
        self.planetary_data = pd.DataFrame({
            "moon_nakshatra": ["Ashwini", "Bharani", "Krittika", "Rohini", 
                              "Mrigashira", "Ardra", "Punarvasu", "Pushya", 
                              "Ashlesha", "Magha"],
            "moon_nakshatra_pada": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
        }, index=dates)
        
        # Add financial classification
        self.planetary_data["moon_nakshatra_financial"] = ["bullish", "bearish", "bearish", "bullish", 
                                                         "bullish", "bearish", "neutral", "bullish", 
                                                         "bearish", "neutral"]
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.name == "Nakshatra Signal Generator"
        assert hasattr(self.generator, "bullish_nakshatras")
        assert hasattr(self.generator, "bearish_nakshatras")
        assert hasattr(self.generator, "pada_strength_modifiers")
    
    def test_generate_signals(self):
        """Test nakshatra signal generation."""
        signals = self.generator.generate_signals(self.market_data, self.planetary_data)
        
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == len(self.market_data)
        
        # Check if signals are generated for bullish and bearish nakshatras
        bullish_signals = signals[signals["signal"] > 0]
        bearish_signals = signals[signals["signal"] < 0]
        
        # Count expected signals
        expected_bullish = (self.planetary_data["moon_nakshatra_financial"] == "bullish").sum()
        expected_bearish = (self.planetary_data["moon_nakshatra_financial"] == "bearish").sum()
        
        # Allow for some signals to be filtered out due to strength threshold
        assert len(bullish_signals) <= expected_bullish
        assert len(bearish_signals) <= expected_bearish
        
        # Check if signal strengths are valid
        if not bullish_signals.empty:
            assert (bullish_signals["strength"] > 0).all() and (bullish_signals["strength"] <= 1).all()
        
        if not bearish_signals.empty:
            assert (bearish_signals["strength"] > 0).all() and (bearish_signals["strength"] <= 1).all()
        
        # Check if pada affects signal strength
        if not signals[signals["signal"] != 0].empty:
            for date in signals[signals["signal"] != 0].index:
                pada = self.planetary_data.loc[date, "moon_nakshatra_pada"]
                assert pada in self.generator.pada_strength_modifiers


class TestVedicYogaSignalGenerator:
    """Tests for the VedicYogaSignalGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = VedicYogaSignalGenerator()
        
        # Create sample market data
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        self.market_data = pd.DataFrame({
            "Open": [100 + i for i in range(10)],
            "High": [105 + i for i in range(10)],
            "Low": [95 + i for i in range(10)],
            "Close": [102 + i for i in range(10)],
            "Volume": [1000000 for _ in range(10)]
        }, index=dates)
        
        # Create sample planetary data with yoga information
        self.planetary_data = pd.DataFrame({
            "bullish_yoga_count": [3, 2, 1, 0, 0, 1, 2, 4, 1, 0],
            "bearish_yoga_count": [1, 1, 2, 3, 4, 2, 1, 0, 3, 2],
            "neutral_yoga_count": [2, 3, 3, 2, 1, 3, 2, 1, 1, 3]
        }, index=dates)
        
        # Add specific yoga details for some days
        self.planetary_data["bullish_yogas"] = [
            ["Gajakesari", "Amala", "Budhaditya"],
            ["Gajakesari", "Amala"],
            ["Budhaditya"],
            [],
            [],
            ["Gajakesari"],
            ["Amala", "Budhaditya"],
            ["Gajakesari", "Amala", "Budhaditya", "Dhana"],
            ["Gajakesari"],
            []
        ]
        
        self.planetary_data["bearish_yogas"] = [
            ["Shakata"],
            ["Shakata"],
            ["Shakata", "Kemadruma"],
            ["Shakata", "Kemadruma", "Daridra"],
            ["Shakata", "Kemadruma", "Daridra", "Sakata"],
            ["Kemadruma", "Daridra"],
            ["Shakata"],
            [],
            ["Shakata", "Kemadruma", "Daridra"],
            ["Kemadruma", "Daridra"]
        ]
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.name == "Yoga Signal Generator"
        assert self.generator.min_yoga_strength == 0.6
    
    def test_generate_signals(self):
        """Test yoga signal generation."""
        signals = self.generator.generate_signals(self.market_data, self.planetary_data)
        
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == len(self.market_data)
        
        # Check if signals are generated based on yoga counts
        for date in signals.index:
            bullish_count = self.planetary_data.loc[date, "bullish_yoga_count"]
            bearish_count = self.planetary_data.loc[date, "bearish_yoga_count"]
            
            if bullish_count > bearish_count and bullish_count > 0:
                # Should be a bullish signal if strong enough
                yoga_strength = min(bullish_count * 0.2, 0.8)
                if yoga_strength >= self.generator.min_yoga_strength:
                    assert signals.loc[date, "signal"] == 1
            
            elif bearish_count > bullish_count and bearish_count > 0:
                # Should be a bearish signal if strong enough
                yoga_strength = min(bearish_count * 0.2, 0.8)
                if yoga_strength >= self.generator.min_yoga_strength:
                    assert signals.loc[date, "signal"] == -1


class TestVedicDashaSignalGenerator:
    """Tests for the VedicDashaSignalGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = VedicDashaSignalGenerator()
        
        # Create sample market data
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        self.market_data = pd.DataFrame({
            "Open": [100 + i for i in range(10)],
            "High": [105 + i for i in range(10)],
            "Low": [95 + i for i in range(10)],
            "Close": [102 + i for i in range(10)],
            "Volume": [1000000 for _ in range(10)]
        }, index=dates)
        
        # Create sample planetary data with dasha information
        self.planetary_data = pd.DataFrame({
            "current_dasha_lord": ["Jupiter", "Venus", "Mercury", "Saturn", 
                                  "Mars", "Rahu", "Moon", "Sun", "Ketu", "Venus"],
            "current_antardasha_lord": ["Venus", "Mercury", "Jupiter", "Rahu", 
                                       "Saturn", "Ketu", "Sun", "Mars", "Moon", "Jupiter"]
        }, index=dates)
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.name == "Dasha Signal Generator"
        assert hasattr(self.generator, "dasha_financial_map")
        assert self.generator.min_signal_strength == 0.6
    
    def test_generate_signals(self):
        """Test dasha signal generation."""
        signals = self.generator.generate_signals(self.market_data, self.planetary_data)
        
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == len(self.market_data)
        
        # Check if signals are generated based on dasha lords
        for date in signals.index:
            dasha_lord = self.planetary_data.loc[date, "current_dasha_lord"]
            dasha_info = self.generator.dasha_financial_map.get(dasha_lord, {})
            
            if dasha_info.get("nature") == "bullish" and dasha_info.get("strength", 0) >= self.generator.min_signal_strength:
                assert signals.loc[date, "signal"] == 1
            
            elif dasha_info.get("nature") == "bearish" and (1 - dasha_info.get("strength", 0)) >= self.generator.min_signal_strength:
                assert signals.loc[date, "signal"] == -1


class TestCombinedSignalGenerator:
    """Tests for the CombinedSignalGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = CombinedSignalGenerator()
        
        # Create sample market data
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        self.market_data = pd.DataFrame({
            "Open": [100 + i for i in range(10)],
            "High": [105 + i for i in range(10)],
            "Low": [95 + i for i in range(10)],
            "Close": [102 + i for i in range(10)],
            "Volume": [1000000 for _ in range(10)]
        }, index=dates)
        
        # Create sample planetary data with all required information
        self.planetary_data = pd.DataFrame({
            # Nakshatra information
            "moon_nakshatra": ["Ashwini", "Bharani", "Krittika", "Rohini", 
                              "Mrigashira", "Ardra", "Punarvasu", "Pushya", 
                              "Ashlesha", "Magha"],
            "moon_nakshatra_pada": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
            "moon_nakshatra_financial": ["bullish", "bearish", "bearish", "bullish", 
                                        "bullish", "bearish", "neutral", "bullish", 
                                        "bearish", "neutral"],
            
            # Yoga information
            "bullish_yoga_count": [3, 2, 1, 0, 0, 1, 2, 4, 1, 0],
            "bearish_yoga_count": [1, 1, 2, 3, 4, 2, 1, 0, 3, 2],
            
            # Dasha information
            "current_dasha_lord": ["Jupiter", "Venus", "Mercury", "Saturn", 
                                  "Mars", "Rahu", "Moon", "Sun", "Ketu", "Venus"],
            "current_antardasha_lord": ["Venus", "Mercury", "Jupiter", "Rahu", 
                                       "Saturn", "Ketu", "Sun", "Mars", "Moon", "Jupiter"]
        }, index=dates)
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.name == "Combined Signal Generator"
        assert len(self.generator.generators) == 3  # Default generators
        assert sum(self.generator.weights.values()) == pytest.approx(1.0)  # Weights should sum to 1
    
    def test_add_generator(self):
        """Test adding a new generator."""
        # Create a new generator
        new_generator = SignalGenerator("New Generator")
        
        # Add it to the combined generator
        self.generator.add_generator(new_generator, 2.0)
        
        assert len(self.generator.generators) == 4
        assert new_generator in self.generator.generators
        assert "New Generator" in self.generator.weights
        assert sum(self.generator.weights.values()) == pytest.approx(1.0)  # Weights should still sum to 1
    
    def test_generate_signals(self):
        """Test combined signal generation."""
        signals = self.generator.generate_signals(self.market_data, self.planetary_data)
        
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == len(self.market_data)
        
        # Check if signals are generated
        assert (signals["signal"] != 0).any()
        
        # Check if signal strengths are valid
        signal_rows = signals[signals["signal"] != 0]
        if not signal_rows.empty:
            assert (signal_rows["strength"] > 0).all() and (signal_rows["strength"] <= 1).all()
            
            # Check if reasons include generator names
            for reason in signal_rows["reason"]:
                assert any(gen.name in reason for gen in self.generator.generators)


class TestSignalFilter:
    """Tests for the SignalFilter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.filter = SignalFilter()
        
        # Create sample dates
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        
        # Create sample signals
        self.signals = pd.DataFrame({
            "signal": [1, 1, 0, -1, -1, 0, 1, 1, -1, 0],
            "strength": [0.8, 0.4, 0.0, 0.9, 0.7, 0.0, 0.6, 0.7, 0.3, 0.0],
            "reason": ["Reason 1", "Reason 2", "", "Reason 4", "Reason 5", 
                      "", "Reason 7", "Reason 8", "Reason 9", ""]
        }, index=dates)
        
        # Create sample market data
        self.market_data = pd.DataFrame({
            "Close": [100, 102, 103, 101, 98, 97, 99, 102, 104, 103],
            "Volume": [1000000 for _ in range(10)]
        }, index=dates)
        
        # Add momentum
        self.market_data["momentum_5d"] = self.market_data["Close"].pct_change(5)
    
    def test_filter_signals(self):
        """Test signal filtering."""
        filtered = self.filter.filter_signals(self.signals)
        
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == len(self.signals)
        
        # Check if weak signals are removed
        assert filtered.loc[self.signals["strength"] < 0.5, "signal"].equals(pd.Series([0, 0, 0], index=filtered.loc[self.signals["strength"] < 0.5].index))
        
        # Check if consecutive signals are filtered
        assert not (filtered["signal"] == 1).all()  # Not all signals should be 1
        
        # Test with market data
        filtered_with_market = self.filter.filter_signals(self.signals, self.market_data)
        assert isinstance(filtered_with_market, pd.DataFrame)
        assert len(filtered_with_market) == len(self.signals)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
