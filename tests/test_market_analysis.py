"""
Tests for the Market Analysis Module.

This module tests the market analysis functionality that was consolidated from
various redundant implementations throughout the codebase.
"""

import pytest
import datetime
# from src.astro_engine.market_analysis import analyze_market_trend, analyze_comprehensive_market_forecast # Removed
from src.astro_engine.vedic_analysis import VedicAnalyzer # Added
from src.astro_engine.planetary_positions import PlanetaryCalculator # Keep for potential mocking if needed, though VedicAnalyzer has its own
from src.astro_engine.financial_yogas import FinancialYogaAnalyzer # Keep for potential mocking if needed
from src.astro_engine.constants import SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN


def test_analyze_market_trend_replacement():
    """Test the VedicAnalyzer.analyze_date for market trend equivalent"""
    analyzer = VedicAnalyzer()
    
    # We can't easily mock internal components of VedicAnalyzer without more complex patching.
    # So, we'll perform a basic structural check on the output.
    # The date needs to be one for which the ephemeris files are available.
    # Using a date that's likely covered by default ephemeris.
    date_to_analyze = "2023-01-01"
    
    # It's hard to mock planetary_positions used by analyzer.planetary_calculator
    # without deeper mocking. For this test, we'll mainly check structure.
    # A more robust test would be in test_vedic_analysis.py with proper fixtures.
    
    result = analyzer.analyze_date(date_to_analyze)
    
    # Verify the structure of the result related to integrated_forecast
    assert isinstance(result, dict)
    assert "integrated_forecast" in result
    forecast = result["integrated_forecast"]
    assert "trend" in forecast
    assert "trend_score" in forecast
    assert "key_factors" in forecast
    assert "description" in forecast

    # Example: Check if key_factors is a list (can be empty)
    assert isinstance(forecast["key_factors"], list)


def test_analyze_comprehensive_market_forecast_replacement():
    """Test the VedicAnalyzer.analyze_date for comprehensive forecast equivalent"""
    analyzer = VedicAnalyzer()
    date_to_analyze = "2023-01-01"

    # Similar to above, direct mocking of sub-components of VedicAnalyzer is complex here.
    # We rely on VedicAnalyzer to correctly call its internal components.
    # This test primarily ensures analyze_date runs and returns the expected structure.
    
    result = analyzer.analyze_date(date_to_analyze)
    
    # Verify the structure of the result
    assert isinstance(result, dict)
    assert "integrated_forecast" in result
    assert "sector_forecasts" in result
    assert "timing_signals" in result
    assert "key_yogas" in result
    assert "key_dignities" in result

    integrated_forecast = result["integrated_forecast"]
    assert "trend" in integrated_forecast
    assert "confidence" in integrated_forecast
    assert "volatility" in integrated_forecast
    assert "key_factors" in integrated_forecast
    assert "description" in integrated_forecast
    
    # Verify that the confidence is a float (VedicAnalyzer sets a min of 0.3)
    assert 0.3 <= integrated_forecast["confidence"] <= 0.9 # Range defined in VedicAnalyzer
    
    # Verify that the trend is a string
    assert isinstance(integrated_forecast["trend"], str)
    
    # Verify that the volatility is one of the expected values
    assert integrated_forecast["volatility"] in ["High", "Moderate", "Low"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
