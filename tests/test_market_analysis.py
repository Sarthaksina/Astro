"""
Tests for the Market Analysis Module.

This module tests the market analysis functionality that was consolidated from
various redundant implementations throughout the codebase.
"""

import pytest
import datetime
from src.astro_engine.market_analysis import analyze_market_trend, analyze_comprehensive_market_forecast
from src.astro_engine.planetary_positions import PlanetaryCalculator
from src.astro_engine.financial_yogas import FinancialYogaAnalyzer
from src.astro_engine.constants import SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN


def test_analyze_market_trend():
    """Test the analyze_market_trend function"""
    # Create a mock calculator
    calculator = PlanetaryCalculator()
    
    # Create mock planetary positions
    mock_positions = {
        SUN: {
            "longitude": 120.5,
            "latitude": 0.0,
            "is_retrograde": False,
            "nakshatra": 10,
            "longitude_speed": 1.0
        },
        MOON: {
            "longitude": 75.3,
            "latitude": 5.0,
            "is_retrograde": False,
            "nakshatra": 6,
            "longitude_speed": 13.0
        },
        MERCURY: {
            "longitude": 118.2,
            "latitude": 1.2,
            "is_retrograde": True,
            "nakshatra": 9,
            "longitude_speed": -0.5
        },
        JUPITER: {
            "longitude": 200.7,
            "latitude": 0.1,
            "is_retrograde": False,
            "nakshatra": 14,
            "longitude_speed": 0.2
        },
        SATURN: {
            "longitude": 300.1,
            "latitude": 0.3,
            "is_retrograde": False,
            "nakshatra": 21,
            "longitude_speed": 0.1
        }
    }
    
    # Mock the get_nakshatra_details method to return predictable results
    def mock_get_nakshatra_details(longitude):
        return {
            "nakshatra_name": "Test Nakshatra",
            "financial_nature": "bullish" if longitude < 100 else "bearish"
        }
    
    # Replace the actual method with our mock
    calculator.get_nakshatra_details = mock_get_nakshatra_details
    
    # Test the function
    result = analyze_market_trend(mock_positions, "2023-01-01", calculator)
    
    # Verify the structure of the result
    assert isinstance(result, dict)
    assert "trend" in result
    assert "strength" in result
    assert "key_factors" in result
    assert "aspects" in result
    
    # Verify that Mercury retrograde is detected
    assert any("Mercury retrograde" in factor for factor in result["key_factors"])
    
    # Verify that the Moon's nakshatra is analyzed
    assert any("Moon in" in factor for factor in result["key_factors"])


def test_analyze_comprehensive_market_forecast():
    """Test the analyze_comprehensive_market_forecast function"""
    # Create a mock calculator
    calculator = PlanetaryCalculator()
    
    # Create a mock yoga analyzer
    yoga_analyzer = FinancialYogaAnalyzer(calculator)
    
    # Mock the get_all_planets method to return predictable results
    def mock_get_all_planets(date):
        return {
            SUN: {
                "longitude": 120.5,
                "latitude": 0.0,
                "is_retrograde": False,
                "nakshatra": 10,
                "longitude_speed": 1.0
            },
            MOON: {
                "longitude": 75.3,
                "latitude": 5.0,
                "is_retrograde": False,
                "nakshatra": 6,
                "longitude_speed": 13.0
            },
            MERCURY: {
                "longitude": 118.2,
                "latitude": 1.2,
                "is_retrograde": True,
                "nakshatra": 9,
                "longitude_speed": -0.5
            },
            JUPITER: {
                "longitude": 200.7,
                "latitude": 0.1,
                "is_retrograde": False,
                "nakshatra": 14,
                "longitude_speed": 0.2
            },
            SATURN: {
                "longitude": 300.1,
                "latitude": 0.3,
                "is_retrograde": False,
                "nakshatra": 21,
                "longitude_speed": 0.1
            }
        }
    
    # Mock the analyze_all_financial_yogas method to return predictable results
    def mock_analyze_all_financial_yogas(positions):
        return {
            "dhana_yogas": [],
            "raja_yogas": [],
            "market_trend_yogas": []
        }
    
    # Mock the get_market_forecast method to return predictable results
    def mock_get_market_forecast(yogas):
        return {
            "trend": "bullish",
            "confidence": 0.75,
            "volatility": "moderate",
            "description": "Test forecast description"
        }
    
    # Replace the actual methods with our mocks
    calculator.get_all_planets = mock_get_all_planets
    yoga_analyzer.analyze_all_financial_yogas = mock_analyze_all_financial_yogas
    yoga_analyzer.get_market_forecast = mock_get_market_forecast
    
    # Test the function
    result = analyze_comprehensive_market_forecast("2023-01-01", calculator, yoga_analyzer)
    
    # Verify the structure of the result
    assert isinstance(result, dict)
    assert "trend" in result
    assert "confidence" in result
    assert "volatility" in result
    assert "key_factors" in result
    assert "aspects" in result
    assert "yogas" in result
    assert "description" in result
    
    # Verify that the confidence is a float between 0 and 1
    assert 0 <= result["confidence"] <= 1
    
    # Verify that the trend is one of the expected values
    assert result["trend"] in ["Bullish", "Bearish", "Volatile", "Sideways"]
    
    # Verify that the volatility is one of the expected values
    assert result["volatility"] in ["High", "Moderate", "Low"]


if __name__ == "__main__":
    pytest.main(["-v", "test_market_analysis.py"])
