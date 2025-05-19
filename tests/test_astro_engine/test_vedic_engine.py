"""
Tests for the Vedic Astronomical Computation Engine.

This module tests the functionality of the Vedic astrological calculations,
including planetary positions, divisional charts, dashas, and financial yogas.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import tempfile
import json

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.astro_engine.planetary_positions import (
    PlanetaryCalculator, SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU
)
from src.astro_engine.vedic_dignities import (
    calculate_dignity_state, check_combustion, calculate_shadbala, calculate_all_dignities
)
from src.astro_engine.divisional_charts import DivisionalCharts
from src.astro_engine.financial_yogas import FinancialYogaAnalyzer
from src.astro_engine.dasha_systems import DashaCalculator


@pytest.fixture
def planetary_calculator():
    """Create a PlanetaryCalculator instance for testing."""
    return PlanetaryCalculator()


@pytest.fixture
def divisional_charts_calculator(planetary_calculator):
    """Create a DivisionalCharts instance for testing."""
    return DivisionalCharts(planetary_calculator)


@pytest.fixture
def financial_yoga_analyzer(planetary_calculator):
    """Create a FinancialYogaAnalyzer instance for testing."""
    return FinancialYogaAnalyzer(planetary_calculator)


@pytest.fixture
def dasha_calculator(planetary_calculator):
    """Create a DashaCalculator instance for testing."""
    return DashaCalculator(planetary_calculator)


@pytest.fixture
def sample_date():
    """Sample date for testing."""
    return "2023-01-01"


@pytest.fixture
def sample_positions(planetary_calculator, sample_date):
    """Sample planetary positions for testing."""
    return planetary_calculator.get_all_planets(sample_date)


def test_planetary_calculator_initialization(planetary_calculator):
    """Test PlanetaryCalculator initialization."""
    assert planetary_calculator is not None


def test_get_julian_day(planetary_calculator):
    """Test conversion of date to Julian Day."""
    date = "2023-01-01"
    jd = planetary_calculator.get_julian_day(date)
    
    # Julian day for 2023-01-01 should be around 2459945.5
    assert 2459945 < jd < 2459946


def test_get_planet_position(planetary_calculator, sample_date):
    """Test getting a planet's position."""
    # Get Sun position
    sun_position = planetary_calculator.get_planet_position(SUN, sample_date)
    
    # Check that the position contains expected keys
    assert "longitude" in sun_position
    assert "latitude" in sun_position
    assert "distance" in sun_position
    assert "longitude_speed" in sun_position
    assert "is_retrograde" in sun_position
    
    # Sun should be in Sagittarius or Capricorn on January 1st
    sun_sign = int(sun_position["longitude"] / 30)
    assert sun_sign in [8, 9]  # Sagittarius or Capricorn


def test_get_all_planets(planetary_calculator, sample_date):
    """Test getting positions for all planets."""
    positions = planetary_calculator.get_all_planets(sample_date)
    
    # Check that all planets are included
    expected_planets = [SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU]
    for planet in expected_planets:
        assert planet in positions
        assert "longitude" in positions[planet]


def test_get_nakshatra_details(planetary_calculator):
    """Test getting nakshatra details from longitude."""
    # Test for each nakshatra
    for i in range(27):
        longitude = i * (360 / 27) + 5  # Middle of each nakshatra
        nakshatra = planetary_calculator.get_nakshatra_details(longitude)
        
        assert "nakshatra" in nakshatra
        assert "name" in nakshatra
        assert "ruler" in nakshatra
        assert "degree" in nakshatra
        assert nakshatra["nakshatra"] == i + 1  # 1-based nakshatra


def test_calculate_vimshottari_dasha(planetary_calculator, sample_date):
    """Test calculation of Vimshottari dasha."""
    # Get Moon position
    moon_position = planetary_calculator.get_planet_position(MOON, sample_date)
    moon_longitude = moon_position["longitude"]
    
    # Calculate Vimshottari dasha
    dasha = planetary_calculator.calculate_vimshottari_dasha(moon_longitude, sample_date)
    
    # Check that the dasha contains expected keys
    assert "birth_moon_longitude" in dasha
    assert "dasha_periods" in dasha
    assert len(dasha["dasha_periods"]) > 0
    
    # Check first dasha period
    first_period = dasha["dasha_periods"][0]
    assert "planet" in first_period
    assert "start_date" in first_period
    assert "end_date" in first_period
    assert "duration_years" in first_period


def test_calculate_divisional_chart(divisional_charts_calculator):
    """Test calculation of divisional charts."""
    # Test D-9 (Navamsha) calculation
    longitude = 45.0  # 15° Taurus
    navamsha = divisional_charts_calculator.calculate_varga(longitude, 9)
    
    # Check that the result is within expected range
    assert 0 <= navamsha < 360
    
    # Test D-10 (Dashamsha) calculation
    dashamsha = divisional_charts_calculator.calculate_varga(longitude, 10)
    
    # Check that the result is within expected range
    assert 0 <= dashamsha < 360


def test_calculate_all_vargas(divisional_charts_calculator):
    """Test calculation of all standard divisional charts."""
    longitude = 75.0  # 15° Gemini
    vargas = divisional_charts_calculator.calculate_all_vargas(longitude)
    
    # Check that all standard vargas are included
    expected_vargas = [1, 2, 3, 4, 7, 9, 10, 12, 16, 20, 24, 27, 30, 40, 45, 60]
    for varga in expected_vargas:
        assert varga in vargas
        assert 0 <= vargas[varga] < 360


def test_calculate_planet_vargas(divisional_charts_calculator, sample_date):
    """Test calculation of divisional chart positions for a planet."""
    planet_vargas = divisional_charts_calculator.calculate_planet_vargas(JUPITER, sample_date)
    
    # Check that the result contains expected keys
    for varga in [1, 9, 10]:  # Check a few important vargas
        assert varga in planet_vargas
        assert "longitude" in planet_vargas[varga]
        assert "sign" in planet_vargas[varga]
        assert "sign_name" in planet_vargas[varga]
        assert "lord" in planet_vargas[varga]


def test_get_financial_strength_in_vargas(divisional_charts_calculator, sample_date):
    """Test calculation of financial strength based on divisional charts."""
    # Calculate Jupiter's positions in divisional charts
    jupiter_vargas = divisional_charts_calculator.calculate_planet_vargas(JUPITER, sample_date)
    
    # Calculate financial strength
    strength = divisional_charts_calculator.get_financial_strength_in_vargas(JUPITER, jupiter_vargas)
    
    # Check that the result contains expected keys
    assert "overall_strength" in strength
    assert "market_impact" in strength
    assert "volatility" in strength
    assert "varga_strengths" in strength
    
    # Check that the values are within expected ranges
    assert 0 <= strength["overall_strength"] <= 1
    assert 0 <= strength["market_impact"] <= 1
    assert 0 <= strength["volatility"] <= 1


def test_analyze_dhana_yogas(financial_yoga_analyzer, sample_positions):
    """Test analysis of Dhana Yogas (wealth combinations)."""
    dhana_yogas = financial_yoga_analyzer.analyze_dhana_yogas(sample_positions)
    
    # Check that the result is a list
    assert isinstance(dhana_yogas, list)
    
    # If any yogas are found, check their structure
    for yoga in dhana_yogas:
        assert "name" in yoga
        assert "strength" in yoga
        assert "description" in yoga
        assert "market_impact" in yoga
        assert "planets_involved" in yoga


def test_analyze_raja_yogas(financial_yoga_analyzer, sample_positions):
    """Test analysis of Raja Yogas (power combinations)."""
    raja_yogas = financial_yoga_analyzer.analyze_raja_yogas(sample_positions)
    
    # Check that the result is a list
    assert isinstance(raja_yogas, list)
    
    # If any yogas are found, check their structure
    for yoga in raja_yogas:
        assert "name" in yoga
        assert "strength" in yoga
        assert "description" in yoga
        assert "market_impact" in yoga
        assert "planets_involved" in yoga


def test_analyze_all_financial_yogas(financial_yoga_analyzer, sample_positions):
    """Test analysis of all financial yogas."""
    all_yogas = financial_yoga_analyzer.analyze_all_financial_yogas(sample_positions)
    
    # Check that the result contains expected keys
    assert "dhana_yogas" in all_yogas
    assert "raja_yogas" in all_yogas
    assert "trend_yogas" in all_yogas
    
    # Check that the values are lists
    assert isinstance(all_yogas["dhana_yogas"], list)
    assert isinstance(all_yogas["raja_yogas"], list)
    assert isinstance(all_yogas["trend_yogas"], list)


def test_get_market_forecast(financial_yoga_analyzer, sample_positions):
    """Test generation of market forecast based on yogas."""
    # Analyze yogas
    yogas = financial_yoga_analyzer.analyze_all_financial_yogas(sample_positions)
    
    # Generate forecast
    forecast = financial_yoga_analyzer.get_market_forecast(yogas)
    
    # Check that the result contains expected keys
    assert "trend" in forecast
    assert "confidence" in forecast
    assert "volatility" in forecast
    assert "description" in forecast
    
    # Check that the values are within expected ranges
    assert forecast["trend"] in ["bullish", "bearish", "neutral"]
    assert 0 <= forecast["confidence"] <= 1
    assert forecast["volatility"] in ["high", "moderate", "low"]


def test_calculate_vimshottari_dasha_periods(dasha_calculator, planetary_calculator, sample_date):
    """Test calculation of Vimshottari dasha periods."""
    # Get Moon position
    moon_position = planetary_calculator.get_planet_position(MOON, sample_date)
    moon_longitude = moon_position["longitude"]
    
    # Calculate Vimshottari dasha
    dasha_data = dasha_calculator.calculate_vimshottari_dasha(moon_longitude, sample_date)
    
    # Check that the result contains expected keys
    assert "birth_date" in dasha_data
    assert "birth_nakshatra" in dasha_data
    assert "birth_nakshatra_progression" in dasha_data
    assert "dasha_sequence" in dasha_data
    
    # Check that the dasha sequence contains expected data
    assert len(dasha_data["dasha_sequence"]) > 0
    
    first_dasha = dasha_data["dasha_sequence"][0]
    assert "planet" in first_dasha
    assert "start_date" in first_dasha
    assert "end_date" in first_dasha
    assert "duration_years" in first_dasha
    assert "antardashas" in first_dasha
    
    # Check that antardashas are calculated
    assert len(first_dasha["antardashas"]) > 0
    
    first_antardasha = first_dasha["antardashas"][0]
    assert "planet" in first_antardasha
    assert "start_date" in first_antardasha
    assert "end_date" in first_antardasha
    assert "duration_days" in first_antardasha
    assert "pratyantardashas" in first_antardasha
    
    # Check that pratyantardashas are calculated
    assert len(first_antardasha["pratyantardashas"]) > 0


def test_find_current_dasha(dasha_calculator, planetary_calculator, sample_date):
    """Test finding the current dasha for a given date."""
    # Get Moon position
    moon_position = planetary_calculator.get_planet_position(MOON, sample_date)
    moon_longitude = moon_position["longitude"]
    
    # Calculate Vimshottari dasha
    dasha_data = dasha_calculator.calculate_vimshottari_dasha(moon_longitude, sample_date)
    
    # Find current dasha
    current_date = datetime.fromisoformat(sample_date) + timedelta(days=365)
    current_dasha = dasha_calculator.find_current_dasha(dasha_data, current_date)
    
    # Check that the result contains expected keys
    assert "date" in current_dasha
    assert "mahadasha" in current_dasha
    assert "antardasha" in current_dasha
    assert "pratyantardasha" in current_dasha
    
    # Check that the dasha details are present
    assert current_dasha["mahadasha"] is not None
    assert "planet" in current_dasha["mahadasha"]
    assert "start_date" in current_dasha["mahadasha"]
    assert "end_date" in current_dasha["mahadasha"]
    assert "elapsed_percentage" in current_dasha["mahadasha"]


def test_get_financial_forecast_from_dasha(dasha_calculator, planetary_calculator, sample_date):
    """Test generation of financial forecast based on dasha periods."""
    # Get Moon position
    moon_position = planetary_calculator.get_planet_position(MOON, sample_date)
    moon_longitude = moon_position["longitude"]
    
    # Calculate Vimshottari dasha
    dasha_data = dasha_calculator.calculate_vimshottari_dasha(moon_longitude, sample_date)
    
    # Find current dasha
    current_date = datetime.fromisoformat(sample_date) + timedelta(days=365)
    current_dasha = dasha_calculator.find_current_dasha(dasha_data, current_date)
    
    # Generate forecast
    forecast = dasha_calculator.get_financial_forecast_from_dasha(current_dasha)
    
    # Check that the result contains expected keys
    assert "trend" in forecast
    assert "volatility" in forecast
    assert "description" in forecast
    
    # Check that the values are within expected ranges
    assert forecast["trend"] in ["bullish", "bearish", "neutral"]
    assert forecast["volatility"] in ["extreme", "very high", "high", "moderate", "low"]


def test_dignity_state_calculation():
    """Test calculation of planetary dignity states."""
    # Test Sun in Leo (own sign)
    sun_in_leo = calculate_dignity_state(SUN, 130.0)  # 10° Leo
    assert sun_in_leo["state"] == "own_sign"
    assert sun_in_leo["strength"] > 0.8
    
    # Test Jupiter in Cancer (exaltation)
    jupiter_in_cancer = calculate_dignity_state(JUPITER, 95.0)  # 5° Cancer
    assert jupiter_in_cancer["state"] == "exalted"
    assert jupiter_in_cancer["strength"] > 0.9
    assert jupiter_in_cancer["exact_degree"] is True
    
    # Test Saturn in Aries (debilitation)
    saturn_in_aries = calculate_dignity_state(SATURN, 5.0)  # 5° Aries
    assert saturn_in_aries["state"] == "debilitated"
    assert saturn_in_aries["strength"] < 0.3


def test_combustion_check():
    """Test checking if a planet is combust."""
    # Test Mercury very close to Sun (combust)
    mercury_combust = check_combustion(MERCURY, 100.0, 101.0)
    assert mercury_combust["is_combust"] is True
    assert mercury_combust["combustion_degree"] > 0.9
    
    # Test Venus far from Sun (not combust)
    venus_not_combust = check_combustion(VENUS, 100.0, 150.0)
    assert venus_not_combust["is_combust"] is False
    assert venus_not_combust["combustion_degree"] == 0.0


def test_edge_case_out_of_range_longitude(planetary_calculator):
    """Test handling of out-of-range longitude values."""
    # Test negative longitude
    nakshatra = planetary_calculator.get_nakshatra_details(-10.0)
    assert 1 <= nakshatra["nakshatra"] <= 27
    
    # Test longitude > 360
    nakshatra = planetary_calculator.get_nakshatra_details(370.0)
    assert 1 <= nakshatra["nakshatra"] <= 27


def test_edge_case_invalid_varga(divisional_charts_calculator):
    """Test handling of invalid varga numbers."""
    longitude = 100.0
    
    # Test invalid varga number
    with pytest.raises(ValueError):
        divisional_charts_calculator.calculate_varga(longitude, 15)


def test_failure_case_invalid_planet(planetary_calculator, sample_date):
    """Test handling of invalid planet IDs."""
    # Test invalid planet ID
    with pytest.raises(Exception):
        planetary_calculator.get_planet_position(999, sample_date)


def test_integration_full_analysis(planetary_calculator, divisional_charts_calculator, 
                                  financial_yoga_analyzer, dasha_calculator, sample_date):
    """Test full integration of all components."""
    # Get planetary positions
    positions = planetary_calculator.get_all_planets(sample_date)
    
    # Calculate divisional charts for Jupiter
    jupiter_vargas = divisional_charts_calculator.calculate_planet_vargas(JUPITER, sample_date)
    jupiter_strength = divisional_charts_calculator.get_financial_strength_in_vargas(JUPITER, jupiter_vargas)
    
    # Analyze financial yogas
    yogas = financial_yoga_analyzer.analyze_all_financial_yogas(positions)
    yoga_forecast = financial_yoga_analyzer.get_market_forecast(yogas)
    
    # Calculate dasha periods
    moon_longitude = positions[MOON]["longitude"]
    dasha_data = dasha_calculator.calculate_vimshottari_dasha(moon_longitude, sample_date)
    current_dasha = dasha_calculator.find_current_dasha(dasha_data, sample_date)
    dasha_forecast = dasha_calculator.get_financial_forecast_from_dasha(current_dasha)
    
    # Check that all components produce valid results
    assert jupiter_strength["overall_strength"] >= 0
    assert yoga_forecast["trend"] in ["bullish", "bearish", "neutral"]
    assert dasha_forecast["trend"] in ["bullish", "bearish", "neutral"]
    
    # The test passes if all components work together without errors
