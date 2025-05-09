# Cosmic Market Oracle - Tests for Planetary Positions Module

import pytest
import datetime
from src.astro_engine.planetary_positions import PlanetaryCalculator, get_planetary_aspects


def test_planetary_calculator_initialization():
    """Test that the PlanetaryCalculator can be initialized"""
    calculator = PlanetaryCalculator()
    assert calculator is not None


def test_julian_day_calculation():
    """Test Julian Day calculation"""
    calculator = PlanetaryCalculator()
    
    # Test with a known date
    test_date = datetime.datetime(2000, 1, 1, 12, 0, 0)
    jd = calculator.get_julian_day(test_date)
    
    # J2000.0 epoch is defined as 2000-01-01T12:00:00 TT, which is JD 2451545.0
    assert abs(jd - 2451545.0) < 0.001


def test_get_planet_position():
    """Test retrieving a planet's position"""
    calculator = PlanetaryCalculator()
    
    # Use a fixed date for consistent testing
    test_date = datetime.datetime(2023, 1, 1, 12, 0, 0)
    
    # Get Sun position
    sun_position = calculator.get_planet_position(0, test_date)  # 0 = Sun
    
    # Basic validation of returned data structure
    assert isinstance(sun_position, dict)
    assert "longitude" in sun_position
    assert "latitude" in sun_position
    assert "is_retrograde" in sun_position
    assert "nakshatra" in sun_position
    
    # Sun should never be retrograde
    assert sun_position["is_retrograde"] is False
    
    # Longitude should be between 0 and 360
    assert 0 <= sun_position["longitude"] < 360
    
    # Nakshatra should be between 1 and 27
    assert 1 <= sun_position["nakshatra"] <= 27


def test_get_all_planets():
    """Test retrieving positions for all planets"""
    calculator = PlanetaryCalculator()
    
    # Use a fixed date for consistent testing
    test_date = datetime.datetime(2023, 1, 1, 12, 0, 0)
    
    # Get all planet positions
    positions = calculator.get_all_planets(test_date)
    
    # Should have entries for all major planets
    assert len(positions) >= 10  # At least Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto
    
    # Check that each planet has the expected data
    for planet_id, position in positions.items():
        assert "longitude" in position
        assert "latitude" in position
        assert "is_retrograde" in position
        assert "nakshatra" in position


def test_planetary_aspects():
    """Test calculating aspects between planets"""
    calculator = PlanetaryCalculator()
    
    # Use a fixed date for consistent testing
    test_date = datetime.datetime(2023, 1, 1, 12, 0, 0)
    
    # Get all planet positions
    positions = calculator.get_all_planets(test_date)
    
    # Calculate aspects
    aspects = get_planetary_aspects(positions)
    
    # Basic validation of returned data
    assert isinstance(aspects, list)
    
    # If there are aspects, check their structure
    if aspects:
        aspect = aspects[0]
        assert "planet1" in aspect
        assert "planet2" in aspect
        assert "aspect_type" in aspect
        assert "orb" in aspect
        assert "applying" in aspect