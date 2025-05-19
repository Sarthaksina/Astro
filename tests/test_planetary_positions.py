# Cosmic Market Oracle - Tests for Planetary Positions Module

import pytest
import datetime
from src.astro_engine.planetary_positions import PlanetaryCalculator, get_planetary_aspects, is_aspect_applying


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


def test_is_aspect_applying():
    """Test the is_aspect_applying function"""
    # Test case 1: Applying conjunction (planets moving toward each other)
    # Planet 1 at 10° moving forward at 1°/day
    # Planet 2 at 15° moving backward at -0.5°/day
    # They are getting closer, so aspect should be applying
    assert is_aspect_applying(10.0, 15.0, 1.0, -0.5, 0) is True
    
    # Test case 2: Separating conjunction (planets moving away from each other)
    # Planet 1 at 10° moving backward at -1°/day
    # Planet 2 at 15° moving forward at 0.5°/day
    # They are moving apart, so aspect should be separating
    assert is_aspect_applying(10.0, 15.0, -1.0, 0.5, 0) is False
    
    # Test case 3: Applying opposition (planets moving toward 180° difference)
    # Planet 1 at 10° moving forward at 1°/day
    # Planet 2 at 170° moving backward at -0.5°/day
    # They are moving toward opposition, so aspect should be applying
    assert is_aspect_applying(10.0, 170.0, 1.0, -0.5, 180) is True
    
    # Test case 4: Separating trine (planets moving away from 120° difference)
    # Planet 1 at 10° moving forward at 1°/day
    # Planet 2 at 130° moving forward at 1.5°/day
    # Second planet is moving faster, so they're separating from trine
    assert is_aspect_applying(10.0, 130.0, 1.0, 1.5, 120) is False
    
    # Test case 5: Edge case with planets at exact aspect
    # Planet 1 at 90° moving forward at 1°/day
    # Planet 2 at 0° moving forward at 0.5°/day
    # First planet is moving faster away from square, so aspect should be separating
    assert is_aspect_applying(90.0, 0.0, 1.0, 0.5, 90) is False