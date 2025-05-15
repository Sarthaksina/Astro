# Cosmic Market Oracle - Tests for Planetary Positions Module

import pytest
import datetime
import numpy as np
from src.astro_engine.planetary_positions import (
    PlanetaryCalculator, get_planetary_aspects, is_aspect_applying,
    SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO, RAHU, KETU
)


@pytest.fixture
def calculator():
    """Create a PlanetaryCalculator instance for testing."""
    calc = PlanetaryCalculator()
    yield calc
    calc.close()  # Clean up after tests


class TestPlanetaryCalculator:
    """Tests for the PlanetaryCalculator class."""
    
    def test_get_julian_day(self, calculator):
        """Test conversion of dates to Julian Day."""
        # Test with datetime object
        date = datetime.datetime(2023, 1, 1, 12, 0, 0)
        jd = calculator.get_julian_day(date)
        assert isinstance(jd, float)
        assert abs(jd - 2459946.0) < 0.1  # Approximate JD for 2023-01-01
        
        # Test with ISO format string
        jd_str = calculator.get_julian_day("2023-01-01T12:00:00")
        assert abs(jd_str - jd) < 1e-6  # Should be the same as the datetime version
    
    def test_get_planet_position(self, calculator):
        """Test calculation of planetary positions."""
        # Test Sun position for a specific date
        date = "2023-01-01"
        sun_pos = calculator.get_planet_position(SUN, date)
        
        # Verify structure of returned data
        assert isinstance(sun_pos, dict)
        assert "longitude" in sun_pos
        assert "latitude" in sun_pos
        assert "distance" in sun_pos
        assert "longitude_speed" in sun_pos
        assert "is_retrograde" in sun_pos
        assert "nakshatra" in sun_pos
        
        # Verify data types
        assert isinstance(sun_pos["longitude"], float)
        assert isinstance(sun_pos["is_retrograde"], bool)
        assert isinstance(sun_pos["nakshatra"], int)
        
        # Sun should never be retrograde
        assert sun_pos["is_retrograde"] is False
        
        # Test Mercury position (which can be retrograde)
        mercury_pos = calculator.get_planet_position(MERCURY, date)
        assert isinstance(mercury_pos["is_retrograde"], bool)
        
        # Test with heliocentric coordinates
        earth_helio = calculator.get_planet_position(3, date, heliocentric=True)  # 3 = Earth
        assert isinstance(earth_helio, dict)
        assert "longitude" in earth_helio
    
    def test_get_all_planets(self, calculator):
        """Test getting positions for all planets."""
        date = "2023-01-01"
        planets = calculator.get_all_planets(date)
        
        # Verify all planets are included
        assert SUN in planets
        assert MOON in planets
        assert JUPITER in planets
        assert SATURN in planets
        assert RAHU in planets
        
        # Test with and without nodes
        planets_no_nodes = calculator.get_all_planets(date, include_nodes=False)
        assert RAHU not in planets_no_nodes
        assert KETU not in planets_no_nodes
        
        # Test heliocentric positions
        planets_helio = calculator.get_all_planets(date, heliocentric=True)
        assert "longitude" in planets_helio[MARS]
    
    def test_get_ayanamsa(self, calculator):
        """Test calculation of ayanamsa (precession of equinoxes)."""
        date = "2023-01-01"
        ayanamsa = calculator.get_ayanamsa(date)
        
        # Verify it's a float and in a reasonable range
        assert isinstance(ayanamsa, float)
        assert 20 < ayanamsa < 30  # Approximate range for current era


def test_get_planetary_aspects():
    """Test calculation of aspects between planets."""
    # Create sample planetary positions
    positions = {
        SUN: {"longitude": 0.0, "longitude_speed": 1.0},
        MOON: {"longitude": 90.0, "longitude_speed": 13.0},
        JUPITER: {"longitude": 120.0, "longitude_speed": 0.1},
        SATURN: {"longitude": 180.0, "longitude_speed": -0.1}  # Retrograde
    }
    
    aspects = get_planetary_aspects(positions, orb=6.0)
    
    # Verify aspects are detected correctly
    assert isinstance(aspects, list)
    assert len(aspects) > 0
    
    # Check for specific aspects
    sun_moon_square = None
    jupiter_saturn_opposition = None
    
    for aspect in aspects:
        if (aspect["planet1"] == SUN and aspect["planet2"] == MOON and aspect["aspect_type"] == "Square") or \
           (aspect["planet1"] == MOON and aspect["planet2"] == SUN and aspect["aspect_type"] == "Square"):
            sun_moon_square = aspect
        
        if (aspect["planet1"] == JUPITER and aspect["planet2"] == SATURN and aspect["aspect_type"] == "Opposition") or \
           (aspect["planet1"] == SATURN and aspect["planet2"] == JUPITER and aspect["aspect_type"] == "Opposition"):
            jupiter_saturn_opposition = aspect
    
    assert sun_moon_square is not None, "Sun-Moon square aspect not detected"
    assert jupiter_saturn_opposition is not None, "Jupiter-Saturn opposition not detected"
    
    # Test with different orb
    narrow_aspects = get_planetary_aspects(positions, orb=1.0)
    assert len(narrow_aspects) <= len(aspects), "Narrower orb should detect fewer aspects"


def test_is_aspect_applying():
    """Test determination of applying vs separating aspects."""
    # Create sample planetary positions
    positions = {
        SUN: {"longitude": 0.0, "longitude_speed": 1.0},
        MOON: {"longitude": 178.0, "longitude_speed": 13.0},  # Approaching opposition
        JUPITER: {"longitude": 182.0, "longitude_speed": -0.1},  # Moving away from opposition
    }
    
    # Sun-Moon opposition is applying (getting closer)
    assert is_aspect_applying(positions, SUN, MOON, 180.0) is True
    
    # Sun-Jupiter opposition is separating (moving apart)
    assert is_aspect_applying(positions, SUN, JUPITER, 180.0) is False
    
    # Test conjunction
    positions2 = {
        VENUS: {"longitude": 358.0, "longitude_speed": 1.2},
        MARS: {"longitude": 2.0, "longitude_speed": 0.8},
    }
    
    # Venus-Mars conjunction is applying
    assert is_aspect_applying(positions2, VENUS, MARS, 0.0) is True
    
    # Test with planets moving in opposite directions
    positions3 = {
        MERCURY: {"longitude": 85.0, "longitude_speed": 1.0},
        SATURN: {"longitude": 95.0, "longitude_speed": -0.5},
    }
    
    # Mercury-Saturn square is applying (both moving toward 90°)
    assert is_aspect_applying(positions3, MERCURY, SATURN, 90.0) is True
