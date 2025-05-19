"""
Centralized constants for the Cosmic Market Oracle.
Contains all planetary and astrological constants used across the application.
"""

import swisseph as swe

# Calculation flags
GEOCENTRIC = 0  # Default calculation from Earth's perspective
HELIOCENTRIC = 1  # Calculation from Sun's perspective

# Planet IDs from Swiss Ephemeris
SUN = swe.SUN
MOON = swe.MOON
MERCURY = swe.MERCURY
VENUS = swe.VENUS
MARS = swe.MARS
JUPITER = swe.JUPITER
SATURN = swe.SATURN
URANUS = swe.URANUS
NEPTUNE = swe.NEPTUNE
PLUTO = swe.PLUTO
RAHU = swe.MEAN_NODE  # North Node
KETU = -1  # South Node (calculated from Rahu)

# Planet names mapping
PLANET_NAMES = {
    SUN: "Sun",
    MOON: "Moon",
    MERCURY: "Mercury",
    VENUS: "Venus",
    MARS: "Mars",
    JUPITER: "Jupiter",
    SATURN: "Saturn",
    URANUS: "Uranus",
    NEPTUNE: "Neptune",
    PLUTO: "Pluto",
    RAHU: "Rahu",
    KETU: "Ketu"
}

def get_planet_name(planet_id: int) -> str:
    """
    Get the name of a planet from its Swiss Ephemeris ID.
    
    Args:
        planet_id: Swiss Ephemeris planet ID
        
    Returns:
        Name of the planet
    """
    return PLANET_NAMES.get(planet_id, f"Unknown Planet {planet_id}")
