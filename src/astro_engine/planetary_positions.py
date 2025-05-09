# Cosmic Market Oracle - Planetary Positions Module

"""
This module provides high-precision planetary position calculations using the Swiss Ephemeris
library, with specialized functions for financial market analysis.

It calculates both geocentric and heliocentric positions, retrograde status, and other
astronomical parameters relevant to Vedic astrological analysis of financial markets.
"""

import datetime
from typing import Dict, List, Optional, Tuple, Union

import swisseph as swe

# Define constants for planets and other celestial bodies
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

# Calculation flags
GEOCENTRIC = swe.FLG_SWIEPH | swe.FLG_SPEED
HELIOCENTRIC = swe.FLG_SWIEPH | swe.FLG_SPEED | swe.FLG_HELCTR


class PlanetaryCalculator:
    """Handles high-precision planetary calculations for financial astrology."""

    def __init__(self, ephemeris_path: Optional[str] = None):
        """
        Initialize the planetary calculator with the Swiss Ephemeris.
        
        Args:
            ephemeris_path: Optional path to the Swiss Ephemeris files.
                            If None, the default path will be used.
        """
        if ephemeris_path:
            swe.set_ephe_path(ephemeris_path)
        
        # Initialize Swiss Ephemeris
        swe.set_sid_mode(swe.SIDM_LAHIRI)  # Set Ayanamsa to Lahiri (most common in Vedic)
    
    def get_julian_day(self, date: Union[datetime.datetime, str]) -> float:
        """
        Convert a date to Julian Day.
        
        Args:
            date: A datetime object or ISO format date string (YYYY-MM-DD)
            
        Returns:
            Julian Day as a float
        """
        if isinstance(date, str):
            date = datetime.datetime.fromisoformat(date)
            
        jd = swe.julday(
            date.year,
            date.month,
            date.day,
            date.hour + date.minute/60.0 + date.second/3600.0
        )
        return jd
    
    def get_planet_position(
        self, 
        planet_id: int, 
        date: Union[datetime.datetime, str, float],
        heliocentric: bool = False
    ) -> Dict[str, Union[float, bool]]:
        """
        Calculate the position of a planet at a given date.
        
        Args:
            planet_id: Swiss Ephemeris planet ID
            date: Date as datetime, ISO string, or Julian Day
            heliocentric: If True, calculate heliocentric position
            
        Returns:
            Dictionary containing position information:
                - longitude: Zodiacal longitude (0-360 degrees)
                - latitude: Celestial latitude
                - distance: Distance from Earth/Sun in AU
                - longitude_speed: Speed in longitude (degrees/day)
                - latitude_speed: Speed in latitude (degrees/day)
                - distance_speed: Speed in distance (AU/day)
                - is_retrograde: True if planet is retrograde
                - nakshatra: Nakshatra (lunar mansion) position (1-27)
                - nakshatra_degree: Degree within nakshatra (0-13.333...)
        """
        # Convert date to Julian Day if needed
        if not isinstance(date, float):
            jd = self.get_julian_day(date)
        else:
            jd = date
            
        # Set calculation flags
        flags = HELIOCENTRIC if heliocentric else GEOCENTRIC
        
        # Special case for Ketu (South Node)
        if planet_id == KETU:
            # Calculate Rahu (North Node) and adjust for Ketu
            xx, retflags = swe.calc_ut(jd, RAHU, flags)
            # Ketu is exactly opposite Rahu
            longitude = (xx[0] + 180) % 360
            latitude = -xx[1]  # Opposite latitude
            distance = xx[2]
            longitude_speed = xx[3]
            latitude_speed = -xx[4]
            distance_speed = xx[5]
        else:
            # Calculate planet position
            xx, retflags = swe.calc_ut(jd, planet_id, flags)
            longitude = xx[0]
            latitude = xx[1]
            distance = xx[2]
            longitude_speed = xx[3]
            latitude_speed = xx[4]
            distance_speed = xx[5]
        
        # Calculate nakshatra (lunar mansion)
        # Each nakshatra is 13.333... degrees (360/27)
        nakshatra_size = 360 / 27
        nakshatra = int(longitude / nakshatra_size) + 1  # 1-27 numbering
        nakshatra_degree = longitude % nakshatra_size
        
        return {
            "longitude": longitude,
            "latitude": latitude,
            "distance": distance,
            "longitude_speed": longitude_speed,
            "latitude_speed": latitude_speed,
            "distance_speed": distance_speed,
            "is_retrograde": longitude_speed < 0,
            "nakshatra": nakshatra,
            "nakshatra_degree": nakshatra_degree
        }
    
    def get_all_planets(
        self, 
        date: Union[datetime.datetime, str, float],
        include_nodes: bool = True,
        heliocentric: bool = False
    ) -> Dict[int, Dict[str, Union[float, bool]]]:
        """
        Calculate positions for all major planets at a given date.
        
        Args:
            date: Date as datetime, ISO string, or Julian Day
            include_nodes: Whether to include Rahu and Ketu (lunar nodes)
            heliocentric: If True, calculate heliocentric positions
            
        Returns:
            Dictionary mapping planet IDs to their position information
        """
        planets = [SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO]
        
        if include_nodes:
            planets.extend([RAHU, KETU])
            
        # Convert date to Julian Day if needed
        if not isinstance(date, float):
            jd = self.get_julian_day(date)
        else:
            jd = date
            
        results = {}
        for planet_id in planets:
            results[planet_id] = self.get_planet_position(planet_id, jd, heliocentric)
            
        return results
    
    def get_ayanamsa(self, date: Union[datetime.datetime, str, float]) -> float:
        """
        Calculate the ayanamsa (precession of the equinoxes) at a given date.
        
        Args:
            date: Date as datetime, ISO string, or Julian Day
            
        Returns:
            Ayanamsa value in degrees
        """
        # Convert date to Julian Day if needed
        if not isinstance(date, float):
            jd = self.get_julian_day(date)
        else:
            jd = date
            
        return swe.get_ayanamsa_ut(jd)
    
    def close(self):
        """
        Close the Swiss Ephemeris to free resources.
        """
        swe.close()
        
    def __del__(self):
        """
        Ensure Swiss Ephemeris is closed when the object is deleted.
        """
        try:
            self.close()
        except:
            pass


def get_planetary_aspects(
    positions: Dict[int, Dict[str, Union[float, bool]]],
    orb: float = 6.0
) -> List[Dict[str, Union[int, float, str]]]:
    """
    Calculate aspects between planets based on their positions.
    
    Args:
        positions: Dictionary of planetary positions as returned by get_all_planets()
        orb: Maximum orb (deviation from exact aspect) in degrees
        
    Returns:
        List of aspects with planet pairs, aspect type, and orb
    """
    # Define major aspects and their ideal angles
    aspects = {
        "Conjunction": 0,
        "Opposition": 180,
        "Trine": 120,
        "Square": 90,
        "Sextile": 60,
    }
    
    results = []
    
    # Get all planet pairs
    planet_ids = list(positions.keys())
    for i, planet1 in enumerate(planet_ids):
        for planet2 in planet_ids[i+1:]:
            # Skip Sun-Moon nodes aspects as they're less relevant
            if (planet1 == SUN and planet2 in [RAHU, KETU]) or \
               (planet2 == SUN and planet1 in [RAHU, KETU]):
                continue
                
            # Calculate the angular difference
            lon1 = positions[planet1]["longitude"]
            lon2 = positions[planet2]["longitude"]
            
            # Find the smallest angle between the planets
            diff = abs(lon1 - lon2) % 360
            if diff > 180:
                diff = 360 - diff
                
            # Check each aspect type
            for aspect_name, aspect_angle in aspects.items():
                # Calculate how close we are to the exact aspect
                aspect_diff = abs(diff - aspect_angle)
                
                # If within orb, record the aspect
                if aspect_diff <= orb:
                    results.append({
                        "planet1": planet1,
                        "planet2": planet2,
                        "aspect_type": aspect_name,
                        "aspect_angle": aspect_angle,
                        "actual_angle": diff,
                        "orb": aspect_diff,
                        "applying": is_aspect_applying(positions, planet1, planet2, aspect_angle)
                    })
    
    return results


def is_aspect_applying(
    positions: Dict[int, Dict[str, Union[float, bool]]],
    planet1: int,
    planet2: int,
    aspect_angle: float
) -> bool:
    """
    Determine if an aspect is applying (getting closer to exact) or separating.
    
    Args:
        positions: Dictionary of planetary positions
        planet1: ID of first planet
        planet2: ID of second planet
        aspect_angle: The ideal angle of the aspect
        
    Returns:
        True if the aspect is applying, False if separating
    """
    # Get speeds
    speed1 = positions[planet1]["longitude_speed"]
    speed2 = positions[planet2]["longitude_speed"]
    
    # Get current positions
    lon1 = positions[planet1]["longitude"]
    lon2 = positions[planet2]["longitude"]
    
    # Calculate relative speed and position
    rel_speed = speed1 - speed2
    rel_pos = (lon1 - lon2) % 360
    
    # Adjust for aspect angle
    if aspect_angle == 0:  # Conjunction
        # Applying if getting closer to 0° difference
        if rel_pos <= 180:
            return rel_speed < 0  # Planet1 catching up to Planet2
        else:
            return rel_speed > 0  # Planet2 catching up to Planet1
            
    elif aspect_angle == 180:  # Opposition
        # Applying if getting closer to 180° difference
        if rel_pos <= 180:
            return rel_speed > 0  # Moving apart toward 180°
        else:
            return rel_speed < 0  # Moving together toward 180°
    
    # For other aspects, more complex calculation needed
    # This is a simplified approach
    current_diff = abs(rel_pos - aspect_angle)
    
    # Simulate future position
    future_rel_pos = (lon1 + speed1 - (lon2 + speed2)) % 360
    future_diff = abs(future_rel_pos - aspect_angle)
    
    # If future difference is less than current, aspect is applying
    return future_diff < current_diff