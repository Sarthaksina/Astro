# Cosmic Market Oracle - Planetary Positions Module

"""
This module provides high-precision planetary position calculations using the Swiss Ephemeris
library, with specialized functions for financial market analysis.

It calculates both geocentric and heliocentric positions, retrograde status, and other
astronomical parameters relevant to Vedic astrological analysis of financial markets.

Enhanced with advanced Vedic astrology calculations including:
- Detailed nakshatra calculations with financial significance
- Vimshottari and other dasha systems for market timing
- Shodasavarga (16 divisional charts) for deeper market analysis
- Special lagnas and planetary strengths for financial predictions
- Yogas and combinations relevant to financial markets
"""

from typing import Dict, List, Optional, Tuple, Union
import swisseph as swe
from datetime import datetime, timedelta

from .constants import (
    SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU,
    GEOCENTRIC, HELIOCENTRIC
)


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
    
    def get_julian_day(self, date: Union[datetime, str]) -> float:
        """
        Convert a date to Julian Day.
        
        Args:
            date: A datetime object or ISO format date string (YYYY-MM-DD)
            
        Returns:
            Julian Day as a float
        """
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
            
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
        date: Union[datetime, str, float],
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
            # Ketu is always exactly opposite Rahu
            rahu_pos = self.get_planet_position(RAHU, jd, heliocentric)
            ketu_long = (rahu_pos['longitude'] + 180) % 360
            
            # Create Ketu position data
            position = {
                'longitude': ketu_long,
                'latitude': -rahu_pos['latitude'],  # Opposite latitude
                'distance': rahu_pos['distance'],
                'longitude_speed': rahu_pos['longitude_speed'],
                'latitude_speed': -rahu_pos['latitude_speed'],
                'distance_speed': rahu_pos['distance_speed'],
                'is_retrograde': rahu_pos['is_retrograde'],
                'nakshatra': int(ketu_long / 13.333333) + 1,
                'nakshatra_degree': (ketu_long % 13.333333)
            }
            return position
        
        # Calculate position
        if heliocentric and planet_id == SUN:
            # Sun is at the center in heliocentric system
            position = {
                'longitude': 0.0,
                'latitude': 0.0,
                'distance': 0.0,
                'longitude_speed': 0.0,
                'latitude_speed': 0.0,
                'distance_speed': 0.0,
                'is_retrograde': False,
                'nakshatra': 0,
                'nakshatra_degree': 0.0
            }
        else:
            # Regular calculation
            if heliocentric:
                xx, ret = swe.calc(jd, planet_id, flags=swe.FLG_SWIEPH | swe.FLG_SPEED | swe.FLG_HELCTR)
            else:
                xx, ret = swe.calc(jd, planet_id, flags=swe.FLG_SWIEPH | swe.FLG_SPEED)
                
            # Convert to sidereal longitude (Vedic)
            longitude = xx[0]
            longitude = (longitude - swe.get_ayanamsa(jd)) % 360
            
            # Calculate nakshatra position
            nakshatra = int(longitude / 13.333333) + 1
            nakshatra_degree = longitude % 13.333333
            
            position = {
                'longitude': longitude,
                'latitude': xx[1],
                'distance': xx[2],
                'longitude_speed': xx[3],
                'latitude_speed': xx[4],
                'distance_speed': xx[5],
                'is_retrograde': xx[3] < 0,  # Retrograde if longitude speed is negative
                'nakshatra': nakshatra,
                'nakshatra_degree': nakshatra_degree
            }
            
        return position
    
    def calculate_positions(
        self, 
        date: Union[datetime, str, float],
        planets: Optional[List[int]] = None,
        heliocentric: bool = False
    ) -> Dict:
        """
        Calculate positions for multiple planets at once.
        
        Args:
            date: Date as datetime, ISO string, or Julian Day
            planets: List of planet IDs to calculate. If None, calculate all planets.
            heliocentric: If True, calculate heliocentric positions
            
        Returns:
            Dictionary with planet IDs as keys and position dictionaries as values.
            Also includes the date as a datetime object.
        """
        if planets is None:
            planets = [SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU]
            
        # Convert date to Julian Day if needed
        if not isinstance(date, float):
            jd = self.get_julian_day(date)
            if isinstance(date, str):
                date_obj = datetime.fromisoformat(date)
            else:
                date_obj = date
        else:
            jd = date
            date_obj = swe.revjul(jd)
            date_obj = datetime(date_obj[0], date_obj[1], date_obj[2])
            
        # Calculate positions for all planets
        positions = {'date': date_obj}
        for planet_id in planets:
            positions[planet_id] = self.get_planet_position(planet_id, jd, heliocentric)
            
        return positions
    
    def get_ayanamsa(self, date: Union[datetime, str, float]) -> float:
        """
        Get the ayanamsa (precession of the equinoxes) at a given date.
        
        Args:
            date: Date as datetime, ISO string, or Julian Day
            
        Returns:
            Ayanamsa in degrees
        """
        # Convert date to Julian Day if needed
        if not isinstance(date, float):
            jd = self.get_julian_day(date)
        else:
            jd = date
            
        return swe.get_ayanamsa(jd)
    
    def get_planetary_aspects(self, positions: Dict[int, Dict[str, Union[float, bool]]], orb: float = 6.0) -> List[Dict]:
        """
        Calculate aspects between planets.
        
        Args:
            positions: Dictionary of planetary positions as returned by calculate_positions
            orb: Maximum orb (deviation from exact aspect) in degrees
            
        Returns:
            List of aspect dictionaries, each containing:
                - planet1: First planet ID
                - planet2: Second planet ID
                - aspect_type: Type of aspect (conjunction, opposition, trine, etc.)
                - angle: Exact angle of the aspect
                - orb: Actual orb (deviation from exact aspect)
                - applying: True if the aspect is applying (getting closer to exact)
        """
        aspects = []
        
        # Define aspect types and their angles
        aspect_types = {
            'conjunction': 0,
            'opposition': 180,
            'trine': 120,
            'square': 90,
            'sextile': 60
        }
        
        # Calculate aspects between all planet pairs
        planets = [p for p in positions.keys() if isinstance(p, int)]
        for i, planet1 in enumerate(planets):
            for planet2 in planets[i+1:]:  # Only check each pair once
                # Get longitudes
                long1 = positions[planet1]['longitude']
                long2 = positions[planet2]['longitude']
                
                # Calculate angular distance
                angle = abs(long1 - long2)
                if angle > 180:
                    angle = 360 - angle
                
                # Check for aspects
                for aspect_name, aspect_angle in aspect_types.items():
                    # Calculate orb
                    aspect_orb = abs(angle - aspect_angle)
                    if aspect_orb > 180:
                        aspect_orb = 360 - aspect_orb
                        
                    # If within allowed orb, add to aspects list
                    if aspect_orb <= orb:
                        # Determine if aspect is applying or separating
                        speed1 = positions[planet1]['longitude_speed']
                        speed2 = positions[planet2]['longitude_speed']
                        relative_speed = speed1 - speed2
                        
                        # For conjunction and opposition, applying if getting closer
                        # For other aspects, depends on the relative positions
                        is_applying = self.is_aspect_applying(planet1, planet2, positions)
                        
                        aspects.append({
                            'planet1': planet1,
                            'planet2': planet2,
                            'aspect_type': aspect_name,
                            'aspect_angle': aspect_angle,
                            'actual_angle': angle,
                            'orb': aspect_orb,
                            'applying': is_applying
                        })
        
        return aspects
    
    def is_aspect_applying(self, planet1: int, planet2: int, positions: Dict[int, Dict[str, Union[float, bool]]]) -> bool:
        """
        Determine if an aspect between two planets is applying (getting closer to exact).
        
        Args:
            planet1: First planet ID
            planet2: Second planet ID
            positions: Dictionary of planetary positions
            
        Returns:
            True if the aspect is applying, False if separating
        """
        # Get longitudes and speeds
        long1 = positions[planet1]['longitude']
        long2 = positions[planet2]['longitude']
        speed1 = positions[planet1]['longitude_speed']
        speed2 = positions[planet2]['longitude_speed']
        
        # Calculate shortest angular distance (0-180 degrees)
        angle = (long1 - long2) % 360
        if angle > 180:
            angle = 360 - angle
            
        # Calculate relative speed
        relative_speed = speed1 - speed2
        
        # If planets are moving toward each other, the aspect is applying
        # This depends on their relative positions
        if angle <= 180:
            # If planet1 is ahead of planet2
            if long1 > long2:
                return relative_speed < 0  # Applying if planet1 is moving slower or retrograde
            else:
                return relative_speed > 0  # Applying if planet1 is moving faster
        else:
            # If planet1 is behind planet2
            if long1 < long2:
                return relative_speed < 0  # Applying if planet1 is moving slower or retrograde
            else:
                return relative_speed > 0  # Applying if planet1 is moving faster
    
    def calculate_vimshottari_dasha(self, birth_moon_longitude: float, birth_date: Union[datetime, str]) -> Dict:
        """
        Calculate Vimshottari dasha periods based on Moon's position at birth.
        
        Args:
            birth_moon_longitude: Moon's longitude at birth (0-360 degrees)
            birth_date: Birth date as datetime or ISO string
            
        Returns:
            Dictionary containing dasha periods with start and end dates
        """
        # Convert birth date to datetime if needed
        if isinstance(birth_date, str):
            birth_date = datetime.fromisoformat(birth_date)
            
        # Calculate birth nakshatra and remainder
        nakshatra = int(birth_moon_longitude / 13.333333)
        remainder = birth_moon_longitude % 13.333333
        
        # Vimshottari dasha order and years
        dasha_lords = [KETU, VENUS, SUN, MOON, MARS, RAHU, JUPITER, SATURN, MERCURY]
        dasha_years = [7, 20, 6, 10, 7, 18, 16, 19, 17]
        total_years = sum(dasha_years)  # 120 years
        
        # Calculate portion of first dasha already elapsed
        first_lord_index = nakshatra % 9
        first_lord = dasha_lords[first_lord_index]
        first_lord_years = dasha_years[first_lord_index]
        
        # Calculate elapsed portion
        elapsed_portion = remainder / 13.333333
        elapsed_years = first_lord_years * elapsed_portion
        
        # Calculate remaining portion of first dasha
        remaining_years = first_lord_years - elapsed_years
        
        # Calculate dasha periods
        dashas = []
        
        # First dasha (partial)
        current_date = birth_date
        end_date = current_date + timedelta(days=remaining_years * 365.25)
        
        dashas.append({
            'planet': first_lord,
            'start_date': current_date,
            'end_date': end_date,
            'years': remaining_years
        })
        
        # Subsequent dashas
        current_date = end_date
        for i in range(1, 9):
            lord_index = (first_lord_index + i) % 9
            lord = dasha_lords[lord_index]
            years = dasha_years[lord_index]
            
            end_date = current_date + timedelta(days=years * 365.25)
            
            dashas.append({
                'planet': lord,
                'start_date': current_date,
                'end_date': end_date,
                'years': years
            })
            
            current_date = end_date
        
        return {
            'birth_date': birth_date,
            'birth_moon_longitude': birth_moon_longitude,
            'birth_nakshatra': nakshatra + 1,  # 1-based nakshatra
            'dashas': dashas
        }
