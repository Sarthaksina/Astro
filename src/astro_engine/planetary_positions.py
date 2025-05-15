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
    
    def get_nakshatra_details(self, longitude: float) -> Dict[str, Union[int, float, str]]:
        """
        Get detailed information about a nakshatra based on longitude.
        
        Args:
            longitude: Zodiacal longitude in degrees (0-360)
            
        Returns:
            Dictionary with nakshatra details including:
                - nakshatra_num: Nakshatra number (1-27)
                - nakshatra_name: Name of the nakshatra
                - nakshatra_lord: Planetary lord of the nakshatra
                - pada: Pada (quarter) of the nakshatra (1-4)
                - financial_nature: Financial significance (bullish/bearish/neutral)
                - degree: Degree within nakshatra (0-13.333...)
        """
        # Nakshatra names in order
        nakshatra_names = [
            "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
            "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
            "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
            "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
            "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
        ]
        
        # Nakshatra lords in Vimshottari dasha order
        nakshatra_lords = [
            "Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter",
            "Saturn", "Mercury", "Ketu", "Venus", "Sun", "Moon", "Mars",
            "Rahu", "Jupiter", "Saturn", "Mercury", "Ketu", "Venus", "Sun",
            "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"
        ]
        
        # Financial nature of nakshatras (simplified for financial markets)
        financial_nature = [
            "bullish", "bearish", "neutral", "bullish", "neutral", "bearish",
            "bullish", "bullish", "bearish", "neutral", "bullish", "bullish",
            "neutral", "bullish", "volatile", "volatile", "bullish", "bearish",
            "bearish", "neutral", "bullish", "bullish", "volatile", "bearish",
            "neutral", "bullish", "neutral"
        ]
        
        # Calculate nakshatra
        nakshatra_size = 360 / 27
        nakshatra_num = int(longitude / nakshatra_size) + 1  # 1-27 numbering
        nakshatra_degree = longitude % nakshatra_size
        
        # Calculate pada (quarter)
        pada = int(nakshatra_degree / (nakshatra_size / 4)) + 1  # 1-4 numbering
        
        return {
            "nakshatra_num": nakshatra_num,
            "nakshatra_name": nakshatra_names[nakshatra_num - 1],
            "nakshatra_lord": nakshatra_lords[nakshatra_num - 1],
            "pada": pada,
            "financial_nature": financial_nature[nakshatra_num - 1],
            "degree": nakshatra_degree
        }
    
    def calculate_vimshottari_dasha(self, birth_moon_longitude: float, birth_date: Union[datetime.datetime, str]) -> Dict:
        """
        Calculate Vimshottari dasha periods from birth date and Moon position.
        
        Args:
            birth_moon_longitude: Longitude of Moon at birth
            birth_date: Birth date as datetime or ISO string
            
        Returns:
            Dictionary with dasha periods and their dates
        """
        # Vimshottari dasha periods for each planet (in years)
        dasha_years = {
            "Ketu": 7,
            "Venus": 20,
            "Sun": 6,
            "Moon": 10,
            "Mars": 7,
            "Rahu": 18,
            "Jupiter": 16,
            "Saturn": 19,
            "Mercury": 17
        }
        
        # Get nakshatra details for Moon
        nakshatra_details = self.get_nakshatra_details(birth_moon_longitude)
        nakshatra_lord = nakshatra_details["nakshatra_lord"]
        
        # Calculate consumed portion of nakshatra
        nakshatra_size = 360 / 27
        consumed_portion = nakshatra_details["degree"] / nakshatra_size
        
        # Convert birth date to datetime if needed
        if isinstance(birth_date, str):
            birth_date = datetime.datetime.fromisoformat(birth_date)
        
        # Calculate dasha periods
        dasha_periods = {}
        current_date = birth_date
        
        # Find the starting lord in the Vimshottari sequence
        dasha_lords = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
        start_index = dasha_lords.index(nakshatra_lord)
        
        # Calculate remaining duration of current dasha
        remaining_years = dasha_years[nakshatra_lord] * (1 - consumed_portion)
        
        # Add current dasha
        end_date = current_date + datetime.timedelta(days=remaining_years * 365.25)
        dasha_periods[nakshatra_lord] = {
            "start_date": current_date,
            "end_date": end_date,
            "duration_years": remaining_years
        }
        
        current_date = end_date
        
        # Calculate subsequent dashas
        for i in range(1, 9):
            lord_index = (start_index + i) % 9
            lord = dasha_lords[lord_index]
            years = dasha_years[lord]
            
            end_date = current_date + datetime.timedelta(days=years * 365.25)
            dasha_periods[lord] = {
                "start_date": current_date,
                "end_date": end_date,
                "duration_years": years
            }
            
            current_date = end_date
        
        return dasha_periods
    
    def calculate_divisional_chart(self, longitude: float, division: int) -> float:
        """
        Calculate position in a divisional chart (varga).
        
        Args:
            longitude: Zodiacal longitude in degrees (0-360)
            division: Division number (1, 2, 3, 4, 7, 9, 10, 12, 16, 20, 24, 27, 30, 40, 45, 60)
            
        Returns:
            Longitude in the divisional chart (0-360)
        """
        # D-1 is the same as the birth chart
        if division == 1:
            return longitude
        
        # Get sign number (0-11) and position within sign
        sign = int(longitude / 30)
        pos_in_sign = longitude % 30
        
        # Calculate divisional chart position based on division
        if division == 2:  # Hora
            if sign % 2 == 0:  # Even sign
                if pos_in_sign < 15:
                    return 0  # Sun
                else:
                    return 30  # Moon
            else:  # Odd sign
                if pos_in_sign < 15:
                    return 30  # Moon
                else:
                    return 0  # Sun
        elif division == 9:  # Navamsa (D-9) - most important divisional chart
            # Each sign is divided into 9 parts of 3°20' each
            navamsa_size = 30 / 9
            navamsa_index = int(pos_in_sign / navamsa_size)
            
            # Calculate navamsa sign
            if sign % 3 == 0:  # Fiery signs (Aries, Leo, Sagittarius)
                navamsa_sign = (navamsa_index) % 12
            elif sign % 3 == 1:  # Earthy signs (Taurus, Virgo, Capricorn)
                navamsa_sign = (navamsa_index + 4) % 12
            else:  # Airy and Watery signs
                navamsa_sign = (navamsa_index + 8) % 12
            
            # Calculate position within navamsa sign
            navamsa_pos = (pos_in_sign % navamsa_size) * (30 / navamsa_size)
            
            return navamsa_sign * 30 + navamsa_pos
        elif division == 10:  # Dasamsa (D-10) - career and financial success
            # Each sign is divided into 10 parts of 3° each
            dasamsa_size = 30 / 10
            dasamsa_index = int(pos_in_sign / dasamsa_size)
            
            # Calculate dasamsa sign
            if sign < 6:  # First half of zodiac
                dasamsa_sign = (sign + dasamsa_index) % 12
            else:  # Second half of zodiac
                dasamsa_sign = (sign + dasamsa_index + 6) % 12
            
            # Calculate position within dasamsa sign
            dasamsa_pos = (pos_in_sign % dasamsa_size) * (30 / dasamsa_size)
            
            return dasamsa_sign * 30 + dasamsa_pos
        else:
            # Generic calculation for other divisions
            varga_size = 30 / division
            varga_index = int(pos_in_sign / varga_size)
            varga_sign = (sign + varga_index) % 12
            varga_pos = (pos_in_sign % varga_size) * (30 / varga_size)
            
            return varga_sign * 30 + varga_pos
    
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
) -> List[Dict[str, Union[int, str, float, bool]]]:
    """
    Calculate aspects between planets based on their positions.
    
    Args:
        positions: Dictionary of planetary positions as returned by get_all_planets()
{{ ... }
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
{{ ... }
    
    # For other aspects, more complex calculation needed
    # This is a simplified approach
    current_diff = abs(rel_pos - aspect_angle)
    
    # Simulate future position
    future_rel_pos = (lon1 + speed1 - (lon2 + speed2)) % 360
    future_diff = abs(future_rel_pos - aspect_angle)
    
    # If future difference is less than current, aspect is applying
    return future_diff < current_diff


def analyze_market_trend(
    planetary_positions: Dict[int, Dict[str, Union[float, bool]]],
    date: Union[datetime.datetime, str, float],
    calculator: PlanetaryCalculator) -> Dict[str, Union[str, float, List[str]]]:
    """
    Analyze planetary positions to predict market trends.
    
    Args:
        planetary_positions: Dictionary of planetary positions
        date: The date for analysis
        calculator: PlanetaryCalculator instance for additional calculations
        
    Returns:
        Dictionary with market trend analysis including:
            - primary_trend: Main market direction (bullish/bearish/neutral/volatile)
            - strength: Strength of the trend (0-100)
            - key_factors: List of key astrological factors affecting the market
            - reversal_probability: Probability of trend reversal (0-100)
            - support_level: Astrological support level
            - resistance_level: Astrological resistance level
    """
    # Initialize result
    result = {
        "primary_trend": "neutral",
        "strength": 50.0,
        "key_factors": [],
        "reversal_probability": 0.0,
        "support_level": 0.0,
        "resistance_level": 0.0
    }
    
    # 1. Analyze benefic vs. malefic balance
    benefics = [JUPITER, VENUS, MERCURY, MOON]
    malefics = [SATURN, MARS, RAHU, KETU, SUN]  # Sun is mild malefic
    
    benefic_strength = 0
    malefic_strength = 0
    
    # Calculate benefic and malefic strengths
    for planet in benefics:
        if planet in planetary_positions:
            # Benefics are stronger when direct (not retrograde)
            if not planetary_positions[planet].get("is_retrograde", False):
                benefic_strength += 1.0
            else:
                benefic_strength += 0.5
                
            # Check if in own sign or exalted
            longitude = planetary_positions[planet]["longitude"]
            sign = int(longitude / 30)
            
            # Jupiter in Sagittarius, Pisces or exalted in Cancer
            if planet == JUPITER and sign in [8, 11, 3]:
                benefic_strength += 1.0
                result["key_factors"].append("Jupiter well-placed")
                
            # Venus in Taurus, Libra or exalted in Pisces
            elif planet == VENUS and sign in [1, 6, 11]:
                benefic_strength += 1.0
                result["key_factors"].append("Venus well-placed")
                
            # Mercury in Gemini, Virgo or exalted in Virgo
            elif planet == MERCURY and sign in [2, 5]:
                benefic_strength += 1.0
                result["key_factors"].append("Mercury well-placed")
                
            # Moon in Cancer or exalted in Taurus
            elif planet == MOON and sign in [3, 1]:
                benefic_strength += 1.0
                result["key_factors"].append("Moon well-placed")
    
    for planet in malefics:
        if planet in planetary_positions:
            # Malefics are stronger when retrograde
            if planetary_positions[planet].get("is_retrograde", False):
                malefic_strength += 1.0
            else:
                malefic_strength += 0.7
                
            # Check if in own sign or exalted
            longitude = planetary_positions[planet]["longitude"]
            sign = int(longitude / 30)
            
            # Saturn in Capricorn, Aquarius or exalted in Libra
            if planet == SATURN and sign in [9, 10, 6]:
                malefic_strength += 1.0
                result["key_factors"].append("Saturn strong")
                
            # Mars in Aries, Scorpio or exalted in Capricorn
            elif planet == MARS and sign in [0, 7, 9]:
                malefic_strength += 1.0
                result["key_factors"].append("Mars strong")
                
            # Sun in Leo or exalted in Aries
            elif planet == SUN and sign in [4, 0]:
                malefic_strength += 0.8
                result["key_factors"].append("Sun strong")
    
    # 2. Analyze Moon nakshatra for short-term market sentiment
    if MOON in planetary_positions:
        moon_longitude = planetary_positions[MOON]["longitude"]
        moon_nakshatra = calculator.get_nakshatra_details(moon_longitude)
        
        result["key_factors"].append(f"Moon in {moon_nakshatra['nakshatra_name']} nakshatra ({moon_nakshatra['financial_nature']})") 
        
        # Adjust trend based on nakshatra financial nature
        if moon_nakshatra['financial_nature'] == "bullish":
            benefic_strength += 0.8
        elif moon_nakshatra['financial_nature'] == "bearish":
            malefic_strength += 0.8
        elif moon_nakshatra['financial_nature'] == "volatile":
            result["reversal_probability"] += 15.0
    
    # 3. Check for important aspects
    aspects = get_planetary_aspects(planetary_positions, orb=7.0)
    
    for aspect in aspects:
        if aspect["aspect_type"] in ["conjunction", "trine", "sextile"]:
            # Benefic aspects
            if aspect["planet1"] in benefics and aspect["planet2"] in benefics:
                benefic_strength += 0.7
                result["key_factors"].append(f"Benefic {aspect['aspect_type']} between {get_planet_name(aspect['planet1'])} and {get_planet_name(aspect['planet2'])}")
            
            # Mixed aspects can still be positive
            elif (aspect["planet1"] in benefics and aspect["planet2"] in malefics) or \
                 (aspect["planet1"] in malefics and aspect["planet2"] in benefics):
                benefic_strength += 0.3
        
        elif aspect["aspect_type"] in ["opposition", "square"]:
            # Malefic aspects
            if aspect["planet1"] in malefics and aspect["planet2"] in malefics:
                malefic_strength += 0.7
                result["key_factors"].append(f"Malefic {aspect['aspect_type']} between {get_planet_name(aspect['planet1'])} and {get_planet_name(aspect['planet2'])}")
                result["reversal_probability"] += 10.0
            
            # Mixed aspects can be challenging
            elif (aspect["planet1"] in benefics and aspect["planet2"] in malefics) or \
                 (aspect["planet1"] in malefics and aspect["planet2"] in benefics):
                malefic_strength += 0.3
                result["reversal_probability"] += 5.0
    
    # 4. Check for financial yogas
    yogas = analyze_financial_yogas(planetary_positions, calculator)
    
    for yoga in yogas:
        result["key_factors"].append(f"{yoga['name']} ({yoga['strength']}% strength)")
        
        if yoga["market_impact"] == "bullish":
            benefic_strength += yoga["strength"] / 100.0
        elif yoga["market_impact"] == "bearish":
            malefic_strength += yoga["strength"] / 100.0
        elif yoga["market_impact"] == "volatile":
            result["reversal_probability"] += yoga["strength"] / 5.0
    
    # 5. Determine primary trend
    total_strength = benefic_strength + malefic_strength
    if total_strength > 0:
        benefic_percentage = (benefic_strength / total_strength) * 100
        malefic_percentage = (malefic_strength / total_strength) * 100
        
        if benefic_percentage >= 60:
            result["primary_trend"] = "bullish"
            result["strength"] = benefic_percentage
        elif malefic_percentage >= 60:
            result["primary_trend"] = "bearish"
            result["strength"] = malefic_percentage
        elif result["reversal_probability"] >= 30:
            result["primary_trend"] = "volatile"
            result["strength"] = min(benefic_percentage, malefic_percentage) + result["reversal_probability"] / 2
        else:
            result["primary_trend"] = "neutral"
            result["strength"] = min(benefic_percentage, malefic_percentage)
    
    # 6. Calculate support and resistance levels (simplified)
    # This is a placeholder - in a real implementation, this would be more sophisticated
    if SUN in planetary_positions and MOON in planetary_positions:
        sun_longitude = planetary_positions[SUN]["longitude"]
        moon_longitude = planetary_positions[MOON]["longitude"]
        
        # Support level based on harmonious angle from Moon to Sun
        support_angle = (sun_longitude + 120) % 360  # Trine from Sun
        result["support_level"] = support_angle
        
        # Resistance level based on challenging angle from Moon to Sun
        resistance_angle = (sun_longitude + 90) % 360  # Square from Sun
        result["resistance_level"] = resistance_angle
    
    # Limit to top 5 key factors to avoid overwhelming information
    result["key_factors"] = result["key_factors"][:5]
    
    return result
) -> Dict[str, Union[str, float, List[str]]]:
    """
    Analyze market trend based on planetary positions using Vedic astrology principles.
    
    Args:
        planetary_positions: Dictionary of planetary positions
        date: Date for the analysis
        calculator: PlanetaryCalculator instance for additional calculations
        
    Returns:
        Dictionary with market trend analysis including:
            - primary_trend: Overall market trend (bullish/bearish/neutral/volatile)
            - strength: Strength of the trend (0-100)
            - key_factors: List of key astrological factors influencing the market
            - reversal_probability: Probability of trend reversal (0-100)
    """
    # Initialize variables
    bullish_factors = 0
    bearish_factors = 0
    volatile_factors = 0
    key_factors = []
    
    # Convert date to datetime if needed
    if isinstance(date, float):
        # Convert Julian day to datetime
        y, m, d, h, mi, s = swe.revjul(date)
        date = datetime.datetime(y, m, d, int(h), int(mi), int(s))
    elif isinstance(date, str):
        date = datetime.datetime.fromisoformat(date)
    
    # 1. Analyze Moon's nakshatra - important for short-term market movements
    if MOON in planetary_positions:
        moon_longitude = planetary_positions[MOON]["longitude"]
        moon_nakshatra = calculator.get_nakshatra_details(moon_longitude)
        
        if moon_nakshatra["financial_nature"] == "bullish":
            bullish_factors += 2
            key_factors.append(f"Moon in bullish {moon_nakshatra['nakshatra_name']} nakshatra")
        elif moon_nakshatra["financial_nature"] == "bearish":
            bearish_factors += 2
            key_factors.append(f"Moon in bearish {moon_nakshatra['nakshatra_name']} nakshatra")
        elif moon_nakshatra["financial_nature"] == "volatile":
            volatile_factors += 2
            key_factors.append(f"Moon in volatile {moon_nakshatra['nakshatra_name']} nakshatra")
    
    # 2. Check retrograde planets - important for market reversals
    retrograde_planets = []
    financial_planets = [MERCURY, VENUS, JUPITER, SATURN]
    
    for planet in financial_planets:
        if planet in planetary_positions and planetary_positions[planet]["is_retrograde"]:
            retrograde_planets.append(planet)
    
    if MERCURY in retrograde_planets:
        volatile_factors += 2
        key_factors.append("Mercury retrograde - communication issues and market volatility")
    
    if JUPITER in retrograde_planets:
        bearish_factors += 1
        key_factors.append("Jupiter retrograde - reduced market optimism")
    
    if SATURN in retrograde_planets:
        bullish_factors += 1
        key_factors.append("Saturn retrograde - temporary easing of restrictions")
    
    # 3. Analyze aspects between financial planets
    aspects = get_planetary_aspects(planetary_positions, orb=6.0)
    
    for aspect in aspects:
        # Jupiter-Saturn aspects - economic cycles
        if (aspect["planet1"] == JUPITER and aspect["planet2"] == SATURN) or \
           (aspect["planet1"] == SATURN and aspect["planet2"] == JUPITER):
            if aspect["aspect_type"] == "Conjunction":
                volatile_factors += 3
                key_factors.append("Jupiter-Saturn conjunction - major economic cycle shift")
            elif aspect["aspect_type"] == "Trine":
                bullish_factors += 2
                key_factors.append("Jupiter-Saturn trine - economic stability and growth")
            elif aspect["aspect_type"] == "Square":
                bearish_factors += 2
                key_factors.append("Jupiter-Saturn square - economic tension and uncertainty")
        
        # Sun-Jupiter aspects - market optimism
        if (aspect["planet1"] == SUN and aspect["planet2"] == JUPITER) or \
           (aspect["planet1"] == JUPITER and aspect["planet2"] == SUN):
            if aspect["aspect_type"] in ["Conjunction", "Trine"]:
                bullish_factors += 2
                key_factors.append(f"Sun-Jupiter {aspect['aspect_type'].lower()} - market optimism")
        
        # Saturn-Mars aspects - market fear
        if (aspect["planet1"] == SATURN and aspect["planet2"] == MARS) or \
           (aspect["planet1"] == MARS and aspect["planet2"] == SATURN):
            if aspect["aspect_type"] in ["Conjunction", "Square", "Opposition"]:
                bearish_factors += 2
                key_factors.append(f"Saturn-Mars {aspect['aspect_type'].lower()} - market fear and volatility")
    
    # 4. Check for Rahu-Ketu axis (Nodes) alignment with financial planets
    if RAHU in planetary_positions and KETU in planetary_positions:
        rahu_longitude = planetary_positions[RAHU]["longitude"]
        ketu_longitude = planetary_positions[KETU]["longitude"]
        
        for planet_id, position in planetary_positions.items():
            if planet_id in [SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN]:
                planet_longitude = position["longitude"]
                
                # Check conjunction with Rahu (within 10 degrees)
                rahu_diff = abs(rahu_longitude - planet_longitude)
                if rahu_diff > 180:
                    rahu_diff = 360 - rahu_diff
                
                # Check conjunction with Ketu (within 10 degrees)
                ketu_diff = abs(ketu_longitude - planet_longitude)
                if ketu_diff > 180:
                    ketu_diff = 360 - ketu_diff
                
                if rahu_diff <= 10:
                    volatile_factors += 2
                    key_factors.append(f"Rahu conjunct {get_planet_name(planet_id)} - unpredictable market expansion")
                
                if ketu_diff <= 10:
                    volatile_factors += 2
                    key_factors.append(f"Ketu conjunct {get_planet_name(planet_id)} - unexpected market contraction")
    
    # Calculate trend strength and determine primary trend
    total_factors = bullish_factors + bearish_factors + volatile_factors
    if total_factors == 0:
        total_factors = 1  # Avoid division by zero
    
    bullish_strength = (bullish_factors / total_factors) * 100
    bearish_strength = (bearish_factors / total_factors) * 100
    volatile_strength = (volatile_factors / total_factors) * 100
    
    # Determine primary trend
    if volatile_strength > 40:
        primary_trend = "volatile"
        trend_strength = volatile_strength
    elif bullish_strength > bearish_strength:
        primary_trend = "bullish"
        trend_strength = bullish_strength
    elif bearish_strength > bullish_strength:
        primary_trend = "bearish"
        trend_strength = bearish_strength
    else:
        primary_trend = "neutral"
        trend_strength = 50
    
    # Calculate reversal probability
    reversal_probability = volatile_strength * 0.7 + len(retrograde_planets) * 10
    if reversal_probability > 100:
        reversal_probability = 100
    
    return {
        "primary_trend": primary_trend,
        "strength": trend_strength,
        "key_factors": key_factors,
        "reversal_probability": reversal_probability
    }


def get_planet_name(planet_id: int) -> str:
    """
    Get the name of a planet from its Swiss Ephemeris ID.
    
    Args:
        planet_id: Swiss Ephemeris planet ID
        
    Returns:
        Planet name as string
    """
    planet_names = {
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
    
    return planet_names.get(planet_id, f"Unknown Planet {planet_id}")


def analyze_financial_yogas(
    planetary_positions: Dict[int, Dict[str, Union[float, bool]]],
    calculator: PlanetaryCalculator
) -> List[Dict[str, Union[str, float]]]:
    """
    Analyze financial yogas (planetary combinations) relevant to market prediction.
    
    Args:
        planetary_positions: Dictionary of planetary positions
        calculator: PlanetaryCalculator instance for additional calculations
        
    Returns:
        List of financial yogas with their names, strengths, and descriptions
    """
    yogas = []
    
    # 1. Lakshmi Yoga - Jupiter in own sign or exalted and Venus well-placed
    # Indicates financial prosperity and market growth
    if JUPITER in planetary_positions and VENUS in planetary_positions:
        jupiter_longitude = planetary_positions[JUPITER]["longitude"]
        venus_longitude = planetary_positions[VENUS]["longitude"]
        
        jupiter_sign = int(jupiter_longitude / 30)
        venus_sign = int(venus_longitude / 30)
        
        # Jupiter in own sign (Sagittarius or Pisces) or exalted (Cancer)
        jupiter_strong = jupiter_sign in [8, 11, 3]  # Sagittarius, Pisces, Cancer
        
        # Venus in own sign (Taurus or Libra) or exalted (Pisces)
        venus_strong = venus_sign in [1, 6, 11]  # Taurus, Libra, Pisces
        
        if jupiter_strong and venus_strong:
            yogas.append({
                "name": "Lakshmi Yoga",
                "strength": 85,
                "description": "Strong Jupiter and Venus indicate financial prosperity and market growth",
                "market_impact": "bullish"
            })
    
    # 2. Dhana Yoga - 2nd and 11th house lords well-placed
    # Simplified version for market analysis
    if JUPITER in planetary_positions and VENUS in planetary_positions and MERCURY in planetary_positions:
        # Check if these financial planets are strong (not retrograde and in good signs)
        jupiter_strong = not planetary_positions[JUPITER]["is_retrograde"]
        venus_strong = not planetary_positions[VENUS]["is_retrograde"]
        mercury_strong = not planetary_positions[MERCURY]["is_retrograde"]
        
        if jupiter_strong and venus_strong and mercury_strong:
            yogas.append({
                "name": "Dhana Yoga",
                "strength": 70,
                "description": "Strong financial planets indicate wealth generation",
                "market_impact": "bullish"
            })
    
    # 3. Viparita Raja Yoga - Malefics in 6th, 8th, or 12th houses
    # Simplified for market analysis - indicates unexpected market reversals
    if SATURN in planetary_positions and MARS in planetary_positions:
        saturn_longitude = planetary_positions[SATURN]["longitude"]
        mars_longitude = planetary_positions[MARS]["longitude"]
        
        # Check if Saturn and Mars are in opposition (180 degrees)
        angle_diff = abs(saturn_longitude - mars_longitude)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        if 170 <= angle_diff <= 190:  # Within 10 degrees of opposition
            yogas.append({
                "name": "Viparita Raja Yoga",
                "strength": 75,
                "description": "Saturn and Mars in opposition indicate unexpected market reversals",
                "market_impact": "volatile"
            })
    
    # 4. Neecha Bhanga Raja Yoga - Debilitated planet getting cancellation
    # Simplified for market analysis - indicates recovery from market lows
    if JUPITER in planetary_positions and SATURN in planetary_positions:
        jupiter_longitude = planetary_positions[JUPITER]["longitude"]
        saturn_longitude = planetary_positions[SATURN]["longitude"]
        
        jupiter_sign = int(jupiter_longitude / 30)
        saturn_sign = int(saturn_longitude / 30)
        
        # Jupiter debilitated in Capricorn but Saturn (lord of Capricorn) is strong
        if jupiter_sign == 9 and saturn_sign in [9, 10]:  # Jupiter in Capricorn, Saturn in Capricorn or Aquarius
            yogas.append({
                "name": "Neecha Bhanga Raja Yoga",
                "strength": 65,
                "description": "Debilitated Jupiter with strong Saturn indicates recovery from market lows",
                "market_impact": "bullish"
            })
    
    return yogas