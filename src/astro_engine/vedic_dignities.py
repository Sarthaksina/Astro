# Cosmic Market Oracle - Vedic Dignities Module

"""
This module calculates planetary dignities and strengths according to Vedic astrological principles,
with a focus on financial market applications.

It includes calculations for:
- Shadbala (six-fold strength)
- Planetary dignity states (exaltation, debilitation, own sign, etc.)
- Combustion status
- Retrograde strength
- Avastha (planetary states)
- Vargottama (same sign in D-1 and D-9)
"""

from typing import Dict, List, Optional, Union, Tuple
import math
from .planetary_positions import (
    PlanetaryCalculator, SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU
)

# Planetary dignity definitions
EXALTATION_DEGREES = {
    SUN: 10,      # Aries 10°
    MOON: 3,      # Taurus 3°
    MERCURY: 15,  # Virgo 15°
    VENUS: 27,    # Pisces 27°
    MARS: 28,     # Capricorn 28°
    JUPITER: 5,   # Cancer 5°
    SATURN: 20,   # Libra 20°
    RAHU: 20,     # Taurus 20° (some traditions)
    KETU: 20      # Scorpio 20° (some traditions)
}

EXALTATION_SIGNS = {
    SUN: 0,       # Aries
    MOON: 1,      # Taurus
    MERCURY: 5,   # Virgo
    VENUS: 11,    # Pisces
    MARS: 9,      # Capricorn
    JUPITER: 3,   # Cancer
    SATURN: 6,    # Libra
    RAHU: 1,      # Taurus
    KETU: 7       # Scorpio
}

DEBILITATION_SIGNS = {
    SUN: 6,       # Libra
    MOON: 7,      # Scorpio
    MERCURY: 11,  # Pisces
    VENUS: 5,     # Virgo
    MARS: 3,      # Cancer
    JUPITER: 9,   # Capricorn
    SATURN: 0,    # Aries
    RAHU: 7,      # Scorpio
    KETU: 1       # Taurus
}

OWN_SIGNS = {
    SUN: [4],                 # Leo
    MOON: [3],                # Cancer
    MERCURY: [2, 5],          # Gemini, Virgo
    VENUS: [1, 6],            # Taurus, Libra
    MARS: [0, 7],             # Aries, Scorpio
    JUPITER: [8, 11],         # Sagittarius, Pisces
    SATURN: [9, 10],          # Capricorn, Aquarius
    RAHU: [],                 # No own sign
    KETU: []                  # No own sign
}

# Combustion limits (degrees from Sun)
COMBUSTION_LIMITS = {
    MERCURY: 14,
    VENUS: 10,
    MARS: 17,
    JUPITER: 11,
    SATURN: 15
}

# Planetary friendship chart (0=enemy, 1=neutral, 2=friend)
PLANETARY_RELATIONSHIPS = {
    SUN: {SUN: 1, MOON: 2, MERCURY: 1, VENUS: 0, MARS: 2, JUPITER: 2, SATURN: 0, RAHU: 0, KETU: 0},
    MOON: {SUN: 2, MOON: 1, MERCURY: 1, VENUS: 2, MARS: 0, JUPITER: 2, SATURN: 0, RAHU: 0, KETU: 0},
    MERCURY: {SUN: 2, MOON: 1, MERCURY: 1, VENUS: 2, MARS: 1, JUPITER: 0, SATURN: 2, RAHU: 1, KETU: 1},
    VENUS: {SUN: 0, MOON: 2, MERCURY: 2, VENUS: 1, MARS: 0, JUPITER: 0, SATURN: 2, RAHU: 1, KETU: 1},
    MARS: {SUN: 2, MOON: 0, MERCURY: 1, VENUS: 0, MARS: 1, JUPITER: 2, SATURN: 0, RAHU: 0, KETU: 0},
    JUPITER: {SUN: 2, MOON: 2, MERCURY: 0, VENUS: 0, MARS: 2, JUPITER: 1, SATURN: 0, RAHU: 0, KETU: 0},
    SATURN: {SUN: 0, MOON: 0, MERCURY: 2, VENUS: 2, MARS: 0, JUPITER: 0, SATURN: 1, RAHU: 2, KETU: 2}
}

# Sign lords
SIGN_LORDS = {
    0: MARS,      # Aries
    1: VENUS,     # Taurus
    2: MERCURY,   # Gemini
    3: MOON,      # Cancer
    4: SUN,       # Leo
    5: MERCURY,   # Virgo
    6: VENUS,     # Libra
    7: MARS,      # Scorpio
    8: JUPITER,   # Sagittarius
    9: SATURN,    # Capricorn
    10: SATURN,   # Aquarius
    11: JUPITER   # Pisces
}


def calculate_dignity_state(planet_id: int, longitude: float) -> Dict[str, Union[str, float]]:
    """
    Calculate the dignity state of a planet.
    
    Args:
        planet_id: Swiss Ephemeris planet ID
        longitude: Zodiacal longitude in degrees (0-360)
        
    Returns:
        Dictionary with dignity information:
            - state: Dignity state (exalted, debilitated, own_sign, friendly, neutral, enemy)
            - strength: Dignity strength (0-1)
            - exact_degree: Whether at exact exaltation/debilitation degree
    """
    sign = int(longitude / 30)
    degree_in_sign = longitude % 30
    
    # Initialize result
    result = {
        "state": "neutral",
        "strength": 0.5,
        "exact_degree": False
    }
    
    # Check exaltation
    if sign == EXALTATION_SIGNS.get(planet_id):
        result["state"] = "exalted"
        result["strength"] = 1.0
        
        # Check if at exact exaltation degree
        if abs(degree_in_sign - EXALTATION_DEGREES.get(planet_id, 0)) < 1:
            result["exact_degree"] = True
            result["strength"] = 1.0  # Maximum strength
        else:
            # Strength decreases as it moves away from exact degree
            distance = min(
                abs(degree_in_sign - EXALTATION_DEGREES.get(planet_id, 0)),
                30 - abs(degree_in_sign - EXALTATION_DEGREES.get(planet_id, 0))
            )
            result["strength"] = 1.0 - (distance / 30)
    
    # Check debilitation
    elif sign == DEBILITATION_SIGNS.get(planet_id):
        result["state"] = "debilitated"
        result["strength"] = 0.0
        
        # Check if at exact debilitation degree (opposite of exaltation)
        if abs(degree_in_sign - EXALTATION_DEGREES.get(planet_id, 0)) < 1:
            result["exact_degree"] = True
            result["strength"] = 0.0  # Minimum strength
        else:
            # Strength increases as it moves away from exact degree
            distance = min(
                abs(degree_in_sign - EXALTATION_DEGREES.get(planet_id, 0)),
                30 - abs(degree_in_sign - EXALTATION_DEGREES.get(planet_id, 0))
            )
            result["strength"] = distance / 30
    
    # Check own sign
    elif sign in OWN_SIGNS.get(planet_id, []):
        result["state"] = "own_sign"
        result["strength"] = 0.85
    
    # For other states, we need to know the sign lord
    else:
        sign_lord = SIGN_LORDS.get(sign)
        
        # Skip if sign lord not defined or same as planet
        if sign_lord is None or sign_lord == planet_id:
            return result
        
        # Check relationship with sign lord
        relationship = PLANETARY_RELATIONSHIPS.get(planet_id, {}).get(sign_lord, 1)
        
        if relationship == 2:  # Friend
            result["state"] = "friendly"
            result["strength"] = 0.75
        elif relationship == 0:  # Enemy
            result["state"] = "enemy"
            result["strength"] = 0.25
    
    return result


def check_combustion(planet_id: int, planet_longitude: float, sun_longitude: float) -> Dict[str, Union[bool, float]]:
    """
    Check if a planet is combust (too close to the Sun).
    
    Args:
        planet_id: Swiss Ephemeris planet ID
        planet_longitude: Planet's zodiacal longitude in degrees (0-360)
        sun_longitude: Sun's zodiacal longitude in degrees (0-360)
        
    Returns:
        Dictionary with combustion information:
            - is_combust: Whether the planet is combust
            - combustion_degree: Degree of combustion (0-1, where 1 is exact conjunction)
    """
    # Moon and Sun can't be combust
    if planet_id in [SUN, MOON, RAHU, KETU]:
        return {"is_combust": False, "combustion_degree": 0.0}
    
    # Calculate angular distance from Sun
    angular_distance = abs(planet_longitude - sun_longitude)
    if angular_distance > 180:
        angular_distance = 360 - angular_distance
    
    # Check if within combustion limit
    combustion_limit = COMBUSTION_LIMITS.get(planet_id, 0)
    is_combust = angular_distance < combustion_limit
    
    # Calculate degree of combustion (1 = exact conjunction, 0 = at the limit)
    combustion_degree = 0.0
    if is_combust and combustion_limit > 0:
        combustion_degree = 1.0 - (angular_distance / combustion_limit)
    
    return {
        "is_combust": is_combust,
        "combustion_degree": combustion_degree
    }


def calculate_vargottama(rashi_longitude: float, navamsa_longitude: float) -> bool:
    """
    Check if a planet is in Vargottama (same sign in D-1 and D-9).
    
    Args:
        rashi_longitude: Planet's longitude in D-1 chart (0-360)
        navamsa_longitude: Planet's longitude in D-9 chart (0-360)
        
    Returns:
        True if the planet is in Vargottama, False otherwise
    """
    rashi_sign = int(rashi_longitude / 30)
    navamsa_sign = int(navamsa_longitude / 30)
    
    return rashi_sign == navamsa_sign


def calculate_shadbala(
    planet_id: int, 
    longitude: float, 
    is_retrograde: bool,
    all_positions: Dict[int, Dict[str, Union[float, bool]]],
    calculator: PlanetaryCalculator
) -> Dict[str, float]:
    """
    Calculate Shadbala (six-fold strength) of a planet.
    
    Args:
        planet_id: Swiss Ephemeris planet ID
        longitude: Planet's zodiacal longitude in degrees (0-360)
        is_retrograde: Whether the planet is retrograde
        all_positions: Dictionary of all planetary positions
        calculator: PlanetaryCalculator instance
        
    Returns:
        Dictionary with Shadbala components:
            - sthana_bala: Positional strength
            - dig_bala: Directional strength
            - kala_bala: Temporal strength
            - chesta_bala: Motional strength
            - naisargika_bala: Natural strength
            - drik_bala: Aspectual strength
            - total_shadbala: Total of all strengths
            - shadbala_ratio: Ratio to minimum required strength
    """
    # Initialize result with default values
    result = {
        "sthana_bala": 0.0,
        "dig_bala": 0.0,
        "kala_bala": 0.0,
        "chesta_bala": 0.0,
        "naisargika_bala": 0.0,
        "drik_bala": 0.0,
        "total_shadbala": 0.0,
        "shadbala_ratio": 0.0
    }
    
    # 1. Sthana Bala (Positional Strength)
    dignity = calculate_dignity_state(planet_id, longitude)
    if dignity["state"] == "exalted":
        result["sthana_bala"] = 1.0
    elif dignity["state"] == "own_sign":
        result["sthana_bala"] = 0.75
    elif dignity["state"] == "friendly":
        result["sthana_bala"] = 0.5
    elif dignity["state"] == "neutral":
        result["sthana_bala"] = 0.25
    elif dignity["state"] == "enemy":
        result["sthana_bala"] = 0.1
    elif dignity["state"] == "debilitated":
        result["sthana_bala"] = 0.0
    
    # 2. Dig Bala (Directional Strength)
    # Simplified calculation based on quadrants
    sign = int(longitude / 30)
    quadrant = sign % 4
    
    # Different planets have strength in different quadrants
    if planet_id == JUPITER and quadrant == 0:  # Jupiter in 1st quadrant
        result["dig_bala"] = 1.0
    elif planet_id == SUN and quadrant == 1:  # Sun in 2nd quadrant
        result["dig_bala"] = 1.0
    elif planet_id == SATURN and quadrant == 2:  # Saturn in 3rd quadrant
        result["dig_bala"] = 1.0
    elif planet_id == MOON or planet_id == VENUS and quadrant == 3:  # Moon/Venus in 4th quadrant
        result["dig_bala"] = 1.0
    elif planet_id == MARS and (quadrant == 0 or quadrant == 3):  # Mars in 1st or 4th
        result["dig_bala"] = 0.75
    elif planet_id == MERCURY:  # Mercury has moderate strength everywhere
        result["dig_bala"] = 0.5
    else:
        result["dig_bala"] = 0.25
    
    # 3. Kala Bala (Temporal Strength) - simplified
    # In a full implementation, this would consider day/night, seasons, etc.
    result["kala_bala"] = 0.5  # Default value
    
    # 4. Chesta Bala (Motional Strength)
    if is_retrograde:
        if planet_id in [JUPITER, VENUS, MERCURY]:  # Benefics
            result["chesta_bala"] = 0.6  # Retrograde benefics lose some strength
        else:  # Malefics
            result["chesta_bala"] = 0.8  # Retrograde malefics gain strength
    else:
        if planet_id in [JUPITER, VENUS, MERCURY]:  # Benefics
            result["chesta_bala"] = 0.8  # Direct benefics have good strength
        else:  # Malefics
            result["chesta_bala"] = 0.6  # Direct malefics have moderate strength
    
    # 5. Naisargika Bala (Natural Strength)
    # Fixed natural strengths of planets
    natural_strengths = {
        SUN: 0.6,
        MOON: 0.6,
        MARS: 0.7,
        MERCURY: 0.5,
        JUPITER: 0.8,
        VENUS: 0.7,
        SATURN: 0.5
    }
    result["naisargika_bala"] = natural_strengths.get(planet_id, 0.5)
    
    # 6. Drik Bala (Aspectual Strength)
    # Simplified - check aspects from other planets
    result["drik_bala"] = 0.5  # Default value
    
    # Calculate total Shadbala
    result["total_shadbala"] = (
        result["sthana_bala"] + 
        result["dig_bala"] + 
        result["kala_bala"] + 
        result["chesta_bala"] + 
        result["naisargika_bala"] + 
        result["drik_bala"]
    )
    
    # Calculate Shadbala ratio (relative to minimum required)
    # Minimum required values vary by planet in traditional Vedic astrology
    minimum_required = {
        SUN: 3.5,
        MOON: 3.0,
        MARS: 3.5,
        MERCURY: 3.5,
        JUPITER: 3.5,
        VENUS: 3.0,
        SATURN: 3.0
    }
    min_req = minimum_required.get(planet_id, 3.5)
    result["shadbala_ratio"] = result["total_shadbala"] / min_req
    
    return result


def calculate_all_dignities(
    positions: Dict[int, Dict[str, Union[float, bool]]],
    calculator: PlanetaryCalculator
) -> Dict[int, Dict[str, Union[str, float, bool, Dict]]]:
    """
    Calculate all dignity and strength parameters for all planets.
    
    Args:
        positions: Dictionary of planetary positions
        calculator: PlanetaryCalculator instance
        
    Returns:
        Dictionary with all dignity and strength information for each planet
    """
    result = {}
    
    # Get Sun longitude for combustion check
    sun_longitude = positions.get(SUN, {}).get("longitude", 0.0)
    
    for planet_id, position in positions.items():
        # Skip nodes for some calculations
        if planet_id in [RAHU, KETU]:
            continue
            
        longitude = position.get("longitude", 0.0)
        is_retrograde = position.get("is_retrograde", False)
        
        # Calculate D9 (Navamsa) position
        navamsa_longitude = calculator.calculate_divisional_chart(longitude, 9)
        
        # Gather all dignity information
        planet_result = {
            "dignity": calculate_dignity_state(planet_id, longitude),
            "combustion": check_combustion(planet_id, longitude, sun_longitude),
            "vargottama": calculate_vargottama(longitude, navamsa_longitude),
            "shadbala": calculate_shadbala(planet_id, longitude, is_retrograde, positions, calculator)
        }
        
        # Calculate financial strength (custom for market prediction)
        financial_strength = calculate_financial_strength(
            planet_id, 
            planet_result["dignity"]["strength"],
            planet_result["combustion"]["combustion_degree"],
            planet_result["shadbala"]["shadbala_ratio"],
            is_retrograde
        )
        planet_result["financial_strength"] = financial_strength
        
        result[planet_id] = planet_result
    
    return result


def calculate_financial_strength(
    planet_id: int,
    dignity_strength: float,
    combustion_degree: float,
    shadbala_ratio: float,
    is_retrograde: bool
) -> Dict[str, float]:
    """
    Calculate the financial strength of a planet for market prediction.
    
    Args:
        planet_id: Swiss Ephemeris planet ID
        dignity_strength: Dignity strength (0-1)
        combustion_degree: Degree of combustion (0-1)
        shadbala_ratio: Shadbala ratio
        is_retrograde: Whether the planet is retrograde
        
    Returns:
        Dictionary with financial strength components:
            - overall: Overall financial strength (0-1)
            - trend_influence: Influence on market trend (0-1)
            - volatility_factor: Contribution to market volatility (0-1)
            - reversal_potential: Potential to cause market reversals (0-1)
    """
    # Base financial importance of each planet
    financial_importance = {
        SUN: 0.6,      # Government, leadership, authority
        MOON: 0.7,     # Public sentiment, liquidity
        MERCURY: 0.8,  # Communication, trade, commerce
        VENUS: 0.9,    # Wealth, luxury, prosperity
        MARS: 0.6,     # Energy, action, conflict
        JUPITER: 1.0,  # Expansion, optimism, growth
        SATURN: 0.9,   # Contraction, discipline, restriction
        RAHU: 0.7,     # Speculation, innovation
        KETU: 0.5      # Sudden changes, disruption
    }
    
    # Calculate base strength
    base_strength = dignity_strength * (1.0 - combustion_degree) * shadbala_ratio
    
    # Adjust for retrograde motion
    if is_retrograde:
        if planet_id in [JUPITER, VENUS]:  # Financial benefics
            base_strength *= 0.7  # Weakened when retrograde
        elif planet_id in [SATURN, MARS]:  # Financial malefics
            base_strength *= 1.2  # Strengthened when retrograde (but cap at 1.0)
            base_strength = min(base_strength, 1.0)
    
    # Calculate overall financial strength
    overall = base_strength * financial_importance.get(planet_id, 0.5)
    
    # Calculate specialized financial factors
    
    # Trend influence (how strongly it affects market direction)
    if planet_id in [JUPITER, VENUS, SUN]:  # Bullish planets
        trend_influence = overall
    elif planet_id in [SATURN, MARS, KETU]:  # Bearish planets
        trend_influence = 1.0 - overall
    else:  # Neutral planets
        trend_influence = 0.5
    
    # Volatility factor (how much it contributes to market volatility)
    if planet_id in [MARS, RAHU, KETU]:  # Volatile planets
        volatility_factor = overall
    elif planet_id in [JUPITER, VENUS]:  # Stability planets
        volatility_factor = 0.3 * overall
    else:  # Moderate planets
        volatility_factor = 0.5 * overall
    
    # Reversal potential (ability to cause market reversals)
    if planet_id in [MARS, RAHU, MERCURY] and is_retrograde:  # Strong reversal planets
        reversal_potential = overall
    elif planet_id in [SATURN, KETU]:  # Moderate reversal planets
        reversal_potential = 0.7 * overall
    else:  # Weak reversal planets
        reversal_potential = 0.3 * overall
    
    return {
        "overall": overall,
        "trend_influence": trend_influence,
        "volatility_factor": volatility_factor,
        "reversal_potential": reversal_potential
    }
