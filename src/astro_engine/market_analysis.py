"""
Market Analysis Module for the Cosmic Market Oracle.

This module provides comprehensive market analysis functionality by integrating
various astrological techniques including planetary positions, aspects, yogas,
and other factors to generate holistic market forecasts and trend predictions.
"""

from typing import Dict, List, Optional, Union, Tuple
import datetime
import numpy as np

from .planetary_positions import PlanetaryCalculator, get_planetary_aspects
from .constants import (
    SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO, RAHU, KETU,
    get_planet_name
)
from .financial_yogas import FinancialYogaAnalyzer


def analyze_market_trend(
    planetary_positions: Dict[int, Dict[str, Union[float, bool]]],
    date: Union[datetime.datetime, str, float],
    calculator: Optional[PlanetaryCalculator] = None
) -> Dict[str, Union[str, float, List[str]]]:
    """
    Analyze market trend based on planetary positions using Vedic astrology principles.
    
    Args:
        planetary_positions: Dictionary of planetary positions
        date: Date for analysis
        calculator: Optional PlanetaryCalculator instance
        
    Returns:
        Dictionary containing market analysis:
            - trend: Overall market trend (Bullish/Bearish/Sideways/Volatile)
            - strength: Numerical strength indicator (0-100)
            - key_factors: List of key influencing factors
            - aspects: Important planetary aspects
    """
    # Initialize calculator if not provided
    calculator = calculator or PlanetaryCalculator()
    
    # Initialize analysis variables
    bullish_strength = 0
    bearish_strength = 0
    volatile_strength = 0
    key_factors = []
    
    # Convert date to datetime if needed
    if isinstance(date, float):
        date = calculator.julian_day_to_datetime(date)
    elif isinstance(date, str):
        date = datetime.datetime.fromisoformat(date)
    
    # Analyze Moon's nakshatra
    if MOON in planetary_positions:
        moon_longitude = planetary_positions[MOON]["longitude"]
        moon_nakshatra = calculator.get_nakshatra_details(moon_longitude)
        
        if moon_nakshatra["financial_nature"] == "bullish":
            bullish_strength += 20
            key_factors.append(f"Moon in bullish {moon_nakshatra['nakshatra_name']}")
        elif moon_nakshatra["financial_nature"] == "bearish":
            bearish_strength += 20
            key_factors.append(f"Moon in bearish {moon_nakshatra['nakshatra_name']}")
        elif moon_nakshatra["financial_nature"] == "volatile":
            volatile_strength += 20
            key_factors.append(f"Moon in volatile {moon_nakshatra['nakshatra_name']}")
    
    # Check retrograde planets
    retrograde_planets = []
    financial_planets = [MERCURY, VENUS, JUPITER, SATURN]
    
    for planet in financial_planets:
        if planet in planetary_positions and planetary_positions[planet]["is_retrograde"]:
            retrograde_planets.append(planet)
    
    if MERCURY in retrograde_planets:
        volatile_strength += 20
        key_factors.append("Mercury retrograde - market volatility")
    
    if JUPITER in retrograde_planets:
        bearish_strength += 15
        key_factors.append("Jupiter retrograde - reduced optimism")
    
    if SATURN in retrograde_planets:
        bullish_strength += 15
        key_factors.append("Saturn retrograde - temporary easing")
    
    # Analyze aspects
    aspects = get_planetary_aspects(planetary_positions)
    
    for aspect in aspects:
        # Jupiter-Saturn aspects
        if (aspect["planet1"] == JUPITER and aspect["planet2"] == SATURN) or \
           (aspect["planet1"] == SATURN and aspect["planet2"] == JUPITER):
            if aspect["aspect_type"] == "conjunction":
                volatile_strength += 30
                key_factors.append("Jupiter-Saturn conjunction - major cycle shift")
            elif aspect["aspect_type"] == "trine":
                bullish_strength += 25
                key_factors.append("Jupiter-Saturn trine - economic stability")
            elif aspect["aspect_type"] == "square":
                bearish_strength += 25
                key_factors.append("Jupiter-Saturn square - economic tension")
        
        # Sun-Jupiter aspects
        if (aspect["planet1"] == SUN and aspect["planet2"] == JUPITER) or \
           (aspect["planet1"] == JUPITER and aspect["planet2"] == SUN):
            if aspect["aspect_type"] in ["conjunction", "trine"]:
                bullish_strength += 20
                key_factors.append("Sun-Jupiter aspect - market optimism")
        
        # Saturn-Mars aspects
        if (aspect["planet1"] == SATURN and aspect["planet2"] == MARS) or \
           (aspect["planet1"] == MARS and aspect["planet2"] == SATURN):
            if aspect["aspect_type"] in ["conjunction", "square", "opposition"]:
                bearish_strength += 20
                key_factors.append("Saturn-Mars aspect - market fear")
    
    # Calculate total strength and determine primary trend
    total_strength = bullish_strength + bearish_strength + volatile_strength
    
    if total_strength == 0:
        trend = "Sideways"
        strength = 0
    elif volatile_strength / total_strength > 0.4:  # If volatile strength > 40%
        trend = "Volatile"
        strength = volatile_strength
    elif bullish_strength > bearish_strength:
        trend = "Bullish"
        strength = bullish_strength
    else:
        trend = "Bearish"
        strength = bearish_strength
    
    return {
        "trend": trend,
        "strength": strength,
        "key_factors": key_factors,
        "aspects": aspects
    }


def analyze_comprehensive_market_forecast(
    date: Union[datetime.datetime, str, float],
    calculator: Optional[PlanetaryCalculator] = None,
    yoga_analyzer: Optional[FinancialYogaAnalyzer] = None
) -> Dict[str, Union[str, float, List[Dict], Dict]]:
    """
    Generate a comprehensive market forecast by combining multiple analysis techniques.
    
    Args:
        date: Date for analysis
        calculator: Optional PlanetaryCalculator instance
        yoga_analyzer: Optional FinancialYogaAnalyzer instance
        
    Returns:
        Dictionary containing comprehensive market forecast:
            - trend: Overall market trend (Bullish/Bearish/Sideways/Volatile)
            - confidence: Confidence level in the forecast (0-1)
            - volatility: Expected market volatility (Low/Moderate/High)
            - key_factors: List of key influencing factors
            - aspects: Important planetary aspects
            - yogas: Identified financial yogas
            - description: Textual description of the forecast
    """
    # Initialize components if not provided
    calculator = calculator or PlanetaryCalculator()
    yoga_analyzer = yoga_analyzer or FinancialYogaAnalyzer(calculator)
    
    # Get planetary positions
    positions = calculator.get_all_planets(date)
    
    # Get basic trend analysis
    trend_analysis = analyze_market_trend(positions, date, calculator)
    
    # Get yoga analysis
    yogas = yoga_analyzer.analyze_all_financial_yogas(positions)
    yoga_forecast = yoga_analyzer.get_market_forecast(yogas)
    
    # Combine analyses with appropriate weights
    trend_weight = 0.6
    yoga_weight = 0.4
    
    # Convert trend to numeric values for weighted calculation
    trend_numeric = {
        "Bullish": 1.0,
        "Bearish": -1.0,
        "Volatile": 0.0,
        "Sideways": 0.0
    }.get(trend_analysis["trend"], 0.0)
    
    yoga_numeric = {
        "bullish": 1.0,
        "bearish": -1.0,
        "neutral": 0.0
    }.get(yoga_forecast["trend"], 0.0)
    
    # Calculate combined trend
    combined_value = (trend_numeric * trend_weight) + (yoga_numeric * yoga_weight)
    
    if combined_value > 0.3:
        combined_trend = "Bullish"
    elif combined_value < -0.3:
        combined_trend = "Bearish"
    elif trend_analysis["trend"] == "Volatile" or yoga_forecast["volatility"] == "high":
        combined_trend = "Volatile"
    else:
        combined_trend = "Sideways"
    
    # Calculate confidence
    if combined_trend in ["Bullish", "Bearish"]:
        confidence = abs(combined_value) * 0.5 + 0.5  # Scale to 0.5-1.0
    else:
        confidence = 0.5  # Neutral confidence
    
    # Determine volatility
    if yoga_forecast["volatility"] == "high" or trend_analysis["trend"] == "Volatile":
        volatility = "High"
    elif yoga_forecast["volatility"] == "moderate":
        volatility = "Moderate"
    else:
        volatility = "Low"
    
    # Combine key factors
    key_factors = trend_analysis["key_factors"]
    
    # Generate description
    description = f"Market forecast: {combined_trend} with {volatility.lower()} volatility. "
    description += f"Confidence level: {confidence:.2f}. "
    
    if key_factors:
        description += f"Key factors include: {', '.join(key_factors[:3])}. "
    
    if yoga_forecast["description"]:
        description += yoga_forecast["description"]
    
    return {
        "trend": combined_trend,
        "confidence": confidence,
        "volatility": volatility,
        "key_factors": key_factors,
        "aspects": trend_analysis["aspects"],
        "yogas": yogas,
        "description": description
    }
def get_vedic_market_analysis(date, calculator=None):
    """
    Get comprehensive Vedic market analysis using the new VedicAnalyzer.
    This is the recommended function for new code.
    """
    from .vedic_analysis import VedicAnalyzer
    analyzer = VedicAnalyzer()
    return analyzer.analyze_date(date)

# For backward compatibility
__all__ = [
    'analyze_market_trend', 
    'analyze_comprehensive_market_forecast',
    'get_vedic_market_analysis'
]