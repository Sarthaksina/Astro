"""
Pattern Detection Module for the Cosmic Market Oracle.

This module implements specialized algorithms for detecting planetary patterns
and combinations with potential market impact, supporting advanced feature
engineering for financial astrology.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from datetime import datetime, timedelta
import pandas as pd

# Import constants from the centralized constants module
from src.astro_engine.constants import (
    SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO, RAHU, KETU,
    get_planet_name
)

# Import from centralized enums module
from .enums import FactorCategory, TimeFrame, FeatureType

# Import from astrological_features with relative import
from .astrological_features import AstrologicalFeatureGenerator

# Define MarketImpact enum here since it's specific to this module
from enum import Enum, auto

class MarketImpact(Enum):
    """Potential market impact of astrological factors."""
    BULLISH = auto()
    BEARISH = auto()
    VOLATILE = auto()
    NEUTRAL = auto()
    MIXED = auto()

# Define base classes for factors that were previously imported
class AstrologicalFactor:
    """Base class for astrological factors."""
    def __init__(self, name, category, description, market_impact, time_frames, strength=0.5, reliability=0.5):
        self.name = name
        self.category = category
        self.description = description
        self.market_impact = market_impact
        self.time_frames = time_frames
        self.strength = strength
        self.reliability = reliability

class PlanetaryPosition(AstrologicalFactor):
    """Planetary position factor."""
    def __init__(self, name, description, planet_id, **kwargs):
        super().__init__(name, FactorCategory.PLANETARY_POSITION, description, **kwargs)
        self.planet_id = planet_id

class PlanetaryAspect(AstrologicalFactor):
    """Planetary aspect factor."""
    def __init__(self, name, description, planet1_id, planet2_id, aspect_angle, **kwargs):
        super().__init__(name, FactorCategory.PLANETARY_ASPECT, description, **kwargs)
        self.planet1_id = planet1_id
        self.planet2_id = planet2_id
        self.aspect_angle = aspect_angle

class ZodiacalDistribution(AstrologicalFactor):
    """Zodiacal distribution factor."""
    def __init__(self, name, description, distribution_type, **kwargs):
        super().__init__(name, FactorCategory.ZODIACAL_DISTRIBUTION, description, **kwargs)
        self.distribution_type = distribution_type

class VedicFactor(AstrologicalFactor):
    """Vedic astrological factor."""
    def __init__(self, name, description, factor_type, planet_ids, **kwargs):
        super().__init__(name, FactorCategory.VEDIC_FACTOR, description, **kwargs)
        self.factor_type = factor_type
        self.planet_ids = planet_ids

class SensitivePoint(AstrologicalFactor):
    """Sensitive point factor."""
    def __init__(self, name, description, point_type, **kwargs):
        super().__init__(name, FactorCategory.SENSITIVE_POINT, description, **kwargs)
        self.point_type = point_type

class CyclicalPattern(AstrologicalFactor):
    """Cyclical pattern factor."""
    def __init__(self, name, description, cycle_type, period_days, **kwargs):
        super().__init__(name, FactorCategory.CYCLICAL_PATTERN, description, **kwargs)
        self.cycle_type = cycle_type
        self.period_days = period_days

class CompositeIndicator(AstrologicalFactor):
    """Composite indicator factor."""
    def __init__(self, name, description, components, **kwargs):
        super().__init__(name, FactorCategory.COMPOSITE_INDICATOR, description, **kwargs)
        self.components = components


class PatternDetector:
    """Base class for astrological pattern detection."""
    
    def __init__(self, calculator):
        """
        Initialize the pattern detector.
        
        Args:
            calculator: Astronomical calculator instance
        """
        self.calculator = calculator
        
    def detect_patterns(self, date: datetime, **kwargs) -> List[AstrologicalFactor]:
        """
        Detect astrological patterns for a given date.
        
        Args:
            date: Date to analyze
            **kwargs: Additional parameters
            
        Returns:
            List of detected astrological factors
        """
        raise NotImplementedError("Subclasses must implement detect_patterns")


class AspectPatternDetector(PatternDetector):
    """Detector for planetary aspect patterns."""
    
    def __init__(self, calculator, orb_tolerance: float = 5.0):
        """
        Initialize the aspect pattern detector.
        
        Args:
            calculator: Astronomical calculator instance
            orb_tolerance: Maximum orb in degrees for aspect detection
        """
        super().__init__(calculator)
        self.orb_tolerance = orb_tolerance
        
        # Define standard aspects
        self.aspects = [
            {"name": "Conjunction", "angle": 0, "orb": orb_tolerance, "harmonic": True},
            {"name": "Opposition", "angle": 180, "orb": orb_tolerance, "harmonic": False},
            {"name": "Trine", "angle": 120, "orb": orb_tolerance, "harmonic": True},
            {"name": "Square", "angle": 90, "orb": orb_tolerance, "harmonic": False},
            {"name": "Sextile", "angle": 60, "orb": orb_tolerance, "harmonic": True},
            {"name": "Quincunx", "angle": 150, "orb": orb_tolerance, "harmonic": False},
            {"name": "Semi-Square", "angle": 45, "orb": orb_tolerance, "harmonic": False},
            {"name": "Sesquiquadrate", "angle": 135, "orb": orb_tolerance, "harmonic": False}
        ]
        
        # Define planet pairs with financial significance
        self.financial_planet_pairs = [
            (JUPITER, SATURN),  # Economic cycles
            (VENUS, JUPITER),   # Financial expansion
            (VENUS, SATURN),    # Financial restriction
            (SUN, JUPITER),     # Growth and optimism
            (SUN, SATURN),      # Restriction and pessimism
            (MERCURY, JUPITER), # Trading and expansion
            (MERCURY, SATURN),  # Trading and restriction
            (MARS, JUPITER),    # Aggressive growth
            (MARS, SATURN),     # Aggressive restriction
            (JUPITER, URANUS),  # Sudden expansion
            (SATURN, URANUS),   # Sudden restriction
            (JUPITER, PLUTO),   # Transformative growth
            (SATURN, PLUTO),    # Transformative restriction
            (JUPITER, RAHU),    # Speculative growth
            (SATURN, RAHU),     # Speculative restriction
            (JUPITER, NEPTUNE), # Illusory growth
            (SATURN, NEPTUNE)   # Illusory restriction
        ]
        
    def detect_patterns(self, date: datetime, **kwargs) -> List[AstrologicalFactor]:
        """
        Detect aspect patterns for a given date.
        
        Args:
            date: Date to analyze
            **kwargs: Additional parameters
            
        Returns:
            List of detected aspect patterns
        """
        planets_data = self.calculator.get_all_planets(date, include_nodes=True)
        detected_patterns = []
        
        # Check for aspects between financially significant planet pairs
        for planet1_id, planet2_id in self.financial_planet_pairs:
            if planet1_id not in planets_data or planet2_id not in planets_data:
                continue
                
            lon1 = planets_data[planet1_id]["longitude"]
            lon2 = planets_data[planet2_id]["longitude"]
            
            # Calculate the angular separation
            angle = abs((lon1 - lon2) % 360)
            if angle > 180:
                angle = 360 - angle
                
            # Check if the angle matches any of our defined aspects
            for aspect in self.aspects:
                aspect_angle = aspect["angle"]
                orb = aspect["orb"]
                
                if abs(angle - aspect_angle) <= orb:
                    # Calculate the exact orb
                    exact_orb = abs(angle - aspect_angle)
                    
                    # Calculate normalized strength based on orb (closer = stronger)
                    strength = 1.0 - (exact_orb / orb)
                    
                    # Determine market impact based on planets and aspect type
                    market_impact = self._determine_aspect_impact(
                        planet1_id, planet2_id, aspect["name"], aspect["harmonic"]
                    )
                    
                    # Create a planetary aspect factor
                    planet1_name = self._get_planet_name(planet1_id)
                    planet2_name = self._get_planet_name(planet2_id)
                    
                    aspect_factor = PlanetaryAspect(
                        name=f"{planet1_name}-{planet2_name} {aspect['name']}",
                        description=f"{aspect['name']} aspect between {planet1_name} and {planet2_name}",
                        planet1_id=planet1_id,
                        planet2_id=planet2_id,
                        aspect_angle=aspect_angle,
                        orb=exact_orb,
                        market_impact=market_impact,
                        time_frames=self._get_aspect_time_frames(planet1_id, planet2_id),
                        strength=strength,
                        reliability=0.6  # Default reliability
                    )
                    
                    detected_patterns.append(aspect_factor)
                    
        return detected_patterns
    
    def _determine_aspect_impact(self, planet1_id: int, planet2_id: int, 
                                aspect_name: str, is_harmonic: bool) -> MarketImpact:
        """
        Determine the market impact of an aspect based on the planets involved.
        
        Args:
            planet1_id: First planet ID
            planet2_id: Second planet ID
            aspect_name: Name of the aspect
            is_harmonic: Whether the aspect is harmonic
            
        Returns:
            Market impact of the aspect
        """
        # Benefic planets
        benefics = {VENUS, JUPITER, MOON}
        
        # Malefic planets
        malefics = {MARS, SATURN, RAHU, KETU}
        
        # Neutral planets
        neutrals = {SUN, MERCURY, URANUS, NEPTUNE, PLUTO}
        
        # Determine the nature of the planets
        planet1_benefic = planet1_id in benefics
        planet1_malefic = planet1_id in malefics
        planet2_benefic = planet2_id in benefics
        planet2_malefic = planet2_id in malefics
        
        # Determine impact based on planet nature and aspect harmony
        if is_harmonic:
            if (planet1_benefic and planet2_benefic):
                return MarketImpact.BULLISH
            elif (planet1_malefic and planet2_malefic):
                return MarketImpact.BEARISH
            elif ((planet1_benefic and planet2_malefic) or 
                  (planet1_malefic and planet2_benefic)):
                return MarketImpact.MIXED
            else:
                return MarketImpact.NEUTRAL
        else:  # Disharmonic aspect
            if (planet1_benefic and planet2_benefic):
                return MarketImpact.MIXED
            elif (planet1_malefic and planet2_malefic):
                return MarketImpact.BEARISH
            elif ((planet1_benefic and planet2_malefic) or 
                  (planet1_malefic and planet2_benefic)):
                return MarketImpact.VOLATILE
            else:
                return MarketImpact.NEUTRAL
    
    def _get_aspect_time_frames(self, planet1_id: int, planet2_id: int) -> List[TimeFrame]:
        """
        Determine relevant time frames for an aspect based on the planets involved.
        
        Args:
            planet1_id: First planet ID
            planet2_id: Second planet ID
            
        Returns:
            List of relevant time frames
        """
        # Fast-moving planets
        fast_planets = {MOON, MERCURY, VENUS, SUN, MARS}
        
        # Medium-moving planets
        medium_planets = {JUPITER, SATURN}
        
        # Slow-moving planets
        slow_planets = {URANUS, NEPTUNE, PLUTO, RAHU, KETU}
        
        # Determine time frames based on planet speeds
        if planet1_id in fast_planets and planet2_id in fast_planets:
            return [TimeFrame.DAILY, TimeFrame.WEEKLY]
        elif ((planet1_id in fast_planets and planet2_id in medium_planets) or
              (planet1_id in medium_planets and planet2_id in fast_planets)):
            return [TimeFrame.WEEKLY, TimeFrame.MONTHLY]
        elif planet1_id in medium_planets and planet2_id in medium_planets:
            return [TimeFrame.MONTHLY, TimeFrame.QUARTERLY]
        elif ((planet1_id in fast_planets and planet2_id in slow_planets) or
              (planet1_id in slow_planets and planet2_id in fast_planets)):
            return [TimeFrame.MONTHLY, TimeFrame.QUARTERLY]
        elif ((planet1_id in medium_planets and planet2_id in slow_planets) or
              (planet1_id in slow_planets and planet2_id in medium_planets)):
            return [TimeFrame.QUARTERLY, TimeFrame.YEARLY]
        else:  # Both slow planets
            return [TimeFrame.YEARLY, TimeFrame.MULTI_YEAR]
    
    def _get_planet_name(self, planet_id: int) -> str:
        """
        Get the name of a planet based on its ID.
        
        Args:
            planet_id: Planet ID
            
        Returns:
            Planet name
        """
        planet_names = {
            SUN: "Sun", MOON: "Moon", MERCURY: "Mercury", VENUS: "Venus", 
            MARS: "Mars", JUPITER: "Jupiter", SATURN: "Saturn", 
            URANUS: "Uranus", NEPTUNE: "Neptune", PLUTO: "Pluto",
            RAHU: "Rahu", KETU: "Ketu"
        }
        return planet_names.get(planet_id, f"Planet-{planet_id}")


class GeometricPatternDetector(PatternDetector):
    """Detector for geometric planetary patterns."""
    
    def __init__(self, calculator, orb_tolerance: float = 5.0):
        """
        Initialize the geometric pattern detector.
        
        Args:
            calculator: Astronomical calculator instance
            orb_tolerance: Maximum orb in degrees for pattern detection
        """
        super().__init__(calculator)
        self.orb_tolerance = orb_tolerance
        
        # Define geometric patterns
        self.patterns = [
            {
                "name": "Grand Trine",
                "description": "Three planets forming equilateral triangle (120° aspects)",
                "angles": [120, 120, 120],
                "market_impact": MarketImpact.BULLISH,
                "time_frames": [TimeFrame.WEEKLY, TimeFrame.MONTHLY],
                "strength": 0.8,
                "reliability": 0.7
            },
            {
                "name": "Grand Square",
                "description": "Four planets forming a square (90° aspects)",
                "angles": [90, 90, 90, 90],
                "market_impact": MarketImpact.BEARISH,
                "time_frames": [TimeFrame.WEEKLY, TimeFrame.MONTHLY],
                "strength": 0.8,
                "reliability": 0.7
            },
            {
                "name": "T-Square",
                "description": "Three planets forming a T-shape (two 90° aspects and one 180° aspect)",
                "angles": [90, 90, 180],
                "market_impact": MarketImpact.VOLATILE,
                "time_frames": [TimeFrame.WEEKLY, TimeFrame.MONTHLY],
                "strength": 0.7,
                "reliability": 0.6
            },
            {
                "name": "Yod",
                "description": "Three planets forming a Y-shape (two 150° aspects and one 60° aspect)",
                "angles": [150, 150, 60],
                "market_impact": MarketImpact.VOLATILE,
                "time_frames": [TimeFrame.WEEKLY, TimeFrame.MONTHLY],
                "strength": 0.7,
                "reliability": 0.6
            },
            {
                "name": "Mystic Rectangle",
                "description": "Four planets forming a rectangle (two 60° aspects and two 120° aspects)",
                "angles": [60, 120, 60, 120],
                "market_impact": MarketImpact.BULLISH,
                "time_frames": [TimeFrame.WEEKLY, TimeFrame.MONTHLY],
                "strength": 0.7,
                "reliability": 0.6
            }
        ]
        
    def detect_patterns(self, date: datetime, **kwargs) -> List[AstrologicalFactor]:
        """
        Detect geometric patterns for a given date.
        
        Args:
            date: Date to analyze
            **kwargs: Additional parameters
            
        Returns:
            List of detected geometric patterns
        """
        planets_data = self.calculator.get_all_planets(date, include_nodes=True)
        detected_patterns = []
        
        # Get planet IDs and longitudes
        planet_ids = list(planets_data.keys())
        longitudes = [planets_data[pid]["longitude"] for pid in planet_ids]
        
        # Check for Grand Trine (three planets in trine aspects)
        for pattern in self.patterns:
            if pattern["name"] == "Grand Trine":
                for i in range(len(planet_ids)):
                    for j in range(i+1, len(planet_ids)):
                        for k in range(j+1, len(planet_ids)):
                            # Calculate angles between planets
                            angle_ij = self._angular_separation(longitudes[i], longitudes[j])
                            angle_jk = self._angular_separation(longitudes[j], longitudes[k])
                            angle_ki = self._angular_separation(longitudes[k], longitudes[i])
                            
                            # Check if angles match the pattern
                            if (abs(angle_ij - 120) <= self.orb_tolerance and
                                abs(angle_jk - 120) <= self.orb_tolerance and
                                abs(angle_ki - 120) <= self.orb_tolerance):
                                
                                # Create a composite indicator for the pattern
                                planet_ids_in_pattern = [planet_ids[i], planet_ids[j], planet_ids[k]]
                                planet_names = [self._get_planet_name(pid) for pid in planet_ids_in_pattern]
                                
                                pattern_factor = CompositeIndicator(
                                    name=f"{pattern['name']} ({', '.join(planet_names)})",
                                    description=f"{pattern['description']} involving {', '.join(planet_names)}",
                                    component_factors=[],  # Would be populated with individual aspects
                                    market_impact=pattern["market_impact"],
                                    time_frames=pattern["time_frames"],
                                    strength=pattern["strength"],
                                    reliability=pattern["reliability"]
                                )
                                
                                detected_patterns.append(pattern_factor)
            
            # Similar checks for other patterns...
            # (Implementation for other patterns would follow a similar approach)
        
        return detected_patterns
    
    def _angular_separation(self, lon1: float, lon2: float) -> float:
        """
        Calculate the angular separation between two longitudes.
        
        Args:
            lon1: First longitude
            lon2: Second longitude
            
        Returns:
            Angular separation in degrees (0-180)
        """
        angle = abs((lon1 - lon2) % 360)
        if angle > 180:
            angle = 360 - angle
        return angle
    
    def _get_planet_name(self, planet_id: int) -> str:
        """
        Get the name of a planet based on its ID.
        
        Args:
            planet_id: Planet ID
            
        Returns:
            Planet name
        """
        planet_names = {
            SUN: "Sun", MOON: "Moon", MERCURY: "Mercury", VENUS: "Venus", 
            MARS: "Mars", JUPITER: "Jupiter", SATURN: "Saturn", 
            URANUS: "Uranus", NEPTUNE: "Neptune", PLUTO: "Pluto",
            RAHU: "Rahu", KETU: "Ketu"
        }
        return planet_names.get(planet_id, f"Planet-{planet_id}")


class HistoricalCorrelationDetector(PatternDetector):
    """Detector for historically correlated astrological patterns."""
    
    def __init__(self, calculator, financial_data_provider, lookback_days: int = 365):
        """
        Initialize the historical correlation detector.
        
        Args:
            calculator: Astronomical calculator instance
            financial_data_provider: Provider of financial data
            lookback_days: Number of days to look back for historical correlations
        """
        super().__init__(calculator)
        self.financial_data_provider = financial_data_provider
        self.lookback_days = lookback_days
        self.correlation_cache = {}
        
    def detect_patterns(self, date: datetime, symbol: str = "SPY", **kwargs) -> List[AstrologicalFactor]:
        """
        Detect historically correlated patterns for a given date.
        
        Args:
            date: Date to analyze
            symbol: Financial symbol to analyze
            **kwargs: Additional parameters
            
        Returns:
            List of detected correlated patterns
        """
        detected_patterns = []
        
        # Get historical data
        start_date = date - timedelta(days=self.lookback_days)
        
        # Check if we have cached correlations for this symbol
        cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{date.strftime('%Y%m%d')}"
        if cache_key in self.correlation_cache:
            correlated_factors = self.correlation_cache[cache_key]
        else:
            # Get financial data
            financial_data = self.financial_data_provider.get_historical_data(
                symbol, start_date, date
            )
            
            # Calculate daily returns
            if len(financial_data) > 1:
                financial_data['return'] = financial_data['close'].pct_change()
                
                # Get astrological factors for each date
                dates = financial_data.index
                all_factors = []
                
                # Create aspect detector for getting factors
                aspect_detector = AspectPatternDetector(self.calculator)
                
                for d in dates:
                    factors = aspect_detector.detect_patterns(d)
                    for factor in factors:
                        all_factors.append({
                            'date': d,
                            'factor_name': factor.name,
                            'factor_strength': factor.strength
                        })
                
                # Convert to DataFrame
                if all_factors:
                    factors_df = pd.DataFrame(all_factors)
                    
                    # Pivot to get factor strength by date
                    pivot_df = factors_df.pivot_table(
                        index='date', 
                        columns='factor_name', 
                        values='factor_strength',
                        fill_value=0
                    )
                    
                    # Merge with financial data
                    merged_df = pd.merge(
                        financial_data[['return']], 
                        pivot_df, 
                        left_index=True, 
                        right_index=True,
                        how='left'
                    )
                    
                    # Fill NaN values with 0
                    merged_df = merged_df.fillna(0)
                    
                    # Calculate correlations
                    correlations = merged_df.corr()['return'].drop('return')
                    
                    # Filter significant correlations
                    significant_correlations = correlations[abs(correlations) > 0.3]
                    
                    # Create correlated factors
                    correlated_factors = []
                    
                    for factor_name, correlation in significant_correlations.items():
                        # Determine market impact based on correlation
                        if correlation > 0.3:
                            market_impact = MarketImpact.BULLISH
                        elif correlation < -0.3:
                            market_impact = MarketImpact.BEARISH
                        else:
                            market_impact = MarketImpact.NEUTRAL
                        
                        # Create a factor for the correlation
                        correlated_factor = AstrologicalFactor(
                            name=f"{factor_name} (Correlated)",
                            category=FactorCategory.COMPOSITE_INDICATOR,
                            description=f"Historical correlation of {correlation:.2f} with {symbol}",
                            market_impact=market_impact,
                            time_frames=[TimeFrame.DAILY, TimeFrame.WEEKLY],
                            strength=abs(correlation),
                            reliability=0.6
                        )
                        
                        correlated_factors.append(correlated_factor)
                    
                    # Cache the results
                    self.correlation_cache[cache_key] = correlated_factors
                else:
                    correlated_factors = []
            else:
                correlated_factors = []
        
        # Check if any of the correlated factors are present on the current date
        current_factors = AspectPatternDetector(self.calculator).detect_patterns(date)
        current_factor_names = {factor.name for factor in current_factors}
        
        for factor in correlated_factors:
            if factor.name.split(" (Correlated)")[0] in current_factor_names:
                detected_patterns.append(factor)
        
        return detected_patterns


class PatternDetectionManager:
    """Manager for coordinating multiple pattern detectors."""
    
    def __init__(self, calculator, financial_data_provider=None):
        """
        Initialize the pattern detection manager.
        
        Args:
            calculator: Astronomical calculator instance
            financial_data_provider: Provider of financial data (optional)
        """
        self.calculator = calculator
        self.financial_data_provider = financial_data_provider
        
        # Initialize detectors
        self.detectors = {
            'aspect': AspectPatternDetector(calculator),
            'geometric': GeometricPatternDetector(calculator)
        }
        
        # Initialize historical correlation detector if financial data provider is available
        if financial_data_provider:
            self.detectors['historical'] = HistoricalCorrelationDetector(
                calculator, financial_data_provider
            )
    
    def detect_all_patterns(self, date: datetime, **kwargs) -> Dict[str, List[AstrologicalFactor]]:
        """
        Detect all patterns for a given date.
        
        Args:
            date: Date to analyze
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of detected patterns by detector type
        """
        results = {}
        
        for detector_name, detector in self.detectors.items():
            results[detector_name] = detector.detect_patterns(date, **kwargs)
            
        return results
    
    def get_all_factors(self, date: datetime, **kwargs) -> List[AstrologicalFactor]:
        """
        Get all detected factors for a given date.
        
        Args:
            date: Date to analyze
            **kwargs: Additional parameters
            
        Returns:
            List of all detected factors
        """
        all_factors = []
        
        for detector_name, detector in self.detectors.items():
            all_factors.extend(detector.detect_patterns(date, **kwargs))
            
        return all_factors
    
    def get_factors_by_impact(self, date: datetime, impact: MarketImpact, **kwargs) -> List[AstrologicalFactor]:
        """
        Get factors with a specific market impact for a given date.
        
        Args:
            date: Date to analyze
            impact: Market impact to filter by
            **kwargs: Additional parameters
            
        Returns:
            List of factors with the specified market impact
        """
        all_factors = self.get_all_factors(date, **kwargs)
        return [factor for factor in all_factors if factor.market_impact == impact]
    
    def get_factors_by_time_frame(self, date: datetime, time_frame: TimeFrame, **kwargs) -> List[AstrologicalFactor]:
        """
        Get factors relevant to a specific time frame for a given date.
        
        Args:
            date: Date to analyze
            time_frame: Time frame to filter by
            **kwargs: Additional parameters
            
        Returns:
            List of factors relevant to the specified time frame
        """
        all_factors = self.get_all_factors(date, **kwargs)
        return [factor for factor in all_factors if time_frame in factor.time_frames]
    
    def get_market_sentiment(self, date: datetime, **kwargs) -> Dict[str, Any]:
        """
        Calculate overall market sentiment based on detected patterns.
        
        Args:
            date: Date to analyze
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with market sentiment metrics
        """
        all_factors = self.get_all_factors(date, **kwargs)
        
        if not all_factors:
            return {
                "sentiment": "neutral",
                "bullish_score": 0.5,
                "bearish_score": 0.5,
                "volatility_score": 0.5,
                "confidence": 0.0,
                "dominant_factors": []
            }
        
        # Count factors by impact
        bullish_factors = [f for f in all_factors if f.market_impact == MarketImpact.BULLISH]
        bearish_factors = [f for f in all_factors if f.market_impact == MarketImpact.BEARISH]
        volatile_factors = [f for f in all_factors if f.market_impact == MarketImpact.VOLATILE]
        mixed_factors = [f for f in all_factors if f.market_impact == MarketImpact.MIXED]
        
        # Calculate weighted scores
        bullish_score = sum(f.strength * f.reliability for f in bullish_factors)
        bearish_score = sum(f.strength * f.reliability for f in bearish_factors)
        volatile_score = sum(f.strength * f.reliability for f in volatile_factors)
        mixed_score = sum(f.strength * f.reliability for f in mixed_factors)
        
        # Normalize scores
        total_score = bullish_score + bearish_score + volatile_score + mixed_score
        if total_score > 0:
            bullish_score /= total_score
            bearish_score /= total_score
            volatile_score /= total_score
            mixed_score /= total_score
        else:
            bullish_score = bearish_score = volatile_score = mixed_score = 0.25
        
        # Determine overall sentiment
        if bullish_score > 0.4 and bullish_score > bearish_score:
            sentiment = "bullish"
        elif bearish_score > 0.4 and bearish_score > bullish_score:
            sentiment = "bearish"
        elif volatile_score > 0.4:
            sentiment = "volatile"
        else:
            sentiment = "neutral"
        
        # Calculate confidence
        confidence = max(bullish_score, bearish_score, volatile_score, mixed_score)
        
        # Get dominant factors (top 3 by strength * reliability)
        all_factors.sort(key=lambda f: f.strength * f.reliability, reverse=True)
        dominant_factors = all_factors[:3]
        
        return {
            "sentiment": sentiment,
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "volatility_score": volatile_score,
            "confidence": confidence,
            "dominant_factors": [f.name for f in dominant_factors]
        }


# Example usage
if __name__ == "__main__":
    from src.astro_engine.astronomical_calculator import AstronomicalCalculator
    
    # Initialize calculator
    calculator = AstronomicalCalculator()
    
    # Initialize pattern detection manager
    manager = PatternDetectionManager(calculator)
    
    # Detect patterns for current date
    current_date = datetime.now()
    patterns = manager.detect_all_patterns(current_date)
    
    print(f"Detected patterns for {current_date.strftime('%Y-%m-%d')}:")
    for detector_name, detector_patterns in patterns.items():
        print(f"\n{detector_name.capitalize()} patterns:")
        for pattern in detector_patterns:
            print(f"  - {pattern.name}: {pattern.description}")
            print(f"    Impact: {pattern.market_impact.name}, Strength: {pattern.strength:.2f}")
    
    # Get market sentiment
    sentiment = manager.get_market_sentiment(current_date)
    print(f"\nMarket sentiment: {sentiment['sentiment']}")
    print(f"Bullish score: {sentiment['bullish_score']:.2f}")
    print(f"Bearish score: {sentiment['bearish_score']:.2f}")
    print(f"Volatility score: {sentiment['volatility_score']:.2f}")
    print(f"Confidence: {sentiment['confidence']:.2f}")
    print(f"Dominant factors: {', '.join(sentiment['dominant_factors'])}")
