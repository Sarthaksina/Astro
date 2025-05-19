"""
Composite Indicators Module for the Cosmic Market Oracle.

This module implements specialized composite indicators that combine multiple
astrological factors into unified market signals, supporting advanced
feature engineering for financial astrology.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import json

# Import constants from the centralized constants module
from src.astro_engine.constants import (
    SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO, RAHU, KETU,
    get_planet_name
)

# Import from centralized enums module
from .enums import TimeFrame, FactorCategory, FeatureType

# Import from astrological_features with relative import
from .astrological_features import AstrologicalFeatureGenerator

# Define MarketImpact enum here since it's specific to this module
class MarketImpact(Enum):
    """Potential market impact of astrological factors."""
    BULLISH = auto()
    BEARISH = auto()
    VOLATILE = auto()
    NEUTRAL = auto()
    MIXED = auto()


class CompositeIndicatorBuilder:
    """Builder for composite astrological indicators."""
    
    def __init__(self, calculator):
        """
        Initialize the composite indicator builder.
        
        Args:
            calculator: Astronomical calculator instance
        """
        self.calculator = calculator
        
        # Define standard composite indicators
        self.standard_indicators = [
            {
                "name": "Bull-Bear Index",
                "description": "Composite index of bullish vs. bearish astrological factors",
                "components": [
                    {"factor_type": "aspect", "planets": [JUPITER, VENUS], "weight": 1.0},
                    {"factor_type": "aspect", "planets": [SATURN, MARS], "weight": -1.0},
                    {"factor_type": "element", "value": "fire", "weight": 0.5},
                    {"factor_type": "element", "value": "earth", "weight": -0.5},
                    {"factor_type": "retrograde", "planets": [MERCURY], "weight": -0.3},
                    {"factor_type": "phase", "cycle": "moon_phase", "weight": 0.2}
                ],
                "normalization": "minmax",
                "time_frames": [TimeFrame.DAILY, TimeFrame.WEEKLY]
            },
            {
                "name": "Market Volatility Index",
                "description": "Composite index of astrological factors associated with market volatility",
                "components": [
                    {"factor_type": "aspect", "planets": [URANUS, MARS], "weight": 1.0},
                    {"factor_type": "aspect", "planets": [MERCURY, URANUS], "weight": 0.8},
                    {"factor_type": "retrograde", "planets": [MERCURY], "weight": 0.7},
                    {"factor_type": "aspect", "planets": [SUN, PLUTO], "weight": 0.6},
                    {"factor_type": "aspect", "planets": [MOON, URANUS], "weight": 0.5},
                    {"factor_type": "element", "value": "air", "weight": 0.4}
                ],
                "normalization": "minmax",
                "time_frames": [TimeFrame.DAILY, TimeFrame.WEEKLY]
            },
            {
                "name": "Long-Term Trend Index",
                "description": "Composite index of astrological factors associated with long-term market trends",
                "components": [
                    {"factor_type": "aspect", "planets": [JUPITER, SATURN], "weight": 1.0},
                    {"factor_type": "aspect", "planets": [SATURN, PLUTO], "weight": 0.8},
                    {"factor_type": "aspect", "planets": [JUPITER, PLUTO], "weight": 0.7},
                    {"factor_type": "aspect", "planets": [SATURN, URANUS], "weight": 0.6},
                    {"factor_type": "cycle", "name": "jupiter_saturn_cycle", "weight": 0.9},
                    {"factor_type": "cycle", "name": "saturn_uranus_cycle", "weight": 0.7}
                ],
                "normalization": "minmax",
                "time_frames": [TimeFrame.MONTHLY, TimeFrame.QUARTERLY, TimeFrame.YEARLY]
            },
            {
                "name": "Financial Speculation Index",
                "description": "Composite index of astrological factors associated with financial speculation",
                "components": [
                    {"factor_type": "aspect", "planets": [JUPITER, NEPTUNE], "weight": 1.0},
                    {"factor_type": "aspect", "planets": [VENUS, NEPTUNE], "weight": 0.8},
                    {"factor_type": "aspect", "planets": [MERCURY, JUPITER], "weight": 0.7},
                    {"factor_type": "aspect", "planets": [SUN, JUPITER], "weight": 0.6},
                    {"factor_type": "element", "value": "water", "weight": 0.5},
                    {"factor_type": "nakshatra", "planet": MOON, "financial_nature": "volatile", "weight": 0.7}
                ],
                "normalization": "minmax",
                "time_frames": [TimeFrame.DAILY, TimeFrame.WEEKLY]
            },
            {
                "name": "Economic Contraction Index",
                "description": "Composite index of astrological factors associated with economic contraction",
                "components": [
                    {"factor_type": "aspect", "planets": [SATURN, PLUTO], "weight": 1.0},
                    {"factor_type": "aspect", "planets": [MARS, SATURN], "weight": 0.8},
                    {"factor_type": "aspect", "planets": [SATURN, URANUS], "weight": 0.7},
                    {"factor_type": "retrograde", "planets": [JUPITER], "weight": 0.6},
                    {"factor_type": "element", "value": "earth", "weight": 0.5},
                    {"factor_type": "cycle", "name": "saturn_uranus_cycle", "phase_range": [0.4, 0.6], "weight": 0.9}
                ],
                "normalization": "minmax",
                "time_frames": [TimeFrame.MONTHLY, TimeFrame.QUARTERLY]
            }
        ]
    
    def build_indicators(self, date: datetime, pattern_detector=None, cycle_extractor=None) -> Dict[str, float]:
        """
        Build composite indicators for a given date.
        
        Args:
            date: Date to analyze
            pattern_detector: Pattern detector instance (optional)
            cycle_extractor: Cycle extractor instance (optional)
            
        Returns:
            Dictionary of composite indicator values
        """
        indicators = {}
        
        # Get planetary positions
        planets_data = self.calculator.get_all_planets(date, include_nodes=True)
        
        # Calculate element and modality distributions
        element_distribution = self._calculate_element_distribution(planets_data)
        modality_distribution = self._calculate_modality_distribution(planets_data)
        
        # Calculate retrograde status
        retrograde_status = {
            planet_id: planets_data[planet_id].get("is_retrograde", False)
            for planet_id in planets_data
        }
        
        # Get cycle phases if cycle extractor is provided
        cycle_phases = {}
        if cycle_extractor:
            cycle_features = cycle_extractor.extract_features(date)
            for feature_name, feature_value in cycle_features.items():
                if feature_name.endswith("_phase"):
                    cycle_name = feature_name.replace("_phase", "")
                    cycle_phases[cycle_name] = feature_value
        
        # Get pattern factors if pattern detector is provided
        pattern_factors = []
        if pattern_detector:
            pattern_factors = pattern_detector.get_all_factors(date)
        
        # Calculate nakshatra details for Moon
        moon_nakshatra = None
        if MOON in planets_data:
            moon_nakshatra = self.calculator.get_nakshatra_details(planets_data[MOON]["longitude"])
        
        # Build each composite indicator
        for indicator_def in self.standard_indicators:
            indicator_value = self._calculate_indicator_value(
                indicator_def, planets_data, element_distribution, modality_distribution,
                retrograde_status, cycle_phases, pattern_factors, moon_nakshatra
            )
            
            # Store the indicator value
            indicators[indicator_def["name"]] = indicator_value
            
            # Store normalized version (0-1 range)
            indicators[f"{indicator_def['name']}_normalized"] = self._normalize_indicator(
                indicator_value, indicator_def["normalization"]
            )
        
        return indicators
    
    def _calculate_indicator_value(self, indicator_def: Dict[str, Any], planets_data: Dict[int, Dict[str, Any]],
                                 element_distribution: Dict[str, float], modality_distribution: Dict[str, float],
                                 retrograde_status: Dict[int, bool], cycle_phases: Dict[str, float],
                                 pattern_factors: List[AstrologicalFactor], moon_nakshatra: Optional[Dict[str, Any]]) -> float:
        """
        Calculate the value of a composite indicator.
        
        Args:
            indicator_def: Indicator definition
            planets_data: Dictionary of planetary positions
            element_distribution: Element distribution
            modality_distribution: Modality distribution
            retrograde_status: Retrograde status by planet
            cycle_phases: Cycle phases by cycle name
            pattern_factors: List of pattern factors
            moon_nakshatra: Moon nakshatra details
            
        Returns:
            Indicator value
        """
        component_values = []
        
        for component in indicator_def["components"]:
            component_value = 0.0
            
            if component["factor_type"] == "aspect":
                # Check for aspects between specified planets
                planet1_id, planet2_id = component["planets"]
                if planet1_id in planets_data and planet2_id in planets_data:
                    # Check if there's a matching aspect in pattern factors
                    planet1_name = self._get_planet_name(planet1_id)
                    planet2_name = self._get_planet_name(planet2_id)
                    
                    for factor in pattern_factors:
                        if (hasattr(factor, 'planet1_id') and hasattr(factor, 'planet2_id') and
                            ((factor.planet1_id == planet1_id and factor.planet2_id == planet2_id) or
                             (factor.planet1_id == planet2_id and factor.planet2_id == planet1_id))):
                            component_value = factor.strength
                            break
            
            elif component["factor_type"] == "element":
                # Get element distribution value
                element = component["value"]
                component_value = element_distribution.get(element, 0.0)
            
            elif component["factor_type"] == "modality":
                # Get modality distribution value
                modality = component["value"]
                component_value = modality_distribution.get(modality, 0.0)
            
            elif component["factor_type"] == "retrograde":
                # Check retrograde status
                planet_id = component["planets"][0]
                component_value = 1.0 if retrograde_status.get(planet_id, False) else 0.0
            
            elif component["factor_type"] == "cycle":
                # Get cycle phase
                cycle_name = component["name"]
                if cycle_name in cycle_phases:
                    phase = cycle_phases[cycle_name]
                    
                    # Check if phase is in specified range (if provided)
                    if "phase_range" in component:
                        min_phase, max_phase = component["phase_range"]
                        if min_phase <= phase <= max_phase:
                            component_value = 1.0
                        else:
                            component_value = 0.0
                    else:
                        component_value = phase
            
            elif component["factor_type"] == "phase":
                # Get specific cycle phase
                cycle_name = component["cycle"]
                if cycle_name in cycle_phases:
                    component_value = cycle_phases[cycle_name]
            
            elif component["factor_type"] == "nakshatra":
                # Check Moon nakshatra
                if component["planet"] == MOON and moon_nakshatra:
                    if "financial_nature" in component:
                        if moon_nakshatra["financial_nature"] == component["financial_nature"]:
                            component_value = 1.0
                        else:
                            component_value = 0.0
                    else:
                        component_value = float(moon_nakshatra["nakshatra"]) / 27.0  # Normalize to 0-1
            
            # Apply weight
            weighted_value = component_value * component["weight"]
            component_values.append(weighted_value)
        
        # Sum all component values
        indicator_value = sum(component_values)
        
        return indicator_value
    
    def _normalize_indicator(self, value: float, method: str) -> float:
        """
        Normalize an indicator value.
        
        Args:
            value: Raw indicator value
            method: Normalization method
            
        Returns:
            Normalized value
        """
        if method == "minmax":
            # Simple min-max normalization to 0-1 range
            # Assuming typical range of -5 to 5 for raw values
            normalized = (value + 5) / 10.0
            return max(0.0, min(1.0, normalized))
        
        elif method == "sigmoid":
            # Sigmoid normalization
            return 1.0 / (1.0 + np.exp(-value))
        
        elif method == "tanh":
            # Hyperbolic tangent normalization
            return (np.tanh(value) + 1) / 2.0
        
        else:
            # Default: no normalization
            return value
    
    def _calculate_element_distribution(self, planets_data: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate the distribution of planets across elements.
        
        Args:
            planets_data: Dictionary of planetary positions
            
        Returns:
            Dictionary of element distributions
        """
        elements = {
            "fire": 0,
            "earth": 0,
            "air": 0,
            "water": 0
        }
        
        # Define which signs belong to which elements
        fire_signs = {0, 4, 8}    # Aries, Leo, Sagittarius
        earth_signs = {1, 5, 9}   # Taurus, Virgo, Capricorn
        air_signs = {2, 6, 10}    # Gemini, Libra, Aquarius
        water_signs = {3, 7, 11}  # Cancer, Scorpio, Pisces
        
        # Count planets in each element
        for planet_id, planet_data in planets_data.items():
            # Skip nodes
            if planet_id in {RAHU, KETU}:
                continue
                
            longitude = planet_data["longitude"]
            sign = int(longitude / 30)
            
            if sign in fire_signs:
                elements["fire"] += 1
            elif sign in earth_signs:
                elements["earth"] += 1
            elif sign in air_signs:
                elements["air"] += 1
            elif sign in water_signs:
                elements["water"] += 1
        
        # Calculate percentages
        total_planets = sum(elements.values())
        if total_planets > 0:
            for element in elements:
                elements[element] = elements[element] / total_planets
        
        return elements
    
    def _calculate_modality_distribution(self, planets_data: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate the distribution of planets across modalities.
        
        Args:
            planets_data: Dictionary of planetary positions
            
        Returns:
            Dictionary of modality distributions
        """
        modalities = {
            "cardinal": 0,
            "fixed": 0,
            "mutable": 0
        }
        
        # Define which signs belong to which modalities
        cardinal_signs = {0, 3, 6, 9}     # Aries, Cancer, Libra, Capricorn
        fixed_signs = {1, 4, 7, 10}       # Taurus, Leo, Scorpio, Aquarius
        mutable_signs = {2, 5, 8, 11}     # Gemini, Virgo, Sagittarius, Pisces
        
        # Count planets in each modality
        for planet_id, planet_data in planets_data.items():
            # Skip nodes
            if planet_id in {RAHU, KETU}:
                continue
                
            longitude = planet_data["longitude"]
            sign = int(longitude / 30)
            
            if sign in cardinal_signs:
                modalities["cardinal"] += 1
            elif sign in fixed_signs:
                modalities["fixed"] += 1
            elif sign in mutable_signs:
                modalities["mutable"] += 1
        
        # Calculate percentages
        total_planets = sum(modalities.values())
        if total_planets > 0:
            for modality in modalities:
                modalities[modality] = modalities[modality] / total_planets
        
        return modalities
    
    def _get_planet_name(self, planet_id: int) -> str:
        """
        [DEPRECATED] Use the centralized get_planet_name function from src.astro_engine.constants instead.
        
        Get the name of a planet based on its ID.
        
        Args:
            planet_id: Planet ID
            
        Returns:
            Planet name
        """
        return get_planet_name(planet_id)


class ExpertDefinedIndicator:
    """Expert-defined composite indicator based on astrological expertise."""
    
{{ ... }}
    def __init__(self, name: str, description: str, formula: str, calculator):
        """
        Initialize the expert-defined indicator.
        
        Args:
            name: Indicator name
            description: Indicator description
            formula: Formula in JSON format
            calculator: Astronomical calculator instance
        """
        self.name = name
        self.description = description
        self.formula = json.loads(formula)
        self.calculator = calculator
    
    def calculate(self, date: datetime, pattern_detector=None, cycle_extractor=None) -> float:
        """
        Calculate the indicator value for a given date.
        
        Args:
            date: Date to analyze
            pattern_detector: Pattern detector instance (optional)
            cycle_extractor: Cycle extractor instance (optional)
            
        Returns:
            Indicator value
        """
        # Get planetary positions
        planets_data = self.calculator.get_all_planets(date, include_nodes=True)
        
        # Parse and evaluate the formula
        return self._evaluate_formula(self.formula, planets_data, date, pattern_detector, cycle_extractor)
    
    def _evaluate_formula(self, formula: Dict[str, Any], planets_data: Dict[int, Dict[str, Any]], 
                         date: datetime, pattern_detector=None, cycle_extractor=None) -> float:
        """
        Evaluate a formula recursively.
        
        Args:
            formula: Formula definition
            planets_data: Dictionary of planetary positions
            date: Date to analyze
            pattern_detector: Pattern detector instance (optional)
            cycle_extractor: Cycle extractor instance (optional)
            
        Returns:
            Formula value
        """
        if "operation" in formula:
            operation = formula["operation"]
            operands = [self._evaluate_formula(op, planets_data, date, pattern_detector, cycle_extractor) 
                       for op in formula["operands"]]
            
            if operation == "add":
                return sum(operands)
            elif operation == "subtract":
                return operands[0] - sum(operands[1:])
            elif operation == "multiply":
                result = 1.0
                for op in operands:
                    result *= op
                return result
            elif operation == "divide":
                if operands[1] == 0:
                    return 0.0
                return operands[0] / operands[1]
            elif operation == "max":
                return max(operands)
            elif operation == "min":
                return min(operands)
            elif operation == "avg":
                return sum(operands) / len(operands)
            elif operation == "weighted_avg":
                weights = formula["weights"]
                return sum(op * w for op, w in zip(operands, weights)) / sum(weights)
        
        elif "factor_type" in formula:
            factor_type = formula["factor_type"]
            
            if factor_type == "planet_position":
                planet_id = formula["planet_id"]
                if planet_id in planets_data:
                    return planets_data[planet_id]["longitude"]
                return 0.0
            
            elif factor_type == "aspect":
                planet1_id = formula["planet1_id"]
                planet2_id = formula["planet2_id"]
                aspect_angle = formula["aspect_angle"]
                orb = formula.get("orb", 5.0)
                
                if planet1_id in planets_data and planet2_id in planets_data:
                    lon1 = planets_data[planet1_id]["longitude"]
                    lon2 = planets_data[planet2_id]["longitude"]
                    
                    # Calculate the angular separation
                    angle = abs((lon1 - lon2) % 360)
                    if angle > 180:
                        angle = 360 - angle
                    
                    # Check if within orb
                    if abs(angle - aspect_angle) <= orb:
                        # Calculate strength based on exactness
                        return 1.0 - (abs(angle - aspect_angle) / orb)
                
                return 0.0
            
            elif factor_type == "retrograde":
                planet_id = formula["planet_id"]
                if planet_id in planets_data and planets_data[planet_id].get("is_retrograde", False):
                    return 1.0
                return 0.0
            
            elif factor_type == "cycle_phase":
                if cycle_extractor:
                    cycle_name = formula["cycle_name"]
                    cycle_features = cycle_extractor.extract_features(date)
                    phase_key = f"{cycle_name}_phase"
                    
                    if phase_key in cycle_features:
                        phase = cycle_features[phase_key]
                        
                        if "phase_range" in formula:
                            min_phase, max_phase = formula["phase_range"]
                            if min_phase <= phase <= max_phase:
                                return 1.0
                            return 0.0
                        
                        return phase
                
                return 0.0
            
            elif factor_type == "pattern":
                if pattern_detector:
                    pattern_name = formula["pattern_name"]
                    patterns = pattern_detector.get_all_factors(date)
                    
                    for pattern in patterns:
                        if pattern.name == pattern_name:
                            return pattern.strength
                
                return 0.0
        
        elif "constant" in formula:
            return formula["constant"]
        
        return 0.0


class CompositeIndicatorManager:
    """Manager for coordinating multiple composite indicators."""
    
    def __init__(self, calculator, pattern_detector=None, cycle_extractor=None):
        """
        Initialize the composite indicator manager.
        
        Args:
            calculator: Astronomical calculator instance
            pattern_detector: Pattern detector instance (optional)
            cycle_extractor: Cycle extractor instance (optional)
        """
        self.calculator = calculator
        self.pattern_detector = pattern_detector
        self.cycle_extractor = cycle_extractor
        
        # Initialize standard indicator builder
        self.indicator_builder = CompositeIndicatorBuilder(calculator)
        
        # Initialize expert-defined indicators
        self.expert_indicators = []
        
        # Example expert-defined indicator
        self._initialize_expert_indicators()
    
    def _initialize_expert_indicators(self):
        """Initialize expert-defined indicators."""
        # Jupiter-Venus-Saturn Financial Index
        jvs_formula = json.dumps({
            "operation": "weighted_avg",
            "operands": [
                {
                    "factor_type": "aspect",
                    "planet1_id": JUPITER,
                    "planet2_id": VENUS,
                    "aspect_angle": 120,
                    "orb": 8.0
                },
                {
                    "factor_type": "aspect",
                    "planet1_id": VENUS,
                    "planet2_id": SATURN,
                    "aspect_angle": 90,
                    "orb": 8.0
                },
                {
                    "factor_type": "retrograde",
                    "planet_id": MERCURY
                }
            ],
            "weights": [1.0, -0.8, -0.5]
        })
        
        self.expert_indicators.append(
            ExpertDefinedIndicator(
                "Jupiter-Venus-Saturn Financial Index",
                "Expert-defined index based on Jupiter-Venus-Saturn relationships",
                jvs_formula,
                self.calculator
            )
        )
        
        # Market Momentum Index
        momentum_formula = json.dumps({
            "operation": "add",
            "operands": [
                {
                    "operation": "multiply",
                    "operands": [
                        {
                            "factor_type": "cycle_phase",
                            "cycle_name": "moon_phase",
                            "phase_range": [0.0, 0.25]
                        },
                        {
                            "constant": 0.5
                        }
                    ]
                },
                {
                    "operation": "multiply",
                    "operands": [
                        {
                            "factor_type": "aspect",
                            "planet1_id": JUPITER,
                            "planet2_id": SUN,
                            "aspect_angle": 120,
                            "orb": 10.0
                        },
                        {
                            "constant": 0.8
                        }
                    ]
                },
                {
                    "operation": "multiply",
                    "operands": [
                        {
                            "factor_type": "aspect",
                            "planet1_id": SATURN,
                            "planet2_id": MARS,
                            "aspect_angle": 90,
                            "orb": 8.0
                        },
                        {
                            "constant": -0.7
                        }
                    ]
                }
            ]
        })
        
        self.expert_indicators.append(
            ExpertDefinedIndicator(
                "Market Momentum Index",
                "Expert-defined index for market momentum based on multiple factors",
                momentum_formula,
                self.calculator
            )
        )
    
    def calculate_all_indicators(self, date: datetime) -> Dict[str, float]:
        """
        Calculate all composite indicators for a given date.
        
        Args:
            date: Date to analyze
            
        Returns:
            Dictionary of indicator values
        """
        results = {}
        
        # Calculate standard indicators
        standard_indicators = self.indicator_builder.build_indicators(
            date, self.pattern_detector, self.cycle_extractor
        )
        results.update(standard_indicators)
        
        # Calculate expert-defined indicators
        for indicator in self.expert_indicators:
            value = indicator.calculate(date, self.pattern_detector, self.cycle_extractor)
            results[indicator.name] = value
            
            # Add normalized version
            normalized_value = (value + 5) / 10.0  # Simple normalization assuming -5 to 5 range
            normalized_value = max(0.0, min(1.0, normalized_value))
            results[f"{indicator.name}_normalized"] = normalized_value
        
        return results
    
    def get_market_forecast(self, date: datetime) -> Dict[str, Any]:
        """
        Generate a market forecast based on composite indicators.
        
        Args:
            date: Date to analyze
            
        Returns:
            Dictionary with market forecast
        """
        indicators = self.calculate_all_indicators(date)
        
        # Extract normalized indicators
        normalized_indicators = {k: v for k, v in indicators.items() if k.endswith("_normalized")}
        
        # Calculate overall market sentiment
        bull_bear_index = normalized_indicators.get("Bull-Bear Index_normalized", 0.5)
        volatility_index = normalized_indicators.get("Market Volatility Index_normalized", 0.5)
        long_term_index = normalized_indicators.get("Long-Term Trend Index_normalized", 0.5)
        
        # Determine market direction
        if bull_bear_index > 0.7:
            direction = "strongly bullish"
        elif bull_bear_index > 0.55:
            direction = "moderately bullish"
        elif bull_bear_index < 0.3:
            direction = "strongly bearish"
        elif bull_bear_index < 0.45:
            direction = "moderately bearish"
        else:
            direction = "neutral"
        
        # Adjust for volatility
        if volatility_index > 0.7:
            if direction == "neutral":
                direction = "volatile"
            else:
                direction = f"{direction} with high volatility"
        
        # Generate forecast
        forecast = {
            "date": date.strftime("%Y-%m-%d"),
            "direction": direction,
            "confidence": abs(bull_bear_index - 0.5) * 2.0,  # 0-1 scale
            "volatility": volatility_index,
            "long_term_trend": "bullish" if long_term_index > 0.5 else "bearish",
            "indicators": normalized_indicators
        }
        
        return forecast


# Example usage
if __name__ == "__main__":
    from src.astro_engine.astronomical_calculator import AstronomicalCalculator
    from src.feature_engineering.pattern_detection import PatternDetectionManager
    from src.feature_engineering.cyclical_features import PlanetaryCycleExtractor
    
    # Initialize calculator
    calculator = AstronomicalCalculator()
    
    # Initialize pattern detector and cycle extractor
    pattern_detector = PatternDetectionManager(calculator)
    cycle_extractor = PlanetaryCycleExtractor(calculator)
    
    # Initialize composite indicator manager
    manager = CompositeIndicatorManager(calculator, pattern_detector, cycle_extractor)
    
    # Calculate indicators for current date
    current_date = datetime.now()
    indicators = manager.calculate_all_indicators(current_date)
    
    print(f"Calculated {len(indicators)} composite indicators for {current_date.strftime('%Y-%m-%d')}:")
    for indicator_name, indicator_value in sorted(indicators.items()):
        print(f"  - {indicator_name}: {indicator_value:.4f}")
    
    # Generate market forecast
    forecast = manager.get_market_forecast(current_date)
    print(f"\nMarket forecast for {forecast['date']}:")
    print(f"  Direction: {forecast['direction']}")
    print(f"  Confidence: {forecast['confidence']:.2f}")
    print(f"  Volatility: {forecast['volatility']:.2f}")
    print(f"  Long-term trend: {forecast['long_term_trend']}")
