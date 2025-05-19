"""
Astrological Aspects Module for the Cosmic Market Oracle.

This module calculates and analyzes planetary aspects according to Vedic astrological principles,
with a focus on financial market applications. It includes specialized calculations for
aspect angles, orbs, and strength factors that are particularly relevant to market prediction.
"""

from typing import Dict, List, Optional, Union, Tuple
import math
from datetime import datetime
from .constants import PlanetaryCalculator, SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU, get_planet_name

# Define standard aspects and their orbs
CONJUNCTION = 0
OPPOSITION = 180
TRINE = 120
SQUARE = 90
SEXTILE = 60
QUINCUNX = 150
SEMI_SEXTILE = 30
QUINTILE = 72
BI_QUINTILE = 144
SEMI_SQUARE = 45
SESQUI_SQUARE = 135

# Aspect names for reference
ASPECT_NAMES = {
    CONJUNCTION: "Conjunction",
    OPPOSITION: "Opposition",
    TRINE: "Trine",
    SQUARE: "Square",
    SEXTILE: "Sextile",
    QUINCUNX: "Quincunx",
    SEMI_SEXTILE: "Semi-Sextile",
    QUINTILE: "Quintile",
    BI_QUINTILE: "Bi-Quintile",
    SEMI_SQUARE: "Semi-Square",
    SESQUI_SQUARE: "Sesqui-Square"
}

# Standard orbs for aspects (in degrees)
DEFAULT_ORBS = {
    CONJUNCTION: 10,
    OPPOSITION: 10,
    TRINE: 8,
    SQUARE: 8,
    SEXTILE: 6,
    QUINCUNX: 5,
    SEMI_SEXTILE: 3,
    QUINTILE: 2,
    BI_QUINTILE: 2,
    SEMI_SQUARE: 3,
    SESQUI_SQUARE: 3
}

# Financial significance of aspects (0-1 scale)
FINANCIAL_SIGNIFICANCE = {
    CONJUNCTION: 1.0,
    OPPOSITION: 0.9,
    TRINE: 0.8,
    SQUARE: 0.85,
    SEXTILE: 0.7,
    QUINCUNX: 0.5,
    SEMI_SEXTILE: 0.4,
    QUINTILE: 0.3,
    BI_QUINTILE: 0.3,
    SEMI_SQUARE: 0.6,
    SESQUI_SQUARE: 0.6
}

# Aspect nature (bullish, bearish, volatile)
ASPECT_NATURE = {
    CONJUNCTION: "variable",  # Depends on planets involved
    OPPOSITION: "bearish",
    TRINE: "bullish",
    SQUARE: "volatile",
    SEXTILE: "bullish",
    QUINCUNX: "uncertain",
    SEMI_SEXTILE: "neutral",
    QUINTILE: "creative",
    BI_QUINTILE: "creative",
    SEMI_SQUARE: "tense",
    SESQUI_SQUARE: "tense"
}

# Planet-specific orbs (some planets have wider or narrower orbs)
PLANET_SPECIFIC_ORBS = {
    SUN: 1.5,      # Sun has wider orb
    MOON: 1.5,     # Moon has wider orb
    MERCURY: 1.0,  # Standard orb
    VENUS: 1.0,    # Standard orb
    MARS: 1.2,     # Slightly wider
    JUPITER: 1.2,  # Slightly wider
    SATURN: 1.2,   # Slightly wider
    RAHU: 1.3,     # Wider orb for nodes
    KETU: 1.3      # Wider orb for nodes
}

# Special Vedic aspects (drishti)
VEDIC_SPECIAL_ASPECTS = {
    MARS: [4, 7, 8],      # Mars aspects 4th, 7th, and 8th houses
    JUPITER: [5, 7, 9],   # Jupiter aspects 5th, 7th, and 9th houses
    SATURN: [3, 7, 10]    # Saturn aspects 3rd, 7th, and 10th houses
}

# Financial market impact of planetary combinations
FINANCIAL_COMBINATIONS = {
    (JUPITER, VENUS): {
        CONJUNCTION: "bullish",
        TRINE: "bullish",
        SEXTILE: "bullish",
        OPPOSITION: "neutral",
        SQUARE: "volatile"
    },
    (SATURN, MARS): {
        CONJUNCTION: "bearish",
        OPPOSITION: "bearish",
        SQUARE: "very bearish",
        TRINE: "neutral",
        SEXTILE: "neutral"
    },
    (SUN, JUPITER): {
        CONJUNCTION: "bullish",
        TRINE: "bullish",
        SQUARE: "volatile",
        OPPOSITION: "uncertain"
    }
    # Add more combinations as needed
}


class AspectCalculator:
    """Calculator for planetary aspects with financial market applications."""
    
    def __init__(self, calculator: Optional[PlanetaryCalculator] = None):
        """
        Initialize the aspect calculator.
        
        Args:
            calculator: Optional PlanetaryCalculator instance
        """
        self.calculator = calculator or PlanetaryCalculator()
        
    def calculate_aspect_angle(self, longitude1: float, longitude2: float) -> float:
        """
        Calculate the aspect angle between two planetary positions.
        
        Args:
            longitude1: Zodiacal longitude of first planet in degrees (0-360)
            longitude2: Zodiacal longitude of second planet in degrees (0-360)
            
        Returns:
            Aspect angle in degrees (0-180)
        """
        # Calculate the absolute difference between longitudes
        angle_diff = abs(longitude1 - longitude2)
        
        # Normalize to 0-180 range (opposition is the maximum possible aspect angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        return angle_diff
    
    def identify_aspect(self, angle: float, orb_factor: float = 1.0) -> Dict[str, Union[str, float, bool]]:
        """
        Identify the aspect type based on the angle.
        
        Args:
            angle: Aspect angle in degrees (0-180)
            orb_factor: Multiplier for standard orbs (default: 1.0)
            
        Returns:
            Dictionary with aspect details or None if no aspect found
        """
        # Check each standard aspect
        for aspect_angle, name in ASPECT_NAMES.items():
            orb = DEFAULT_ORBS.get(aspect_angle, 0) * orb_factor
            
            # Calculate the difference between the actual angle and the ideal aspect angle
            diff = abs(angle - aspect_angle)
            
            # Check if the angle is within the allowed orb
            if diff <= orb:
                # Calculate aspect strength based on closeness to exact aspect
                strength = 1.0 - (diff / orb) if orb > 0 else 1.0
                
                return {
                    "type": aspect_angle,
                    "name": name,
                    "angle": angle,
                    "diff": diff,
                    "orb": orb,
                    "strength": strength,
                    "exact": diff < 0.5  # Consider aspect exact if within 0.5 degrees
                }
                
        # No aspect found
        return None
    
    def calculate_aspect_between_planets(self, planet1_id: int, planet2_id: int, 
                                        positions: Dict[int, Dict[str, Union[float, bool]]],
                                        orb_factor: float = 1.0) -> Dict[str, Union[str, float, bool]]:
        """
        Calculate the aspect between two planets.
        
        Args:
            planet1_id: Swiss Ephemeris ID of first planet
            planet2_id: Swiss Ephemeris ID of second planet
            positions: Dictionary of planetary positions
            orb_factor: Multiplier for standard orbs (default: 1.0)
            
        Returns:
            Dictionary with aspect details or None if no aspect found
        """
        # Skip if either planet is not in the positions dictionary
        if planet1_id not in positions or planet2_id not in positions:
            return None
            
        # Get longitudes
        longitude1 = positions[planet1_id]["longitude"]
        longitude2 = positions[planet2_id]["longitude"]
        
        # Calculate aspect angle
        angle = self.calculate_aspect_angle(longitude1, longitude2)
        
        # Adjust orb factor based on planet-specific orbs
        adjusted_orb_factor = orb_factor
        adjusted_orb_factor *= PLANET_SPECIFIC_ORBS.get(planet1_id, 1.0)
        adjusted_orb_factor *= PLANET_SPECIFIC_ORBS.get(planet2_id, 1.0)
        
        # Identify aspect
        aspect = self.identify_aspect(angle, adjusted_orb_factor)
        
        if aspect:
            # Add planet information
            aspect["planet1"] = planet1_id
            aspect["planet2"] = planet2_id
            
            # Add financial significance
            aspect["financial_significance"] = FINANCIAL_SIGNIFICANCE.get(aspect["type"], 0.0)
            
            # Determine market impact based on planets involved
            planets_key = tuple(sorted([planet1_id, planet2_id]))
            aspect_type = aspect["type"]
            
            if planets_key in FINANCIAL_COMBINATIONS and aspect_type in FINANCIAL_COMBINATIONS[planets_key]:
                aspect["market_impact"] = FINANCIAL_COMBINATIONS[planets_key][aspect_type]
            else:
                # Default to the general aspect nature
                aspect["market_impact"] = ASPECT_NATURE.get(aspect_type, "neutral")
                
            # Calculate overall financial strength
            aspect["financial_strength"] = aspect["strength"] * aspect["financial_significance"]
            
        return aspect
    
    def calculate_all_aspects(self, positions: Dict[int, Dict[str, Union[float, bool]]], 
                             orb_factor: float = 1.0) -> List[Dict[str, Union[str, float, bool]]]:
        """
        Calculate all aspects between planets in the given positions.
        
        Args:
            positions: Dictionary of planetary positions
            orb_factor: Multiplier for standard orbs (default: 1.0)
            
        Returns:
            List of aspect dictionaries
        """
        aspects = []
        planets = list(positions.keys())
        
        # Calculate aspects between all planet pairs
        for i in range(len(planets)):
            for j in range(i + 1, len(planets)):  # Avoid duplicate pairs
                planet1_id = planets[i]
                planet2_id = planets[j]
                
                aspect = self.calculate_aspect_between_planets(planet1_id, planet2_id, positions, orb_factor)
                if aspect:
                    aspects.append(aspect)
                    
        return aspects
    
    def calculate_vedic_special_aspects(self, positions: Dict[int, Dict[str, Union[float, bool]]]) -> List[Dict]:
        """
        Calculate special Vedic aspects (drishti) between planets.
        
        Args:
            positions: Dictionary of planetary positions
            
        Returns:
            List of special aspect dictionaries
        """
        special_aspects = []
        
        # Process planets with special aspects
        for planet_id, house_aspects in VEDIC_SPECIAL_ASPECTS.items():
            if planet_id not in positions:
                continue
                
            # Get the planet's position
            planet_longitude = positions[planet_id]["longitude"]
            planet_house = int(planet_longitude / 30) + 1  # 1-based house number
            
            # Calculate houses aspected by this planet
            for house_offset in house_aspects:
                # Calculate the aspected house number (1-12)
                aspected_house = ((planet_house + house_offset - 1) % 12) + 1
                
                # Calculate the longitude of the aspected house
                aspected_longitude = ((aspected_house - 1) * 30) + 15  # Middle of the house
                
                # Find planets in the aspected house
                for target_id, target_data in positions.items():
                    if target_id == planet_id:
                        continue  # Skip self-aspect
                        
                    target_longitude = target_data["longitude"]
                    target_house = int(target_longitude / 30) + 1
                    
                    if target_house == aspected_house:
                        # Create a special aspect entry
                        special_aspects.append({
                            "type": "vedic_special",
                            "name": f"Vedic {house_offset}th Aspect",
                            "planet": planet_id,
                            "target": target_id,
                            "house_offset": house_offset,
                            "strength": 0.8,  # Standard strength for special aspects
                            "financial_significance": 0.7,
                            "market_impact": "specialized"  # Needs interpretation based on planets
                        })
                        
        return special_aspects
    
    def get_financial_aspect_forecast(self, aspects: List[Dict[str, Union[str, float, bool]]]) -> Dict:
        """
        Generate a financial market forecast based on planetary aspects.
        
        Args:
            aspects: List of aspect dictionaries
            
        Returns:
            Dictionary with financial forecast details
        """
        if not aspects:
            return {
                "trend": "neutral",
                "volatility": "low",
                "confidence": 0.0,
                "description": "No significant planetary aspects found."
            }
            
        # Count aspects by market impact
        impact_counts = {"bullish": 0, "bearish": 0, "volatile": 0, "neutral": 0, "uncertain": 0}
        impact_strengths = {"bullish": 0.0, "bearish": 0.0, "volatile": 0.0, "neutral": 0.0, "uncertain": 0.0}
        
        # Process each aspect
        for aspect in aspects:
            impact = aspect.get("market_impact", "neutral")
            strength = aspect.get("financial_strength", 0.5)
            
            # Handle specialized impacts
            if impact == "specialized":
                # Determine impact based on planets involved
                planet1 = aspect.get("planet1") or aspect.get("planet")
                planet2 = aspect.get("planet2") or aspect.get("target")
                
                if planet1 in [JUPITER, VENUS] or planet2 in [JUPITER, VENUS]:
                    impact = "bullish"
                elif planet1 in [SATURN, MARS] or planet2 in [SATURN, MARS]:
                    impact = "bearish"
                elif planet1 in [RAHU, KETU] or planet2 in [RAHU, KETU]:
                    impact = "volatile"
                else:
                    impact = "neutral"
            
            # Handle "very bearish" or "very bullish"
            if impact == "very bearish":
                impact = "bearish"
                strength *= 1.5
            elif impact == "very bullish":
                impact = "bullish"
                strength *= 1.5
                
            # Update counts and strengths
            if impact in impact_counts:
                impact_counts[impact] += 1
                impact_strengths[impact] += strength
        
        # Determine overall trend
        bullish_strength = impact_strengths["bullish"]
        bearish_strength = impact_strengths["bearish"]
        volatile_strength = impact_strengths["volatile"]
        
        total_strength = sum(impact_strengths.values())
        if total_strength == 0:
            total_strength = 1  # Avoid division by zero
            
        # Calculate trend scores
        trend_score = (bullish_strength - bearish_strength) / total_strength
        volatility_score = volatile_strength / total_strength
        
        # Determine trend
        if trend_score > 0.3:
            trend = "bullish"
        elif trend_score < -0.3:
            trend = "bearish"
        else:
            trend = "neutral"
            
        # Determine volatility
        if volatility_score > 0.5:
            volatility = "high"
        elif volatility_score > 0.3:
            volatility = "moderate"
        else:
            volatility = "low"
            
        # Calculate confidence based on aspect strengths and counts
        confidence = min(0.9, total_strength / (len(aspects) * 0.5))
        
        # Generate description
        significant_aspects = sorted(aspects, key=lambda a: a.get("financial_strength", 0), reverse=True)[:3]
        aspect_descriptions = []
        
        for aspect in significant_aspects:
            planet1 = aspect.get("planet1") or aspect.get("planet")
            planet2 = aspect.get("planet2") or aspect.get("target")
            aspect_type = aspect.get("name", "aspect")
            
            planet1_name = self._get_planet_name(planet1)
            planet2_name = self._get_planet_name(planet2)
            
            aspect_descriptions.append(f"{planet1_name}-{planet2_name} {aspect_type}")
            
        description = f"Market forecast based on planetary aspects indicates a {trend} trend with {volatility} volatility."
        
        if aspect_descriptions:
            description += f" Key aspects: {', '.join(aspect_descriptions)}."
            
        return {
            "trend": trend,
            "volatility": volatility,
            "confidence": confidence,
            "trend_score": trend_score,
            "volatility_score": volatility_score,
            "description": description,
            "significant_aspects": significant_aspects[:3] if significant_aspects else []
        }
    
    # Removed redundant _get_planet_name method - now using centralized get_planet_name from constants.py


def analyze_aspects_for_date(date: Union[str, datetime], calculator: Optional[PlanetaryCalculator] = None) -> Dict:
    """
    Convenience function to analyze all planetary aspects for a specific date.
    
    Args:
        date: Date to analyze as ISO string or datetime
        calculator: Optional PlanetaryCalculator instance
        
    Returns:
        Dictionary with aspect analysis and financial forecast
    """
    # Create calculator if not provided
    if calculator is None:
        calculator = PlanetaryCalculator()
    
    # Create aspect calculator
    aspect_calculator = AspectCalculator(calculator)
    
    # Get planetary positions
    positions = calculator.get_all_planets(date)
    
    # Calculate standard aspects
    standard_aspects = aspect_calculator.calculate_all_aspects(positions)
    
    # Calculate Vedic special aspects
    vedic_aspects = aspect_calculator.calculate_vedic_special_aspects(positions)
    
    # Combine all aspects
    all_aspects = standard_aspects + vedic_aspects
    
    # Generate financial forecast
    forecast = aspect_calculator.get_financial_aspect_forecast(all_aspects)
    
    return {
        "date": date,
        "standard_aspects": standard_aspects,
        "vedic_aspects": vedic_aspects,
        "forecast": forecast
    }


# Example usage
if __name__ == "__main__":
    calculator = PlanetaryCalculator()
    aspect_calculator = AspectCalculator(calculator)
    
    # Get planetary positions for a specific date
    date = "2023-01-01"
    positions = calculator.get_all_planets(date)
    
    # Calculate all aspects
    aspects = aspect_calculator.calculate_all_aspects(positions)
    
    # Print aspects
    print(f"\nPLANETARY ASPECTS FOR {date}:")
    for aspect in aspects:
        planet1_name = get_planet_name(aspect["planet1"])
        planet2_name = get_planet_name(aspect["planet2"])
        print(f"{planet1_name} {aspect['name']} {planet2_name} (Strength: {aspect['strength']:.2f})")
    
    # Calculate Vedic special aspects
    vedic_aspects = aspect_calculator.calculate_vedic_special_aspects(positions)
    
    # Print Vedic aspects
    print(f"\nVEDIC SPECIAL ASPECTS FOR {date}:")
    for aspect in vedic_aspects:
        planet_name = get_planet_name(aspect["planet"])
        target_name = get_planet_name(aspect["target"])
        print(f"{planet_name} {aspect['name']} to {target_name}")
    
    # Generate financial forecast
    all_aspects = aspects + vedic_aspects
    forecast = aspect_calculator.get_financial_aspect_forecast(all_aspects)
    
    # Print forecast
    print(f"\nFINANCIAL FORECAST: {forecast['description']}")
    print(f"Trend: {forecast['trend']} (Score: {forecast['trend_score']:.2f})")
    print(f"Volatility: {forecast['volatility']} (Score: {forecast['volatility_score']:.2f})")
    print(f"Confidence: {forecast['confidence']:.2f}")
    
    # Use the convenience function
    print("\nUsing convenience function:")
    result = analyze_aspects_for_date("2023-01-15")
    print(f"Forecast: {result['forecast']['description']}")
