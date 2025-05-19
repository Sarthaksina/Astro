"""
Financial Yogas Module for the Cosmic Market Oracle.

This module identifies and analyzes Vedic astrological yogas (planetary combinations)
that have significant implications for financial markets and economic trends.

These specialized yogas are derived from traditional Vedic astrology but adapted
specifically for financial market analysis and prediction.
"""

from typing import Dict, List, Optional, Union, Tuple
import math
from .planetary_positions import PlanetaryCalculator
from .constants import (
    SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU,
    get_planet_name
)
from .vedic_dignities import calculate_dignity_state, check_combustion


class FinancialYogaAnalyzer:
    """Analyzer for financial yogas (planetary combinations)."""
    
    def __init__(self, calculator: Optional[PlanetaryCalculator] = None):
        """
        Initialize the financial yoga analyzer.
        
        Args:
            calculator: Optional PlanetaryCalculator instance
        """
        self.calculator = calculator or PlanetaryCalculator()
        
    def analyze_dhana_yogas(self, positions: Dict[int, Dict[str, Union[float, bool]]]) -> List[Dict]:
        """
        Analyze Dhana Yogas (wealth combinations).
        
        Args:
            positions: Dictionary of planetary positions
            
        Returns:
            List of identified Dhana Yogas with details
        """
        yogas = []
        
        # 1. Lakshmi Yoga - Jupiter in own sign or exalted and Venus well-placed
        if JUPITER in positions and VENUS in positions:
            jupiter_longitude = positions[JUPITER]["longitude"]
            venus_longitude = positions[VENUS]["longitude"]
            
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
                    "market_impact": "bullish",
                    "planets_involved": [JUPITER, VENUS]
                })
        
        # 2. Wealth-producing conjunction of benefics
        benefics = [JUPITER, VENUS, MERCURY, MOON]
        benefic_positions = {planet: positions[planet] for planet in benefics if planet in positions}
        
        if len(benefic_positions) >= 2:
            for planet1 in benefic_positions:
                for planet2 in benefic_positions:
                    if planet1 >= planet2:
                        continue
                        
                    # Check if planets are conjunct (within 5 degrees)
                    lon1 = benefic_positions[planet1]["longitude"]
                    lon2 = benefic_positions[planet2]["longitude"]
                    
                    angle_diff = abs(lon1 - lon2)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                        
                    if angle_diff <= 5:
                        yogas.append({
                            "name": "Benefic Conjunction",
                            "strength": 70,
                            "description": f"Conjunction of {get_planet_name(planet1)} and {get_planet_name(planet2)} indicates financial growth",
                            "market_impact": "bullish",
                            "planets_involved": [planet1, planet2]
                        })
        
        # 3. Gajakesari Yoga - Jupiter and Moon in angular houses (simplified)
        if JUPITER in positions and MOON in positions:
            jupiter_longitude = positions[JUPITER]["longitude"]
            moon_longitude = positions[MOON]["longitude"]
            
            # Check if Jupiter and Moon are in Kendra (square aspect, 90 degrees)
            angle_diff = abs(jupiter_longitude - moon_longitude)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
                
            if 85 <= angle_diff <= 95 or angle_diff <= 5 or 175 <= angle_diff <= 185:
                yogas.append({
                    "name": "Gajakesari Yoga",
                    "strength": 75,
                    "description": "Jupiter and Moon in angular relationship indicates expansion and growth",
                    "market_impact": "bullish",
                    "planets_involved": [JUPITER, MOON]
                })
                
        return yogas
    
    def analyze_raja_yogas(self, positions: Dict[int, Dict[str, Union[float, bool]]]) -> List[Dict]:
        """
        Analyze Raja Yogas (power combinations) with financial implications.
        
        Args:
            positions: Dictionary of planetary positions
            
        Returns:
            List of identified Raja Yogas with details
        """
        yogas = []
        
        # 1. Viparita Raja Yoga - Malefics in 6th, 8th, or 12th houses (simplified)
        malefics = [SATURN, MARS, RAHU]
        malefic_positions = {planet: positions[planet] for planet in malefics if planet in positions}
        
        if len(malefic_positions) >= 2:
            for planet1 in malefic_positions:
                for planet2 in malefic_positions:
                    if planet1 >= planet2:
                        continue
                        
                    # Check if malefics are in opposition (180 degrees)
                    lon1 = malefic_positions[planet1]["longitude"]
                    lon2 = malefic_positions[planet2]["longitude"]
                    
                    angle_diff = abs(lon1 - lon2)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                        
                    if 175 <= angle_diff <= 185:  # Within 5 degrees of opposition
                        yogas.append({
                            "name": "Viparita Raja Yoga",
                            "strength": 75,
                            "description": f"Opposition of {self._get_planet_name(planet1)} and {self._get_planet_name(planet2)} indicates unexpected market reversals",
                            "market_impact": "volatile",
                            "planets_involved": [planet1, planet2]
                        })
        
        # 2. Neecha Bhanga Raja Yoga - Debilitated planet getting cancellation
        for planet in positions:
            if planet in [SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN]:
                longitude = positions[planet]["longitude"]
                dignity = calculate_dignity_state(planet, longitude)
                
                if dignity["state"] == "debilitated":
                    # Check if lord of the sign is strong
                    sign = int(longitude / 30)
                    sign_lord = self._get_sign_lord(sign)
                    
                    if sign_lord in positions:
                        lord_longitude = positions[sign_lord]["longitude"]
                        lord_dignity = calculate_dignity_state(sign_lord, lord_longitude)
                        
                        if lord_dignity["state"] in ["exalted", "own_sign", "moolatrikona"]:
                            yogas.append({
                                "name": "Neecha Bhanga Raja Yoga",
                                "strength": 65,
                                "description": f"Debilitated {self._get_planet_name(planet)} with strong {self._get_planet_name(sign_lord)} indicates recovery from market lows",
                                "market_impact": "bullish",
                                "planets_involved": [planet, sign_lord]
                            })
                            
        return yogas
    
    def analyze_market_trend_yogas(self, positions: Dict[int, Dict[str, Union[float, bool]]]) -> List[Dict]:
        """
        Analyze yogas specifically related to market trends.
        
        Args:
            positions: Dictionary of planetary positions
            
        Returns:
            List of identified market trend yogas with details
        """
        yogas = []
        
        # 1. Bullish Alignment - Jupiter, Venus, and Mercury well-placed
        if all(planet in positions for planet in [JUPITER, VENUS, MERCURY]):
            jupiter_retrograde = positions[JUPITER].get("is_retrograde", False)
            venus_retrograde = positions[VENUS].get("is_retrograde", False)
            mercury_retrograde = positions[MERCURY].get("is_retrograde", False)
            
            # Check if none are retrograde and not combust
            if not (jupiter_retrograde or venus_retrograde or mercury_retrograde):
                jupiter_combust = check_combustion(JUPITER, positions[JUPITER]["longitude"], 
                                                 positions[SUN]["longitude"]) if SUN in positions else {"is_combust": False}
                venus_combust = check_combustion(VENUS, positions[VENUS]["longitude"], 
                                               positions[SUN]["longitude"]) if SUN in positions else {"is_combust": False}
                mercury_combust = check_combustion(MERCURY, positions[MERCURY]["longitude"], 
                                                 positions[SUN]["longitude"]) if SUN in positions else {"is_combust": False}
                
                if not (jupiter_combust["is_combust"] or venus_combust["is_combust"] or mercury_combust["is_combust"]):
                    yogas.append({
                        "name": "Bullish Financial Alignment",
                        "strength": 80,
                        "description": "Jupiter, Venus, and Mercury all strong and direct, indicating bullish market conditions",
                        "market_impact": "bullish",
                        "planets_involved": [JUPITER, VENUS, MERCURY]
                    })
        
        # 2. Bearish Alignment - Saturn, Mars, and Rahu in key positions
        if all(planet in positions for planet in [SATURN, MARS, RAHU]):
            saturn_longitude = positions[SATURN]["longitude"]
            mars_longitude = positions[MARS]["longitude"]
            rahu_longitude = positions[RAHU]["longitude"]
            
            # Check if Saturn and Mars are in square aspect (90 degrees)
            saturn_mars_angle = abs(saturn_longitude - mars_longitude)
            if saturn_mars_angle > 180:
                saturn_mars_angle = 360 - saturn_mars_angle
                
            # Check if Rahu is conjunct with either Saturn or Mars
            rahu_saturn_angle = abs(rahu_longitude - saturn_longitude)
            if rahu_saturn_angle > 180:
                rahu_saturn_angle = 360 - rahu_saturn_angle
                
            rahu_mars_angle = abs(rahu_longitude - mars_longitude)
            if rahu_mars_angle > 180:
                rahu_mars_angle = 360 - rahu_mars_angle
                
            if (85 <= saturn_mars_angle <= 95) and (rahu_saturn_angle <= 10 or rahu_mars_angle <= 10):
                yogas.append({
                    "name": "Bearish Financial Alignment",
                    "strength": 75,
                    "description": "Saturn, Mars, and Rahu in challenging alignment, indicating bearish market conditions",
                    "market_impact": "bearish",
                    "planets_involved": [SATURN, MARS, RAHU]
                })
        
        # 3. Volatility Yoga - Mercury, Rahu, and Mars in close aspect
        if all(planet in positions for planet in [MERCURY, RAHU, MARS]):
            mercury_longitude = positions[MERCURY]["longitude"]
            rahu_longitude = positions[RAHU]["longitude"]
            mars_longitude = positions[MARS]["longitude"]
            
            # Check aspects between these planets
            mercury_rahu_angle = abs(mercury_longitude - rahu_longitude)
            if mercury_rahu_angle > 180:
                mercury_rahu_angle = 360 - mercury_rahu_angle
                
            mercury_mars_angle = abs(mercury_longitude - mars_longitude)
            if mercury_mars_angle > 180:
                mercury_mars_angle = 360 - mercury_mars_angle
                
            if mercury_rahu_angle <= 10 or mercury_mars_angle <= 10:
                yogas.append({
                    "name": "Market Volatility Yoga",
                    "strength": 70,
                    "description": "Mercury in conjunction with Rahu or Mars, indicating increased market volatility",
                    "market_impact": "volatile",
                    "planets_involved": [MERCURY, RAHU if mercury_rahu_angle <= 10 else MARS]
                })
                
        return yogas
    
    def analyze_all_financial_yogas(self, positions: Dict[int, Dict[str, Union[float, bool]]]) -> Dict[str, List[Dict]]:
        """
        Analyze all financial yogas in the given planetary positions.
        
        Args:
            positions: Dictionary of planetary positions
            
        Returns:
            Dictionary mapping yoga categories to lists of identified yogas
        """
        dhana_yogas = self.analyze_dhana_yogas(positions)
        raja_yogas = self.analyze_raja_yogas(positions)
        trend_yogas = self.analyze_market_trend_yogas(positions)
        
        return {
            "dhana_yogas": dhana_yogas,
            "raja_yogas": raja_yogas,
            "trend_yogas": trend_yogas
        }
    
    def get_market_forecast(self, yogas: Dict[str, List[Dict]]) -> Dict[str, Union[str, float]]:
        """
        Generate a market forecast based on identified yogas.
        
        Args:
            yogas: Dictionary of identified yogas by category
            
        Returns:
            Dictionary with market forecast details
        """
        # Flatten all yogas
        all_yogas = []
        for category, yoga_list in yogas.items():
            all_yogas.extend(yoga_list)
            
        if not all_yogas:
            return {
                "trend": "neutral",
                "confidence": 0.5,
                "volatility": "normal",
                "description": "No significant yogas detected, indicating neutral market conditions"
            }
            
        # Calculate overall trend
        bullish_strength = 0
        bearish_strength = 0
        volatile_strength = 0
        total_strength = 0
        
        for yoga in all_yogas:
            strength = yoga["strength"] / 100.0  # Convert to 0-1 scale
            
            if yoga["market_impact"] == "bullish":
                bullish_strength += strength
            elif yoga["market_impact"] == "bearish":
                bearish_strength += strength
            elif yoga["market_impact"] == "volatile":
                volatile_strength += strength
                
            total_strength += strength
            
        # Normalize strengths
        if total_strength > 0:
            bullish_factor = bullish_strength / total_strength
            bearish_factor = bearish_strength / total_strength
            volatile_factor = volatile_strength / total_strength
        else:
            bullish_factor = bearish_factor = volatile_factor = 0
            
        # Determine trend
        if bullish_factor > bearish_factor + 0.2:
            trend = "bullish"
            confidence = bullish_factor
        elif bearish_factor > bullish_factor + 0.2:
            trend = "bearish"
            confidence = bearish_factor
        else:
            trend = "neutral"
            confidence = 0.5
            
        # Determine volatility
        if volatile_factor > 0.3:
            volatility = "high"
        elif volatile_factor > 0.15:
            volatility = "moderate"
        else:
            volatility = "low"
            
        # Generate description
        if trend == "bullish":
            description = "Financial yogas indicate a bullish market trend"
        elif trend == "bearish":
            description = "Financial yogas indicate a bearish market trend"
        else:
            description = "Financial yogas indicate a neutral market trend"
            
        if volatility == "high":
            description += " with high volatility"
        elif volatility == "moderate":
            description += " with moderate volatility"
            
        # Add top yoga
        if all_yogas:
            top_yoga = max(all_yogas, key=lambda y: y["strength"])
            description += f". The strongest influence is {top_yoga['name']}: {top_yoga['description']}"
            
        return {
            "trend": trend,
            "confidence": confidence,
            "volatility": volatility,
            "description": description
        }
    
    # _get_planet_name method removed - now using centralized get_planet_name from constants.py
    
    def _get_sign_lord(self, sign: int) -> int:
        """Get the lord of a sign."""
        sign_lords = {
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
        return sign_lords.get(sign, SUN)


# Example usage
if __name__ == "__main__":
    calculator = PlanetaryCalculator()
    yoga_analyzer = FinancialYogaAnalyzer(calculator)
    
    # Get planetary positions for a specific date
    date = "2023-01-01"
    positions = calculator.get_all_planets(date)
    
    # Analyze financial yogas
    yogas = yoga_analyzer.analyze_all_financial_yogas(positions)
    
    # Print results
    for category, yoga_list in yogas.items():
        print(f"\n{category.upper()}:")
        for yoga in yoga_list:
            print(f"- {yoga['name']}: {yoga['description']} (Strength: {yoga['strength']})")
    
    # Get market forecast
    forecast = yoga_analyzer.get_market_forecast(yogas)
    print(f"\nMARKET FORECAST: {forecast['description']}")
    print(f"Trend: {forecast['trend']} (Confidence: {forecast['confidence']:.2f})")
    print(f"Volatility: {forecast['volatility']}")
