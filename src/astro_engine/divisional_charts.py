"""
Divisional Charts (Varga) Module for the Cosmic Market Oracle.

This module implements the 16 divisional charts (Shodasavarga) used in Vedic astrology,
with a focus on financial market applications. These charts provide deeper insights
into different aspects of market behavior and financial outcomes.

The 16 divisional charts are:
1. Rashi (D-1): Overall market trend
2. Hora (D-2): Wealth and financial resources
3. Drekkana (D-3): Siblings and collaborations
4. Chaturthamsha (D-4): Fixed assets and property
5. Panchamamsha (D-5): Authority and power
6. Shashthamsha (D-6): Obstacles and challenges
7. Saptamamsha (D-7): Children and creativity
8. Ashtamamsha (D-8): Obstacles and difficulties
9. Navamsha (D-9): Fortune and long-term trends
10. Dashamsha (D-10): Career and business activities
11. Ekadashamsha (D-11): Gains and income
12. Dwadashamsha (D-12): Parents and ancestry
13. Shodashamsha (D-16): Vehicles and transportation
14. Vimshamsha (D-20): Spiritual practice and wisdom
15. Chaturvimshamsha (D-24): Learning and education
16. Saptavimshamsha (D-27): Strength and weakness
17. Trimshamsha (D-30): Misfortunes and challenges
18. Khavedamsha (D-40): Auspicious and inauspicious effects
19. Akshavedamsha (D-45): All general indications
20. Shashtiamsha (D-60): All effects and detailed analysis
"""

from typing import Dict, List, Optional, Union, Tuple
import math

from .planetary_positions import PlanetaryCalculator, SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU

# Define the signs and their lords
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", 
         "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]

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

# Financial significance of each divisional chart
VARGA_FINANCIAL_SIGNIFICANCE = {
    1: "Overall market trend and general direction",
    2: "Wealth accumulation and financial resources",
    3: "Market collaborations and sector relationships",
    4: "Fixed assets, real estate, and property markets",
    5: "Power dynamics and market leadership",
    6: "Market challenges, debt, and obstacles",
    7: "Speculative investments and creative ventures",
    8: "Hidden market forces and transformations",
    9: "Long-term market fortune and sustainable trends",
    10: "Business activities and corporate performance",
    11: "Profit realization and income generation",
    12: "Legacy investments and institutional influences",
    16: "Mobility sectors and transportation markets",
    20: "Knowledge economy and wisdom-based assets",
    24: "Educational markets and learning technologies",
    27: "Strength and weakness in market segments",
    30: "Market misfortunes and challenging sectors",
    40: "Auspicious and inauspicious market effects",
    45: "General market indications across sectors",
    60: "Detailed market analysis and specific effects"
}


class DivisionalCharts:
    """Class for calculating and analyzing divisional charts (vargas)."""
    
    def __init__(self, calculator: Optional[PlanetaryCalculator] = None):
        """
        Initialize the divisional charts calculator.
        
        Args:
            calculator: Optional PlanetaryCalculator instance
        """
        self.calculator = calculator or PlanetaryCalculator()
        
    def calculate_varga(self, longitude: float, varga: int) -> float:
        """
        Calculate the position in a divisional chart.
        
        Args:
            longitude: Zodiacal longitude in degrees (0-360)
            varga: Division number (1, 2, 3, 4, 7, 9, 10, 12, 16, 20, 24, 27, 30, 40, 45, 60)
            
        Returns:
            Longitude in the divisional chart (0-360)
        """
        if varga == 1:  # Rashi (D-1)
            return longitude
            
        sign = int(longitude / 30)
        degree_in_sign = longitude % 30
        
        if varga == 2:  # Hora (D-2)
            # First half of odd signs and second half of even signs are ruled by Sun
            # First half of even signs and second half of odd signs are ruled by Moon
            if (sign % 2 == 0 and degree_in_sign >= 15) or (sign % 2 == 1 and degree_in_sign < 15):
                return 4 * 30  # Leo (ruled by Sun)
            else:
                return 3 * 30  # Cancer (ruled by Moon)
                
        elif varga == 3:  # Drekkana (D-3)
            # Divide each sign into 3 parts of 10 degrees each
            drekkana = int(degree_in_sign / 10)
            new_sign = (sign * 3 + drekkana) % 12
            return new_sign * 30 + degree_in_sign % 10 * 3
            
        elif varga == 4:  # Chaturthamsha (D-4)
            # Divide each sign into 4 parts of 7.5 degrees each
            chaturthamsha = int(degree_in_sign / 7.5)
            new_sign = (sign * 4 + chaturthamsha) % 12
            return new_sign * 30 + degree_in_sign % 7.5 * 4
            
        elif varga == 5:  # Panchamamsha (D-5)
            # Divide each sign into 5 parts of 6 degrees each
            panchamamsha = int(degree_in_sign / 6)
            if sign % 2 == 0:  # Odd signs
                new_sign = (sign + panchamamsha) % 12
            else:  # Even signs
                new_sign = (sign + 5 - panchamamsha) % 12
            return new_sign * 30 + degree_in_sign % 6 * 5
            
        elif varga == 6:  # Shashthamsha (D-6)
            # Divide each sign into 6 parts of 5 degrees each
            shashthamsha = int(degree_in_sign / 5)
            new_sign = (sign * 6 + shashthamsha) % 12
            return new_sign * 30 + degree_in_sign % 5 * 6
            
        elif varga == 7:  # Saptamamsha (D-7)
            # Divide each sign into 7 parts of ~4.29 degrees each
            saptamamsha = int(degree_in_sign / (30/7))
            if sign % 2 == 0:  # Odd signs
                new_sign = (sign + saptamamsha) % 12
            else:  # Even signs
                new_sign = (sign + 7 - saptamamsha) % 12
            return new_sign * 30 + degree_in_sign % (30/7) * 7
            
        elif varga == 8:  # Ashtamamsha (D-8)
            # Divide each sign into 8 parts of 3.75 degrees each
            ashtamamsha = int(degree_in_sign / 3.75)
            new_sign = (sign * 8 + ashtamamsha) % 12
            return new_sign * 30 + degree_in_sign % 3.75 * 8
            
        elif varga == 9:  # Navamsha (D-9)
            # Divide each sign into 9 parts of 3.33 degrees each
            navamsha = int(degree_in_sign / (30/9))
            # The starting sign depends on the triplicity of the sign
            start_sign = (sign // 4) * 4  # Fire: 0, Earth: 4, Air: 8
            new_sign = (start_sign + navamsha) % 12
            return new_sign * 30 + degree_in_sign % (30/9) * 9
            
        elif varga == 10:  # Dashamsha (D-10)
            # Divide each sign into 10 parts of 3 degrees each
            dashamsha = int(degree_in_sign / 3)
            if sign % 2 == 0:  # Odd signs
                new_sign = (sign + dashamsha) % 12
            else:  # Even signs
                new_sign = (sign + 9 - dashamsha) % 12
            return new_sign * 30 + degree_in_sign % 3 * 10
            
        elif varga == 11:  # Ekadashamsha (D-11)
            # Divide each sign into 11 parts of ~2.73 degrees each
            ekadashamsha = int(degree_in_sign / (30/11))
            new_sign = (sign * 11 + ekadashamsha) % 12
            return new_sign * 30 + degree_in_sign % (30/11) * 11
            
        elif varga == 12:  # Dwadashamsha (D-12)
            # Divide each sign into 12 parts of 2.5 degrees each
            dwadashamsha = int(degree_in_sign / 2.5)
            new_sign = (sign * 12 + dwadashamsha) % 12
            return new_sign * 30 + degree_in_sign % 2.5 * 12
            
        elif varga == 16:  # Shodashamsha (D-16)
            # Divide each sign into 16 parts of 1.875 degrees each
            shodashamsha = int(degree_in_sign / (30/16))
            new_sign = (sign * 16 + shodashamsha) % 12
            return new_sign * 30 + degree_in_sign % (30/16) * 16
            
        elif varga == 20:  # Vimshamsha (D-20)
            # Divide each sign into 20 parts of 1.5 degrees each
            vimshamsha = int(degree_in_sign / 1.5)
            if sign % 2 == 0:  # Odd signs
                new_sign = (sign + vimshamsha) % 12
            else:  # Even signs
                new_sign = (sign + 19 - vimshamsha) % 12
            return new_sign * 30 + degree_in_sign % 1.5 * 20
            
        elif varga == 24:  # Chaturvimshamsha (D-24)
            # Divide each sign into 24 parts of 1.25 degrees each
            chaturvimshamsha = int(degree_in_sign / 1.25)
            new_sign = (sign * 24 + chaturvimshamsha) % 12
            return new_sign * 30 + degree_in_sign % 1.25 * 24
            
        elif varga == 27:  # Saptavimshamsha (D-27)
            # Divide each sign into 27 parts of ~1.11 degrees each
            saptavimshamsha = int(degree_in_sign / (30/27))
            # The starting sign depends on the triplicity of the sign
            start_sign = (sign // 4) * 9  # Fire: 0, Earth: 9, Air: 18
            new_sign = (start_sign + saptavimshamsha) % 12
            return new_sign * 30 + degree_in_sign % (30/27) * 27
            
        elif varga == 30:  # Trimshamsha (D-30)
            # Divide each sign into 5 unequal parts
            if sign % 2 == 0:  # Odd signs
                if degree_in_sign < 5:
                    return 7 * 30  # Mars: 0-5
                elif degree_in_sign < 10:
                    return 10 * 30  # Saturn: 5-10
                elif degree_in_sign < 18:
                    return 8 * 30  # Jupiter: 10-18
                elif degree_in_sign < 25:
                    return 1 * 30  # Venus: 18-25
                else:
                    return 2 * 30  # Mercury: 25-30
            else:  # Even signs
                if degree_in_sign < 5:
                    return 1 * 30  # Venus: 0-5
                elif degree_in_sign < 12:
                    return 2 * 30  # Mercury: 5-12
                elif degree_in_sign < 20:
                    return 8 * 30  # Jupiter: 12-20
                elif degree_in_sign < 25:
                    return 10 * 30  # Saturn: 20-25
                else:
                    return 7 * 30  # Mars: 25-30
                    
        elif varga == 40:  # Khavedamsha (D-40)
            # Divide each sign into 40 parts of 0.75 degrees each
            khavedamsha = int(degree_in_sign / 0.75)
            new_sign = (sign * 40 + khavedamsha) % 12
            return new_sign * 30 + degree_in_sign % 0.75 * 40
            
        elif varga == 45:  # Akshavedamsha (D-45)
            # Divide each sign into 45 parts of 0.67 degrees each
            akshavedamsha = int(degree_in_sign / (30/45))
            if sign % 2 == 0:  # Odd signs
                new_sign = (sign + akshavedamsha) % 12
            else:  # Even signs
                new_sign = (sign + 44 - akshavedamsha) % 12
            return new_sign * 30 + degree_in_sign % (30/45) * 45
            
        elif varga == 60:  # Shashtiamsha (D-60)
            # Divide each sign into 60 parts of 0.5 degrees each
            shashtiamsha = int(degree_in_sign / 0.5)
            new_sign = (sign * 60 + shashtiamsha) % 12
            return new_sign * 30 + degree_in_sign % 0.5 * 60
            
        else:
            raise ValueError(f"Unsupported varga: {varga}")
    
    def calculate_all_vargas(self, longitude: float) -> Dict[int, float]:
        """
        Calculate all standard divisional charts for a given longitude.
        
        Args:
            longitude: Zodiacal longitude in degrees (0-360)
            
        Returns:
            Dictionary mapping varga number to longitude in that varga
        """
        standard_vargas = [1, 2, 3, 4, 7, 9, 10, 12, 16, 20, 24, 27, 30, 40, 45, 60]
        return {varga: self.calculate_varga(longitude, varga) for varga in standard_vargas}
    
    def get_varga_sign_lord(self, varga_longitude: float) -> int:
        """
        Get the lord of the sign in a divisional chart.
        
        Args:
            varga_longitude: Longitude in the divisional chart (0-360)
            
        Returns:
            Planet ID of the sign lord
        """
        sign = int(varga_longitude / 30)
        return SIGN_LORDS[sign]
    
    def calculate_vargottama(self, rashi_longitude: float, navamsa_longitude: float) -> bool:
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
    
    def calculate_planet_vargas(self, planet_id: int, date: Union[str, float]) -> Dict[int, Dict[str, Union[float, int, str]]]:
        """
        Calculate all divisional chart positions for a planet.
        
        Args:
            planet_id: Swiss Ephemeris planet ID
            date: Date as ISO string or Julian Day
            
        Returns:
            Dictionary mapping varga number to varga details
        """
        # Get the planet's position
        position = self.calculator.get_planet_position(planet_id, date)
        longitude = position["longitude"]
        
        # Calculate all vargas
        varga_longitudes = self.calculate_all_vargas(longitude)
        
        # Create detailed result
        result = {}
        for varga, varga_longitude in varga_longitudes.items():
            varga_sign = int(varga_longitude / 30)
            varga_degree = varga_longitude % 30
            varga_lord = self.get_varga_sign_lord(varga_longitude)
            
            result[varga] = {
                "longitude": varga_longitude,
                "sign": varga_sign,
                "sign_name": SIGNS[varga_sign],
                "degree_in_sign": varga_degree,
                "lord": varga_lord,
                "financial_significance": VARGA_FINANCIAL_SIGNIFICANCE.get(varga, "")
            }
            
            # Add vargottama status for D-9
            if varga == 9:
                result[varga]["vargottama"] = self.calculate_vargottama(longitude, varga_longitude)
                
        return result
    
    def calculate_all_planets_vargas(self, date: Union[str, float], include_nodes: bool = True) -> Dict[int, Dict[int, Dict[str, Union[float, int, str]]]]:
        """
        Calculate all divisional chart positions for all planets.
        
        Args:
            date: Date as ISO string or Julian Day
            include_nodes: Whether to include Rahu and Ketu
            
        Returns:
            Dictionary mapping planet ID to varga details
        """
        planets = [SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN]
        if include_nodes:
            planets.extend([RAHU, KETU])
            
        return {planet: self.calculate_planet_vargas(planet, date) for planet in planets}
    
    def get_financial_strength_in_vargas(self, planet_id: int, vargas_data: Dict[int, Dict[str, Union[float, int, str]]]) -> Dict[str, float]:
        """
        Calculate the financial strength of a planet based on its positions in divisional charts.
        
        Args:
            planet_id: Swiss Ephemeris planet ID
            vargas_data: Varga data for the planet
            
        Returns:
            Dictionary with financial strength components
        """
        # Financial importance of each varga
        varga_importance = {
            1: 1.0,   # Rashi (D-1): Overall trend
            2: 0.9,   # Hora (D-2): Wealth
            3: 0.6,   # Drekkana (D-3): Collaborations
            4: 0.7,   # Chaturthamsha (D-4): Fixed assets
            9: 0.95,  # Navamsha (D-9): Fortune
            10: 0.85, # Dashamsha (D-10): Business
            11: 0.8,  # Ekadashamsha (D-11): Gains
            16: 0.5,  # Shodashamsha (D-16): Vehicles
            30: 0.7,  # Trimshamsha (D-30): Misfortunes
            60: 0.75  # Shashtiamsha (D-60): Detailed analysis
        }
        
        # Financial benefic planets
        benefics = [JUPITER, VENUS, MERCURY, MOON]
        
        # Financial malefic planets
        malefics = [SATURN, MARS, RAHU, KETU]
        
        # Calculate strength in each important varga
        varga_strengths = {}
        for varga, importance in varga_importance.items():
            if varga not in vargas_data:
                continue
                
            varga_data = vargas_data[varga]
            sign = varga_data["sign"]
            lord = varga_data["lord"]
            
            # Base strength
            strength = 0.5
            
            # Adjust for dignity
            if planet_id in [SUN, MOON]:
                # Sun and Moon have only one sign
                if sign == 4 and planet_id == SUN:  # Sun in Leo
                    strength = 1.0
                elif sign == 3 and planet_id == MOON:  # Moon in Cancer
                    strength = 1.0
                    
            elif planet_id in [MERCURY, VENUS, MARS, JUPITER, SATURN]:
                # Planets with two signs
                own_signs = {
                    MERCURY: [2, 5],  # Gemini, Virgo
                    VENUS: [1, 6],    # Taurus, Libra
                    MARS: [0, 7],     # Aries, Scorpio
                    JUPITER: [8, 11], # Sagittarius, Pisces
                    SATURN: [9, 10]   # Capricorn, Aquarius
                }
                
                if sign in own_signs.get(planet_id, []):
                    strength = 1.0
                    
            # Adjust for relationship with sign lord
            if planet_id != lord:
                # Friendship chart (0=enemy, 1=neutral, 2=friend)
                from .vedic_dignities import PLANETARY_RELATIONSHIPS
                
                if planet_id in PLANETARY_RELATIONSHIPS and lord in PLANETARY_RELATIONSHIPS[planet_id]:
                    relationship = PLANETARY_RELATIONSHIPS[planet_id][lord]
                    if relationship == 0:  # Enemy
                        strength *= 0.6
                    elif relationship == 2:  # Friend
                        strength *= 1.2
                        strength = min(strength, 1.0)
            
            # Adjust for vargottama (same sign in D-1 and D-9)
            if varga == 9 and varga_data.get("vargottama", False):
                strength *= 1.3
                strength = min(strength, 1.0)
                
            # Store strength weighted by varga importance
            varga_strengths[varga] = strength * importance
        
        # Calculate overall financial strength
        total_importance = sum(varga_importance.values())
        overall_strength = sum(varga_strengths.values()) / total_importance
        
        # Calculate market impact based on planet nature
        if planet_id in benefics:
            market_impact = overall_strength
        elif planet_id in malefics:
            market_impact = 1.0 - overall_strength
        else:
            market_impact = 0.5
            
        # Calculate volatility factor
        if planet_id in [MARS, RAHU, KETU]:
            volatility = 0.7 + (overall_strength * 0.3)
        elif planet_id in [JUPITER, VENUS]:
            volatility = 0.3 * overall_strength
        else:
            volatility = 0.5 * overall_strength
            
        return {
            "overall_strength": overall_strength,
            "market_impact": market_impact,
            "volatility": volatility,
            "varga_strengths": varga_strengths
        }


# Example usage
if __name__ == "__main__":
    calculator = PlanetaryCalculator()
    varga_calculator = DivisionalCharts(calculator)
    
    # Calculate Jupiter's position in all divisional charts for a specific date
    date = "2023-01-01"
    jupiter_vargas = varga_calculator.calculate_planet_vargas(JUPITER, date)
    
    # Print results
    for varga, data in jupiter_vargas.items():
        print(f"D-{varga}: Jupiter at {data['degree_in_sign']:.2f}° {data['sign_name']}")
        
    # Calculate financial strength
    financial_strength = varga_calculator.get_financial_strength_in_vargas(JUPITER, jupiter_vargas)
    print(f"Financial strength: {financial_strength['overall_strength']:.2f}")
    print(f"Market impact: {financial_strength['market_impact']:.2f}")
