"""
Dasha Systems Module for the Cosmic Market Oracle.

This module implements various Vedic astrological dasha (planetary period) systems
with a focus on financial market applications. Dashas are used to time market cycles
and predict periods of growth, contraction, and volatility.

The module includes:
1. Vimshottari Dasha (120-year cycle)
2. Yogini Dasha (36-year cycle)
3. Chara Dasha (variable length based on signs)
4. Kalachakra Dasha (cycle based on birth nakshatra)
5. Financial market adaptations of traditional dasha systems
"""

from typing import Dict, List, Optional, Union, Tuple
import math
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from datetime import datetime, timedelta
from .planetary_positions import (
    PlanetaryCalculator, SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU
)


class DashaCalculator:
    """Calculator for Vedic astrological dasha systems."""
    
    def __init__(self, calculator: Optional[PlanetaryCalculator] = None):
        """
        Initialize the dasha calculator.
        
        Args:
            calculator: Optional PlanetaryCalculator instance
        """
        self.calculator = calculator or PlanetaryCalculator()
        
        # Vimshottari Dasha periods (years)
        self.vimshottari_periods = {
            KETU: 7,
            VENUS: 20,
            SUN: 6,
            MOON: 10,
            MARS: 7,
            RAHU: 18,
            JUPITER: 16,
            SATURN: 19,
            MERCURY: 17
        }
        
        # Vimshottari Dasha order
        self.vimshottari_order = [
            KETU, VENUS, SUN, MOON, MARS, RAHU, JUPITER, SATURN, MERCURY
        ]
        
        # Yogini Dasha periods (years)
        self.yogini_periods = {
            1: 1,  # Mangala (Mars)
            2: 2,  # Pingala (Sun)
            3: 3,  # Dhanya (Jupiter)
            4: 4,  # Bhramari (Mercury)
            5: 5,  # Bhadrika (Saturn)
            6: 6,  # Ulka (Venus)
            7: 7,  # Siddha (Moon)
            8: 8   # Sankata (Rahu)
        }
        
        # Yogini Dasha planet mapping
        self.yogini_planets = {
            1: MARS,
            2: SUN,
            3: JUPITER,
            4: MERCURY,
            5: SATURN,
            6: VENUS,
            7: MOON,
            8: RAHU
        }
        
        # Financial significance of planets
        self.financial_significance = {
            SUN: {
                "trend": "neutral",
                "volatility": "moderate",
                "sectors": ["government", "gold", "energy", "leadership"],
                "description": "Represents authority, government policy, and leadership"
            },
            MOON: {
                "trend": "variable",
                "volatility": "high",
                "sectors": ["public sentiment", "real estate", "food", "retail"],
                "description": "Represents public sentiment, liquidity, and consumer behavior"
            },
            MERCURY: {
                "trend": "neutral",
                "volatility": "high",
                "sectors": ["communication", "technology", "media", "trade"],
                "description": "Represents communication, technology, and trading activity"
            },
            VENUS: {
                "trend": "bullish",
                "volatility": "low",
                "sectors": ["luxury", "entertainment", "fashion", "art"],
                "description": "Represents luxury goods, entertainment, and growth sectors"
            },
            MARS: {
                "trend": "bearish",
                "volatility": "very high",
                "sectors": ["military", "construction", "sports", "manufacturing"],
                "description": "Represents aggressive action, competition, and conflict"
            },
            JUPITER: {
                "trend": "bullish",
                "volatility": "low",
                "sectors": ["finance", "education", "legal", "religion"],
                "description": "Represents expansion, optimism, and growth"
            },
            SATURN: {
                "trend": "bearish",
                "volatility": "low",
                "sectors": ["infrastructure", "mining", "agriculture", "old industries"],
                "description": "Represents contraction, discipline, and long-term trends"
            },
            RAHU: {
                "trend": "variable",
                "volatility": "extreme",
                "sectors": ["foreign markets", "innovation", "speculation", "new technologies"],
                "description": "Represents speculation, innovation, and sudden growth"
            },
            KETU: {
                "trend": "bearish",
                "volatility": "extreme",
                "sectors": ["spiritual", "pharmaceuticals", "hidden sectors", "dissolution"],
                "description": "Represents dissolution, spiritual values, and hidden factors"
            }
        }
    
    def calculate_vimshottari_dasha(self, birth_moon_longitude: float, birth_date: Union[str, datetime]) -> Dict:
        """
        Calculate Vimshottari dasha periods from birth date and Moon position.
        
        Args:
            birth_moon_longitude: Longitude of Moon at birth
            birth_date: Birth date as datetime or ISO string
            
        Returns:
            Dictionary with dasha periods and their dates
        """
        # Convert birth_date to datetime if it's a string
        if isinstance(birth_date, str):
            birth_date = datetime.fromisoformat(birth_date)
            
        # Calculate birth nakshatra and progression
        nakshatra = int(birth_moon_longitude / (360 / 27))
        progression = (birth_moon_longitude % (360 / 27)) / (360 / 27)
        
        # Determine starting dasha lord based on nakshatra
        nakshatra_lords = [
            KETU, VENUS, SUN, MOON, MARS, RAHU, JUPITER, SATURN, MERCURY,  # 1-9
            KETU, VENUS, SUN, MOON, MARS, RAHU, JUPITER, SATURN, MERCURY,  # 10-18
            KETU, VENUS, SUN, MOON, MARS, RAHU, JUPITER, SATURN, MERCURY   # 19-27
        ]
        
        starting_lord = nakshatra_lords[nakshatra]
        starting_lord_index = self.vimshottari_order.index(starting_lord)
        
        # Calculate remaining duration of the first dasha
        total_years = self.vimshottari_periods[starting_lord]
        remaining_years = total_years * (1 - progression)
        
        # Calculate start date of the first dasha
        first_dasha_start = birth_date - timedelta(days=progression * total_years * 365.25)
        
        # Generate all dasha periods
        result = {
            "birth_date": birth_date,
            "birth_nakshatra": nakshatra + 1,  # 1-based nakshatra
            "birth_nakshatra_progression": progression,
            "dasha_sequence": []
        }
        
        current_date = first_dasha_start
        
        # Add the first mahadasha (main period)
        first_mahadasha = {
            "planet": starting_lord,
            "start_date": current_date,
            "end_date": current_date + timedelta(days=remaining_years * 365.25),
            "duration_years": remaining_years,
            "antardashas": []  # Sub-periods
        }
        
        # Calculate antardashas (sub-periods) for the first mahadasha
        first_mahadasha["antardashas"] = self._calculate_antardashas(
            starting_lord, first_mahadasha["start_date"], remaining_years)
        
        result["dasha_sequence"].append(first_mahadasha)
        current_date = first_mahadasha["end_date"]
        
        # Calculate the remaining mahadashas
        for i in range(1, 9):
            planet_index = (starting_lord_index + i) % 9
            planet = self.vimshottari_order[planet_index]
            duration = self.vimshottari_periods[planet]
            
            mahadasha = {
                "planet": planet,
                "start_date": current_date,
                "end_date": current_date + timedelta(days=duration * 365.25),
                "duration_years": duration,
                "antardashas": []  # Sub-periods
            }
            
            # Calculate antardashas (sub-periods)
            mahadasha["antardashas"] = self._calculate_antardashas(
                planet, mahadasha["start_date"], duration)
            
            result["dasha_sequence"].append(mahadasha)
            current_date = mahadasha["end_date"]
            
        return result
    
    def _calculate_antardashas(self, mahadasha_lord: int, start_date: datetime, duration_years: float) -> List[Dict]:
        """
        Calculate antardashas (sub-periods) for a mahadasha.
        
        Args:
            mahadasha_lord: Planet ID of the mahadasha lord
            start_date: Start date of the mahadasha
            duration_years: Duration of the mahadasha in years
            
        Returns:
            List of antardashas with their details
        """
        antardashas = []
        mahadasha_days = duration_years * 365.25
        
        # Start with the mahadasha lord itself
        lord_index = self.vimshottari_order.index(mahadasha_lord)
        current_date = start_date
        
        for i in range(9):
            planet_index = (lord_index + i) % 9
            planet = self.vimshottari_order[planet_index]
            
            # Calculate proportion and duration
            proportion = self.vimshottari_periods[planet] / 120
            antardasha_days = mahadasha_days * proportion
            
            antardasha = {
                "planet": planet,
                "start_date": current_date,
                "end_date": current_date + timedelta(days=antardasha_days),
                "duration_days": antardasha_days,
                "pratyantardashas": []  # Sub-sub-periods
            }
            
            # Calculate pratyantardashas (sub-sub-periods)
            antardasha["pratyantardashas"] = self._calculate_pratyantardashas(
                planet, antardasha["start_date"], antardasha_days)
            
            antardashas.append(antardasha)
            current_date = antardasha["end_date"]
            
        return antardashas
    
    def _calculate_pratyantardashas(self, antardasha_lord: int, start_date: datetime, duration_days: float) -> List[Dict]:
        """
        Calculate pratyantardashas (sub-sub-periods) for an antardasha.
        
        Args:
            antardasha_lord: Planet ID of the antardasha lord
            start_date: Start date of the antardasha
            duration_days: Duration of the antardasha in days
            
        Returns:
            List of pratyantardashas with their details
        """
        pratyantardashas = []
        
        # Start with the antardasha lord itself
        lord_index = self.vimshottari_order.index(antardasha_lord)
        current_date = start_date
        
        for i in range(9):
            planet_index = (lord_index + i) % 9
            planet = self.vimshottari_order[planet_index]
            
            # Calculate proportion and duration
            proportion = self.vimshottari_periods[planet] / 120
            pratyantardasha_days = duration_days * proportion
            
            pratyantardasha = {
                "planet": planet,
                "start_date": current_date,
                "end_date": current_date + timedelta(days=pratyantardasha_days),
                "duration_days": pratyantardasha_days
            }
            
            pratyantardashas.append(pratyantardasha)
            current_date = pratyantardasha["end_date"]
            
        return pratyantardashas
    
    def calculate_yogini_dasha(self, birth_moon_longitude: float, birth_date: Union[str, datetime]) -> Dict:
        """
        Calculate Yogini dasha periods from birth date and Moon position.
        
        Args:
            birth_moon_longitude: Longitude of Moon at birth
            birth_date: Birth date as datetime or ISO string
            
        Returns:
            Dictionary with dasha periods and their dates
        """
        # Convert birth_date to datetime if it's a string
        if isinstance(birth_date, str):
            birth_date = datetime.fromisoformat(birth_date)
            
        # Calculate birth nakshatra
        nakshatra = int(birth_moon_longitude / (360 / 27)) + 1  # 1-based nakshatra
        
        # Determine starting yogini based on nakshatra
        yogini_mapping = {
            1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5,
            10: 6, 11: 7, 12: 8, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5,
            18: 6, 19: 7, 20: 8, 21: 1, 22: 2, 23: 3, 24: 4, 25: 5,
            26: 6, 27: 7
        }
        
        starting_yogini = yogini_mapping[nakshatra]
        
        # Generate yogini dasha sequence
        result = {
            "birth_date": birth_date,
            "birth_nakshatra": nakshatra,
            "dasha_sequence": []
        }
        
        current_date = birth_date
        
        # Calculate 36 years of yogini dashas
        for i in range(8):
            yogini = (starting_yogini + i - 1) % 8 + 1
            duration = self.yogini_periods[yogini]
            planet = self.yogini_planets[yogini]
            
            dasha = {
                "yogini": yogini,
                "planet": planet,
                "start_date": current_date,
                "end_date": current_date + timedelta(days=duration * 365.25),
                "duration_years": duration
            }
            
            result["dasha_sequence"].append(dasha)
            current_date = dasha["end_date"]
            
        return result
    
    def find_current_dasha(self, dasha_data: Dict, date: Optional[Union[str, datetime]] = None) -> Dict:
        """
        Find the current dasha, antardasha, and pratyantardasha for a given date.
        
        Args:
            dasha_data: Dasha data as returned by calculate_vimshottari_dasha
            date: Date to find dasha for (default: current date)
            
        Returns:
            Dictionary with current dasha details
        """
        # Use current date if none provided
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = datetime.fromisoformat(date)
            
        result = {
            "date": date,
            "mahadasha": None,
            "antardasha": None,
            "pratyantardasha": None
        }
        
        # Find current mahadasha
        for mahadasha in dasha_data["dasha_sequence"]:
            if mahadasha["start_date"] <= date <= mahadasha["end_date"]:
                result["mahadasha"] = {
                    "planet": mahadasha["planet"],
                    "start_date": mahadasha["start_date"],
                    "end_date": mahadasha["end_date"],
                    "elapsed_percentage": (date - mahadasha["start_date"]).total_seconds() / 
                                         (mahadasha["end_date"] - mahadasha["start_date"]).total_seconds() * 100
                }
                
                # Find current antardasha
                for antardasha in mahadasha["antardashas"]:
                    if antardasha["start_date"] <= date <= antardasha["end_date"]:
                        result["antardasha"] = {
                            "planet": antardasha["planet"],
                            "start_date": antardasha["start_date"],
                            "end_date": antardasha["end_date"],
                            "elapsed_percentage": (date - antardasha["start_date"]).total_seconds() / 
                                                 (antardasha["end_date"] - antardasha["start_date"]).total_seconds() * 100
                        }
                        
                        # Find current pratyantardasha
                        for pratyantardasha in antardasha["pratyantardashas"]:
                            if pratyantardasha["start_date"] <= date <= pratyantardasha["end_date"]:
                                result["pratyantardasha"] = {
                                    "planet": pratyantardasha["planet"],
                                    "start_date": pratyantardasha["start_date"],
                                    "end_date": pratyantardasha["end_date"],
                                    "elapsed_percentage": (date - pratyantardasha["start_date"]).total_seconds() / 
                                                         (pratyantardasha["end_date"] - pratyantardasha["start_date"]).total_seconds() * 100
                                }
                                break
                        break
                break
                
        return result
    
    def get_financial_forecast_from_dasha(self, current_dasha: Dict) -> Dict:
        """
        Generate a financial market forecast based on current dasha periods.
        
        Args:
            current_dasha: Current dasha details as returned by find_current_dasha
            
        Returns:
            Dictionary with financial forecast details
        """
        if not current_dasha["mahadasha"]:
            return {
                "trend": "neutral",
                "volatility": "normal",
                "description": "No valid dasha data available for forecast"
            }
            
        # Get planets for each level
        maha_planet = current_dasha["mahadasha"]["planet"]
        antar_planet = current_dasha["antardasha"]["planet"] if current_dasha["antardasha"] else None
        pratyantar_planet = current_dasha["pratyantardasha"]["planet"] if current_dasha["pratyantardasha"] else None
        
        # Get financial significance for each planet
        maha_sig = self.financial_significance[maha_planet]
        antar_sig = self.financial_significance[antar_planet] if antar_planet else None
        pratyantar_sig = self.financial_significance[pratyantar_planet] if pratyantar_planet else None
        
        # Calculate trend
        trend_values = {
            "bullish": 1,
            "neutral": 0,
            "bearish": -1,
            "variable": 0
        }
        
        # Weighted influence: Mahadasha 50%, Antardasha 30%, Pratyantardasha 20%
        trend_score = trend_values[maha_sig["trend"]] * 0.5
        if antar_sig:
            trend_score += trend_values[antar_sig["trend"]] * 0.3
        if pratyantar_sig:
            trend_score += trend_values[pratyantar_sig["trend"]] * 0.2
            
        if trend_score > 0.3:
            trend = "bullish"
        elif trend_score < -0.3:
            trend = "bearish"
        else:
            trend = "neutral"
            
        # Calculate volatility
        volatility_values = {
            "very high": 4,
            "high": 3,
            "moderate": 2,
            "low": 1,
            "extreme": 5
        }
        
        # Weighted influence
        volatility_score = volatility_values[maha_sig["volatility"]] * 0.5
        if antar_sig:
            volatility_score += volatility_values[antar_sig["volatility"]] * 0.3
        if pratyantar_sig:
            volatility_score += volatility_values[pratyantar_sig["volatility"]] * 0.2
            
        if volatility_score > 4:
            volatility = "extreme"
        elif volatility_score > 3:
            volatility = "very high"
        elif volatility_score > 2:
            volatility = "high"
        elif volatility_score > 1.5:
            volatility = "moderate"
        else:
            volatility = "low"
            
        # Generate description
        description = f"Market forecast based on current dasha periods: {self._get_planet_name(maha_planet)} mahadasha"
        if antar_sig:
            description += f", {self._get_planet_name(antar_planet)} antardasha"
        if pratyantar_sig:
            description += f", {self._get_planet_name(pratyantar_planet)} pratyantardasha"
            
        description += f". Overall trend is {trend} with {volatility} volatility."
        
        # Add sector influences
        sectors = set(maha_sig["sectors"])
        if antar_sig:
            sectors.update(antar_sig["sectors"][:2])  # Add top 2 sectors from antardasha
        
        description += f" Key sectors influenced: {', '.join(sectors)}."
        
        return {
            "trend": trend,
            "volatility": volatility,
            "description": description,
            "mahadasha_influence": maha_sig,
            "antardasha_influence": antar_sig,
            "pratyantardasha_influence": pratyantar_sig
        }
    
    def _get_planet_name(self, planet_id: int) -> str:
        """Get the name of a planet from its ID."""
        planet_names = {
            SUN: "Sun",
            MOON: "Moon",
            MERCURY: "Mercury",
            VENUS: "Venus",
            MARS: "Mars",
            JUPITER: "Jupiter",
            SATURN: "Saturn",
            RAHU: "Rahu",
            KETU: "Ketu"
        }
        return planet_names.get(planet_id, f"Planet {planet_id}")


# Example usage
if __name__ == "__main__":
    calculator = PlanetaryCalculator()
    dasha_calculator = DashaCalculator(calculator)
    
    # Calculate Moon position for a birth date
    birth_date = "1980-01-01"
    moon_position = calculator.get_planet_position(MOON, birth_date)
    moon_longitude = moon_position["longitude"]
    
    # Calculate Vimshottari dasha
    vimshottari_dasha = dasha_calculator.calculate_vimshottari_dasha(moon_longitude, birth_date)
    
    # Find current dasha
    current_date = "2023-01-01"
    current_dasha = dasha_calculator.find_current_dasha(vimshottari_dasha, current_date)
    
    # Print current dasha
    print(f"Current Mahadasha: {dasha_calculator._get_planet_name(current_dasha['mahadasha']['planet'])}")
    print(f"Current Antardasha: {dasha_calculator._get_planet_name(current_dasha['antardasha']['planet'])}")
    
    # Get financial forecast
    forecast = dasha_calculator.get_financial_forecast_from_dasha(current_dasha)
    print(f"\nFINANCIAL FORECAST: {forecast['description']}")
    print(f"Trend: {forecast['trend']}")
    print(f"Volatility: {forecast['volatility']}")
