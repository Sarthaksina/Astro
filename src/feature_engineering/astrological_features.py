# Cosmic Market Oracle - Astrological Features Module

"""
This module transforms raw astrological data into machine learning-compatible features.
It includes specialized transformations for cyclical data, planetary relationships,
and other astrological phenomena relevant to financial markets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from ..astro_engine.planetary_positions import PlanetaryCalculator, SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO, RAHU, KETU


class AstrologicalFeatureGenerator:
    """Generates machine learning features from astrological data."""
    
    def __init__(self, calculator: Optional[PlanetaryCalculator] = None):
        """
        Initialize the feature generator.
        
        Args:
            calculator: Optional PlanetaryCalculator instance. If None, a new one will be created.
        """
        self.calculator = calculator or PlanetaryCalculator()
        
        # Planet pairs for aspect and relationship calculations
        self.major_planets = [SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN]
        self.outer_planets = [URANUS, NEPTUNE, PLUTO]
        self.nodes = [RAHU, KETU]
        self.all_planets = self.major_planets + self.outer_planets + self.nodes
        
        # Define aspect angles (in degrees) and their orbs (allowable deviation)
        self.aspects = {
            'conjunction': {'angle': 0, 'orb': 8},
            'opposition': {'angle': 180, 'orb': 8},
            'trine': {'angle': 120, 'orb': 8},
            'square': {'angle': 90, 'orb': 7},
            'sextile': {'angle': 60, 'orb': 6},
            'quincunx': {'angle': 150, 'orb': 5},
            'semi-square': {'angle': 45, 'orb': 4},
            'sesquiquadrate': {'angle': 135, 'orb': 4},
            'semi-sextile': {'angle': 30, 'orb': 3}
        }
        
    def encode_cyclical_feature(self, values: np.ndarray, period: float) -> np.ndarray:
        """
        Encode a cyclical feature (like degrees) using sine and cosine transformations.
        
        Args:
            values: Array of values to encode
            period: The period of the cycle (e.g., 360 for degrees)
            
        Returns:
            Array with two columns: sin and cos transformations
        """
        values = np.asarray(values)
        sin_values = np.sin(2 * np.pi * values / period)
        cos_values = np.cos(2 * np.pi * values / period)
        return np.column_stack((sin_values, cos_values))
    
    def calculate_aspect_strength(self, angle1: float, angle2: float, aspect_type: str) -> float:
        """
        Calculate the strength of an aspect between two planets.
        
        Args:
            angle1: Longitude of first planet (degrees)
            angle2: Longitude of second planet (degrees)
            aspect_type: Type of aspect to check
            
        Returns:
            Strength of aspect (0-1, where 1 is exact)
        """
        aspect = self.aspects[aspect_type]
        target_angle = aspect['angle']
        max_orb = aspect['orb']
        
        # Calculate the angular difference
        diff = abs((angle1 - angle2) % 360)
        if diff > 180:
            diff = 360 - diff
            
        # Calculate the difference from the target aspect angle
        aspect_diff = abs(diff - target_angle)
        
        # If within orb, calculate strength (1 = exact aspect, 0 = at maximum orb)
        if aspect_diff <= max_orb:
            return 1 - (aspect_diff / max_orb)
        else:
            return 0.0
    
    def generate_planet_features(self, date, heliocentric: bool = False) -> Dict[str, float]:
        """
        Generate basic planetary position features for a given date.
        
        Args:
            date: The date to calculate features for
            heliocentric: Whether to use heliocentric positions
            
        Returns:
            Dictionary of planetary features
        """
        planets_data = self.calculator.get_all_planets(date, include_nodes=True, heliocentric=heliocentric)
        features = {}
        
        # Process each planet's position
        for planet_id, position in planets_data.items():
            planet_name = self._get_planet_name(planet_id)
            
            # Store raw longitude
            features[f"{planet_name}_longitude"] = position["longitude"]
            
            # Encode cyclical longitude
            sin_lon, cos_lon = self.encode_cyclical_feature([position["longitude"]], 360)[0]
            features[f"{planet_name}_longitude_sin"] = sin_lon
            features[f"{planet_name}_longitude_cos"] = cos_lon
            
            # Store other position data
            features[f"{planet_name}_latitude"] = position["latitude"]
            features[f"{planet_name}_speed"] = position["longitude_speed"]
            features[f"{planet_name}_is_retrograde"] = 1.0 if position["is_retrograde"] else 0.0
            features[f"{planet_name}_nakshatra"] = position["nakshatra"]
            
            # Encode cyclical nakshatra position
            sin_nak, cos_nak = self.encode_cyclical_feature([position["nakshatra_degree"]], 13.333)[0]
            features[f"{planet_name}_nakshatra_sin"] = sin_nak
            features[f"{planet_name}_nakshatra_cos"] = cos_nak
            
        return features
    
    def generate_aspect_features(self, date) -> Dict[str, float]:
        """
        Generate features representing planetary aspects.
        
        Args:
            date: The date to calculate features for
            
        Returns:
            Dictionary of aspect features
        """
        planets_data = self.calculator.get_all_planets(date, include_nodes=True)
        features = {}
        
        # Calculate aspects between planet pairs
        for i, planet1_id in enumerate(self.all_planets):
            for j, planet2_id in enumerate(self.all_planets[i+1:], i+1):
                planet1_name = self._get_planet_name(planet1_id)
                planet2_name = self._get_planet_name(planet2_id)
                
                angle1 = planets_data[planet1_id]["longitude"]
                angle2 = planets_data[planet2_id]["longitude"]
                
                # Calculate strength of each aspect type
                for aspect_type in self.aspects.keys():
                    strength = self.calculate_aspect_strength(angle1, angle2, aspect_type)
                    if strength > 0:
                        features[f"{planet1_name}_{planet2_name}_{aspect_type}"] = strength
        
        return features
    
    def generate_special_features(self, date) -> Dict[str, float]:
        """
        Generate special astrological features like yogas and sensitive points.
        
        Args:
            date: The date to calculate features for
            
        Returns:
            Dictionary of special features
        """
        planets_data = self.calculator.get_all_planets(date, include_nodes=True)
        features = {}
        
        # Calculate Moon phase (0 = New Moon, 0.5 = Full Moon, 1 = New Moon)
        sun_lon = planets_data[SUN]["longitude"]
        moon_lon = planets_data[MOON]["longitude"]
        moon_phase = ((moon_lon - sun_lon) % 360) / 360
        features["moon_phase"] = moon_phase
        
        # Encode cyclical moon phase
        sin_phase, cos_phase = self.encode_cyclical_feature([moon_phase * 360], 360)[0]
        features["moon_phase_sin"] = sin_phase
        features["moon_phase_cos"] = cos_phase
        
        # Calculate Jupiter-Saturn cycle (0-1 representing position in 20-year cycle)
        jup_lon = planets_data[JUPITER]["longitude"]
        sat_lon = planets_data[SATURN]["longitude"]
        js_angle = (jup_lon - sat_lon) % 360
        js_cycle = js_angle / 360
        features["jupiter_saturn_cycle"] = js_cycle
        
        # Encode cyclical Jupiter-Saturn cycle
        sin_js, cos_js = self.encode_cyclical_feature([js_cycle * 360], 360)[0]
        features["jupiter_saturn_cycle_sin"] = sin_js
        features["jupiter_saturn_cycle_cos"] = cos_js
        
        # Calculate number of retrograde planets
        retrograde_count = sum(1 for planet_id in self.major_planets 
                              if planets_data[planet_id]["is_retrograde"])
        features["retrograde_count"] = retrograde_count
        
        # Calculate element and modality distributions
        elements = self._calculate_element_distribution(planets_data)
        for element, value in elements.items():
            features[f"element_{element}"] = value
            
        modalities = self._calculate_modality_distribution(planets_data)
        for modality, value in modalities.items():
            features[f"modality_{modality}"] = value
        
        return features
    
    def generate_all_features(self, date) -> Dict[str, float]:
        """
        Generate all astrological features for a given date.
        
        Args:
            date: The date to calculate features for
            
        Returns:
            Dictionary of all astrological features
        """
        features = {}
        
        # Combine all feature types
        features.update(self.generate_planet_features(date, heliocentric=False))
        features.update(self.generate_planet_features(date, heliocentric=True))
        features.update(self.generate_aspect_features(date))
        features.update(self.generate_special_features(date))
        
        return features
    
    def generate_features_for_dates(self, dates) -> pd.DataFrame:
        """
        Generate features for a list of dates.
        
        Args:
            dates: List of dates to generate features for
            
        Returns:
            DataFrame with dates as index and features as columns
        """
        feature_dicts = []
        for date in dates:
            features = self.generate_all_features(date)
            feature_dicts.append(features)
            
        return pd.DataFrame(feature_dicts, index=dates)
    
    def _get_planet_name(self, planet_id: int) -> str:
        """
        Get the name of a planet from its ID.
        
        Args:
            planet_id: Swiss Ephemeris planet ID
            
        Returns:
            Planet name as string
        """
        planet_names = {
            SUN: "sun",
            MOON: "moon",
            MERCURY: "mercury",
            VENUS: "venus",
            MARS: "mars",
            JUPITER: "jupiter",
            SATURN: "saturn",
            URANUS: "uranus",
            NEPTUNE: "neptune",
            PLUTO: "pluto",
            RAHU: "rahu",
            KETU: "ketu"
        }
        return planet_names.get(planet_id, f"planet_{planet_id}")
    
    def _calculate_element_distribution(self, planets_data: Dict) -> Dict[str, float]:
        """
        Calculate the distribution of planets across elements (fire, earth, air, water).
        
        Args:
            planets_data: Dictionary of planetary positions
            
        Returns:
            Dictionary with element distributions
        """
        # Define zodiac sign elements (0=Aries, 1=Taurus, etc.)
        sign_elements = {
            0: "fire", 4: "fire", 8: "fire",  # Aries, Leo, Sagittarius
            1: "earth", 5: "earth", 9: "earth",  # Taurus, Virgo, Capricorn
            2: "air", 6: "air", 10: "air",  # Gemini, Libra, Aquarius
            3: "water", 7: "water", 11: "water"  # Cancer, Scorpio, Pisces
        }
        
        elements = {"fire": 0, "earth": 0, "air": 0, "water": 0}
        
        # Count planets in each element
        for planet_id in self.major_planets:
            longitude = planets_data[planet_id]["longitude"]
            sign = int(longitude / 30)  # Each sign is 30 degrees
            element = sign_elements[sign]
            elements[element] += 1
            
        # Normalize by number of planets
        total = sum(elements.values())
        for element in elements:
            elements[element] /= total
            
        return elements
    
    def _calculate_modality_distribution(self, planets_data: Dict) -> Dict[str, float]:
        """
        Calculate the distribution of planets across modalities (cardinal, fixed, mutable).
        
        Args:
            planets_data: Dictionary of planetary positions
            
        Returns:
            Dictionary with modality distributions
        """
        # Define zodiac sign modalities (0=Aries, 1=Taurus, etc.)
        sign_modalities = {
            0: "cardinal", 3: "cardinal", 6: "cardinal", 9: "cardinal",  # Aries, Cancer, Libra, Capricorn
            1: "fixed", 4: "fixed", 7: "fixed", 10: "fixed",  # Taurus, Leo, Scorpio, Aquarius
            2: "mutable", 5: "mutable", 8: "mutable", 11: "mutable"  # Gemini, Virgo, Sagittarius, Pisces
        }
        
        modalities = {"cardinal": 0, "fixed": 0, "mutable": 0}
        
        # Count planets in each modality
        for planet_id in self.major_planets:
            longitude = planets_data[planet_id]["longitude"]
            sign = int(longitude / 30)  # Each sign is 30 degrees
            modality = sign_modalities[sign]
            modalities[modality] += 1
            
        # Normalize by number of planets
        total = sum(modalities.values())
        for modality in modalities:
            modalities[modality] /= total
            
        return modalities