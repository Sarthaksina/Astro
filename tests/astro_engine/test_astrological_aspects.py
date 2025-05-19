"""
Unit tests for the astrological_aspects module.

This module tests the functionality of the AspectCalculator class and related functions
for calculating and analyzing planetary aspects in the context of financial markets.
"""

import unittest
import datetime
from src.astro_engine.planetary_positions import PlanetaryCalculator, SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN
from src.astro_engine.astrological_aspects import (
    AspectCalculator, analyze_aspects_for_date,
    CONJUNCTION, OPPOSITION, TRINE, SQUARE, SEXTILE
)


class TestAspectCalculator(unittest.TestCase):
    """Test cases for the AspectCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = PlanetaryCalculator()
        self.aspect_calculator = AspectCalculator(self.calculator)
        
        # Sample planetary positions for testing
        self.test_positions = {
            SUN: {"longitude": 10.0, "latitude": 0.0, "distance": 1.0, "speed": 1.0, "is_retrograde": False},
            MOON: {"longitude": 100.0, "latitude": 0.0, "distance": 1.0, "speed": 13.0, "is_retrograde": False},
            MERCURY: {"longitude": 10.5, "latitude": 0.0, "distance": 1.0, "speed": 1.2, "is_retrograde": False},
            VENUS: {"longitude": 70.0, "latitude": 0.0, "distance": 1.0, "speed": 1.0, "is_retrograde": False},
            JUPITER: {"longitude": 130.0, "latitude": 0.0, "distance": 5.0, "speed": 0.1, "is_retrograde": False},
            SATURN: {"longitude": 280.0, "latitude": 0.0, "distance": 10.0, "speed": 0.05, "is_retrograde": True}
        }
    
    def test_calculate_aspect_angle(self):
        """Test calculation of aspect angles."""
        # Test exact angles
        self.assertEqual(self.aspect_calculator.calculate_aspect_angle(0, 0), 0)  # Conjunction
        self.assertEqual(self.aspect_calculator.calculate_aspect_angle(0, 180), 180)  # Opposition
        self.assertEqual(self.aspect_calculator.calculate_aspect_angle(0, 120), 120)  # Trine
        self.assertEqual(self.aspect_calculator.calculate_aspect_angle(0, 90), 90)  # Square
        
        # Test normalization to 0-180 range
        self.assertEqual(self.aspect_calculator.calculate_aspect_angle(10, 350), 20)
        self.assertEqual(self.aspect_calculator.calculate_aspect_angle(350, 10), 20)
        
        # Test with sample positions
        self.assertAlmostEqual(
            self.aspect_calculator.calculate_aspect_angle(
                self.test_positions[SUN]["longitude"],
                self.test_positions[MERCURY]["longitude"]
            ),
            0.5  # Almost conjunction
        )
        
        self.assertAlmostEqual(
            self.aspect_calculator.calculate_aspect_angle(
                self.test_positions[SUN]["longitude"],
                self.test_positions[VENUS]["longitude"]
            ),
            60.0  # Sextile
        )
    
    def test_identify_aspect(self):
        """Test identification of aspects."""
        # Test exact aspects
        conjunction = self.aspect_calculator.identify_aspect(0.0)
        self.assertEqual(conjunction["type"], CONJUNCTION)
        self.assertEqual(conjunction["name"], "Conjunction")
        self.assertTrue(conjunction["exact"])
        
        # Test aspect within orb
        trine = self.aspect_calculator.identify_aspect(122.0)
        self.assertEqual(trine["type"], TRINE)
        self.assertFalse(trine["exact"])
        self.assertTrue(0 < trine["strength"] < 1)
        
        # Test no aspect (angle not close to any standard aspect)
        no_aspect = self.aspect_calculator.identify_aspect(47.0)
        self.assertIsNone(no_aspect)
        
        # Test with custom orb factor
        square_wide_orb = self.aspect_calculator.identify_aspect(98.0, orb_factor=1.5)
        self.assertEqual(square_wide_orb["type"], SQUARE)
        
        square_narrow_orb = self.aspect_calculator.identify_aspect(98.0, orb_factor=0.5)
        self.assertIsNone(square_narrow_orb)  # Should be None with narrow orb
    
    def test_calculate_aspect_between_planets(self):
        """Test calculation of aspects between planets."""
        # Test Sun-Mercury conjunction
        sun_mercury = self.aspect_calculator.calculate_aspect_between_planets(
            SUN, MERCURY, self.test_positions
        )
        self.assertEqual(sun_mercury["type"], CONJUNCTION)
        self.assertEqual(sun_mercury["planet1"], SUN)
        self.assertEqual(sun_mercury["planet2"], MERCURY)
        
        # Test Sun-Venus sextile
        sun_venus = self.aspect_calculator.calculate_aspect_between_planets(
            SUN, VENUS, self.test_positions
        )
        self.assertEqual(sun_venus["type"], SEXTILE)
        
        # Test Moon-Jupiter trine
        moon_jupiter = self.aspect_calculator.calculate_aspect_between_planets(
            MOON, JUPITER, self.test_positions
        )
        self.assertEqual(moon_jupiter["type"], TRINE)
        
        # Test Saturn-Sun opposition
        saturn_sun = self.aspect_calculator.calculate_aspect_between_planets(
            SATURN, SUN, self.test_positions
        )
        self.assertEqual(saturn_sun["type"], OPPOSITION)
    
    def test_calculate_all_aspects(self):
        """Test calculation of all aspects."""
        aspects = self.aspect_calculator.calculate_all_aspects(self.test_positions)
        
        # We should have several aspects with our test positions
        self.assertTrue(len(aspects) >= 5)
        
        # Check that all returned items are dictionaries with required keys
        for aspect in aspects:
            self.assertIsInstance(aspect, dict)
            self.assertIn("type", aspect)
            self.assertIn("name", aspect)
            self.assertIn("planet1", aspect)
            self.assertIn("planet2", aspect)
            self.assertIn("strength", aspect)
            self.assertIn("financial_significance", aspect)
    
    def test_calculate_vedic_special_aspects(self):
        """Test calculation of Vedic special aspects."""
        special_aspects = self.aspect_calculator.calculate_vedic_special_aspects(self.test_positions)
        
        # Check that all returned items have the correct format
        for aspect in special_aspects:
            self.assertIsInstance(aspect, dict)
            self.assertEqual(aspect["type"], "vedic_special")
            self.assertIn("planet", aspect)
            self.assertIn("target", aspect)
            self.assertIn("house_offset", aspect)
            self.assertIn("strength", aspect)
    
    def test_get_financial_aspect_forecast(self):
        """Test generation of financial forecast from aspects."""
        # Calculate all aspects
        all_aspects = self.aspect_calculator.calculate_all_aspects(self.test_positions)
        all_aspects += self.aspect_calculator.calculate_vedic_special_aspects(self.test_positions)
        
        # Generate forecast
        forecast = self.aspect_calculator.get_financial_aspect_forecast(all_aspects)
        
        # Check forecast structure
        self.assertIn("trend", forecast)
        self.assertIn("volatility", forecast)
        self.assertIn("confidence", forecast)
        self.assertIn("trend_score", forecast)
        self.assertIn("volatility_score", forecast)
        self.assertIn("description", forecast)
        
        # Trend should be one of the expected values
        self.assertIn(forecast["trend"], ["bullish", "bearish", "neutral"])
        
        # Volatility should be one of the expected values
        self.assertIn(forecast["volatility"], ["high", "moderate", "low"])
        
        # Confidence should be between 0 and 1
        self.assertTrue(0 <= forecast["confidence"] <= 1)
    
    def test_analyze_aspects_for_date(self):
        """Test the convenience function for analyzing aspects for a date."""
        # Test with string date
        result_str = analyze_aspects_for_date("2023-01-01", self.calculator)
        self.assertIn("date", result_str)
        self.assertIn("standard_aspects", result_str)
        self.assertIn("vedic_aspects", result_str)
        self.assertIn("forecast", result_str)
        
        # Test with datetime object
        test_date = datetime.datetime(2023, 1, 1)
        result_dt = analyze_aspects_for_date(test_date, self.calculator)
        self.assertEqual(result_dt["forecast"]["trend"], result_str["forecast"]["trend"])


if __name__ == "__main__":
    unittest.main()
