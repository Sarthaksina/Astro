"""
Vedic Analysis Module for the Cosmic Market Oracle.

This module provides comprehensive Vedic astrological analysis for financial markets,
consolidating core functionality from the vedic_market_analyzer module.
It combines planetary positions, divisional charts, dashas, yogas, and dignities
to generate holistic market forecasts and timing signals.
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

from .planetary_positions import PlanetaryCalculator
from .constants import (
    SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO, RAHU, KETU,
    get_planet_name, GEOCENTRIC, HELIOCENTRIC
)
from .vedic_dignities import (
    calculate_dignity_state, check_combustion, calculate_shadbala, calculate_all_dignities
)
from .divisional_charts import DivisionalCharts
from .financial_yogas import FinancialYogaAnalyzer
from .dasha_systems import DashaCalculator

# Benefic and malefic planets
BENEFIC_PLANETS = {JUPITER, VENUS}
MALEFIC_PLANETS = {SATURN, MARS, RAHU, KETU}


class VedicAnalyzer:
    """
    Integrated Vedic astrological analyzer that combines multiple
    astrological techniques to generate market forecasts and timing signals.
    """
    
    def __init__(self):
        """Initialize the Vedic Analyzer with all required components."""
        self.planetary_calculator = PlanetaryCalculator()
        self.divisional_charts = DivisionalCharts(self.planetary_calculator)
        self.yoga_analyzer = FinancialYogaAnalyzer(self.planetary_calculator)
        self.dasha_calculator = DashaCalculator(self.planetary_calculator)
        
        # Financial significance weights for different components
        self.component_weights = {
            "yogas": 0.35,           # Planetary combinations
            "dashas": 0.25,          # Planetary periods
            "dignities": 0.20,       # Planetary strengths
            "divisional_charts": 0.20  # Divisional chart positions
        }
        
        # Market-relevant planets and their weights
        self.planet_weights = {
            JUPITER: 0.20,  # Expansion, growth, optimism
            SATURN: 0.15,   # Contraction, discipline, fear
            SUN: 0.10,      # Authority, leadership, confidence
            MOON: 0.15,     # Public sentiment, liquidity
            MERCURY: 0.10,  # Communication, trade, commerce
            VENUS: 0.10,    # Luxury, comfort, value
            MARS: 0.10,     # Energy, action, conflict
            RAHU: 0.05,     # Speculation, innovation, disruption
            KETU: 0.05      # Dissolution, spiritual values
        }
        
        # Financial sectors associated with each planet
        self.planet_sectors = {
            SUN: ["government", "gold", "energy", "leadership"],
            MOON: ["real estate", "food", "retail", "public services"],
            MERCURY: ["technology", "media", "telecommunications", "transportation"],
            VENUS: ["luxury", "entertainment", "fashion", "hospitality"],
            MARS: ["defense", "construction", "sports", "manufacturing"],
            JUPITER: ["finance", "education", "legal", "international trade"],
            SATURN: ["infrastructure", "mining", "agriculture", "utilities"],
            RAHU: ["innovation", "disruption", "foreign markets", "emerging tech"],
            KETU: ["pharmaceuticals", "chemicals", "spiritual products", "dissolution"]
        }

    def analyze_date(self, date: Union[str, datetime], 
                    birth_date: Optional[Union[str, datetime]] = None) -> Dict:
        """
        Perform comprehensive Vedic astrological analysis for a specific date.
        
        Args:
            date: The date to analyze
            birth_date: Optional birth date for dasha calculations
            
        Returns:
            Dictionary with comprehensive market analysis
        """
        # Convert string date to datetime if needed
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
            
        # Calculate planetary positions
        positions = self.planetary_calculator.get_all_planets(date)
        
        # Calculate divisional charts
        divisional_charts = self.divisional_charts.calculate_all_divisional_charts(positions)
        
        # Calculate planetary dignities
        dignities = calculate_all_dignities(positions)
        
        # Calculate divisional strengths
        divisional_strengths = self.divisional_charts.calculate_divisional_strengths(divisional_charts)
        
        # Analyze financial yogas
        yogas = self.yoga_analyzer.analyze_financial_yogas(positions, divisional_charts)
        
        # Calculate dasha influences if birth date is provided
        dasha_forecast = None
        if birth_date:
            dasha_forecast = self.dasha_calculator.calculate_dasha_influences(
                date, birth_date, positions
            )
        
        # Generate integrated forecast
        integrated_forecast = self._integrate_forecasts(
            yogas, dasha_forecast, dignities, divisional_strengths, positions
        )
        
        # Generate sector forecasts
        sector_forecasts = self._generate_sector_forecasts(positions, dignities, yogas)
        
        # Generate timing signals
        timing_signals = self._generate_timing_signals(positions, yogas, dignities)
        
        # Compile complete analysis
        analysis = {
            "date": date.isoformat() if isinstance(date, datetime) else date,
            "planetary_positions": {
                get_planet_name(planet_id): {
                    k: v for k, v in details.items() 
                    if k in ["longitude", "is_retrograde", "nakshatra"]
                }
                for planet_id, details in positions.items()
            },
            "integrated_forecast": integrated_forecast,
            "sector_forecasts": sector_forecasts,
            "timing_signals": timing_signals,
            "key_yogas": yogas.get("key_yogas", []),
            "key_dignities": {
                get_planet_name(planet_id): dignity_state
                for planet_id, dignity_state in dignities.items()
                if planet_id in self.planet_weights
            }
        }
        
        # Add dasha information if available
        if dasha_forecast:
            analysis["dasha_influences"] = dasha_forecast
            
        return analysis
        
    def _integrate_forecasts(self, yoga_forecast: Dict, dasha_forecast: Optional[Dict],
                           dignities: Dict, divisional_strengths: Dict, 
                           positions: Dict) -> Dict:
        """
        Integrate different forecast components into a unified market forecast.
        
        Args:
            yoga_forecast: Forecast based on yogas
            dasha_forecast: Forecast based on dashas (optional)
            dignities: Planetary dignity calculations
            divisional_strengths: Strengths in divisional charts
            positions: Planetary positions
            
        Returns:
            Dictionary with integrated forecast
        """
        # Initialize scores
        trend_score = 0.0
        volatility_score = 0.0
        confidence = 0.0
        key_factors = []
        
        # Process yoga influences
        if yoga_forecast:
            yoga_trend = yoga_forecast.get("market_trend", 0)
            yoga_volatility = yoga_forecast.get("volatility", 0)
            yoga_confidence = yoga_forecast.get("confidence", 0.5)
            
            trend_score += yoga_trend * self.component_weights["yogas"]
            volatility_score += yoga_volatility * self.component_weights["yogas"]
            confidence += yoga_confidence * self.component_weights["yogas"]
            
            # Add key yogas to factors
            for yoga in yoga_forecast.get("key_yogas", [])[:3]:  # Top 3 yogas
                key_factors.append(f"Yoga: {yoga}")
        
        # Process dasha influences if available
        if dasha_forecast:
            dasha_trend = dasha_forecast.get("market_trend", 0)
            dasha_volatility = dasha_forecast.get("volatility", 0)
            dasha_confidence = dasha_forecast.get("confidence", 0.5)
            
            trend_score += dasha_trend * self.component_weights["dashas"]
            volatility_score += dasha_volatility * self.component_weights["dashas"]
            confidence += dasha_confidence * self.component_weights["dashas"]
            
            # Add key dasha to factors
            current_dasha = dasha_forecast.get("current_dasha", {})
            if current_dasha:
                key_factors.append(f"Dasha: {current_dasha.get('description', '')}")
        
        # Process dignity influences
        dignity_trend = 0
        dignity_volatility = 0
        dignity_count = 0
        
        for planet_id, dignity in dignities.items():
            if planet_id not in self.planet_weights:
                continue
                
            planet_weight = self.planet_weights[planet_id]
            dignity_value = 0
            
            # Convert dignity state to numerical value
            if dignity == "exalted":
                dignity_value = 1.0
            elif dignity == "own_sign":
                dignity_value = 0.5
            elif dignity == "friendly":
                dignity_value = 0.25
            elif dignity == "neutral":
                dignity_value = 0
            elif dignity == "enemy":
                dignity_value = -0.25
            elif dignity == "debilitated":
                dignity_value = -1.0
                
            # Adjust for benefic/malefic nature
            if planet_id in BENEFIC_PLANETS:
                dignity_trend += dignity_value * planet_weight
            elif planet_id in MALEFIC_PLANETS:
                dignity_trend -= dignity_value * planet_weight
            else:
                # For neutral planets, consider their dignity directly
                dignity_trend += dignity_value * planet_weight * 0.5
                
            # Retrograde planets increase volatility
            if positions[planet_id].get("is_retrograde", False):
                dignity_volatility += 0.2 * planet_weight
                
            dignity_count += 1
            
            # Add significant dignities to factors
            if dignity in ["exalted", "debilitated"]:
                key_factors.append(f"{get_planet_name(planet_id)} {dignity}")
        
        # Normalize dignity trend and add to total
        if dignity_count > 0:
            dignity_trend /= dignity_count
            trend_score += dignity_trend * self.component_weights["dignities"]
            volatility_score += dignity_volatility * self.component_weights["dignities"]
            confidence += 0.7 * self.component_weights["dignities"]  # Dignities are reliable
        
        # Process divisional chart strengths
        div_trend = 0
        div_count = 0
        
        for planet_id, strengths in divisional_strengths.items():
            if planet_id not in self.planet_weights:
                continue
                
            planet_weight = self.planet_weights[planet_id]
            avg_strength = sum(strengths.values()) / len(strengths) if strengths else 0
            
            # Adjust for benefic/malefic nature
            if planet_id in BENEFIC_PLANETS:
                div_trend += avg_strength * planet_weight
            elif planet_id in MALEFIC_PLANETS:
                div_trend -= avg_strength * planet_weight
            else:
                div_trend += avg_strength * planet_weight * 0.5
                
            div_count += 1
            
            # Add significant divisional strengths to factors
            d9_strength = strengths.get(9, 0)  # D9 chart (Navamsa)
            d10_strength = strengths.get(10, 0)  # D10 chart (Dasamsa)
            
            if d9_strength > 0.8 or d10_strength > 0.8:
                key_factors.append(f"{get_planet_name(planet_id)} strong in divisional charts")
        
        # Normalize divisional trend and add to total
        if div_count > 0:
            div_trend /= div_count
            trend_score += div_trend * self.component_weights["divisional_charts"]
            confidence += 0.6 * self.component_weights["divisional_charts"]
        
        # Normalize final scores
        trend_score = max(-1.0, min(1.0, trend_score))  # -1.0 to 1.0
        volatility_score = max(0.0, min(1.0, volatility_score))  # 0.0 to 1.0
        confidence = max(0.3, min(0.9, confidence))  # 0.3 to 0.9
        
        # Determine trend description
        if trend_score > 0.5:
            trend = "Bullish"
            description = "Strong upward momentum expected"
        elif trend_score > 0.2:
            trend = "Moderately Bullish"
            description = "Gradual upward movement likely"
        elif trend_score > -0.2:
            trend = "Neutral"
            description = "Sideways movement expected"
        elif trend_score > -0.5:
            trend = "Moderately Bearish"
            description = "Gradual downward movement likely"
        else:
            trend = "Bearish"
            description = "Strong downward momentum expected"
            
        # Adjust for volatility
        if volatility_score > 0.7:
            trend = f"Volatile {trend}"
            description = f"High volatility with {description.lower()}"
        elif volatility_score > 0.4:
            trend = f"Moderately Volatile {trend}"
            description = f"Some volatility with {description.lower()}"
            
        # Create integrated forecast
        integrated_forecast = {
            "trend": trend,
            "trend_score": float(trend_score),
            "volatility": "High" if volatility_score > 0.7 else "Moderate" if volatility_score > 0.4 else "Low",
            "volatility_score": float(volatility_score),
            "confidence": float(confidence),
            "description": description,
            "key_factors": key_factors[:5]  # Top 5 factors
        }
        
        return integrated_forecast
    
    def _generate_sector_forecasts(self, positions: Dict, dignities: Dict, yogas: Dict) -> Dict:
        """
        Generate forecasts for different market sectors based on planetary positions.
        
        Args:
            positions: Planetary positions
            dignities: Planetary dignity calculations
            yogas: Analyzed yogas
            
        Returns:
            Dictionary with sector forecasts
        """
        sector_forecasts = {}
        
        # Process each planet and its associated sectors
        for planet_id, sectors in self.planet_sectors.items():
            if planet_id not in positions:
                continue
                
            # Get planet details
            planet_position = positions[planet_id]
            dignity = dignities.get(planet_id, "neutral")
            is_retrograde = planet_position.get("is_retrograde", False)
            
            # Calculate base strength
            strength = 0.0
            
            # Adjust based on dignity
            if dignity == "exalted":
                strength += 0.8
            elif dignity == "own_sign":
                strength += 0.5
            elif dignity == "friendly":
                strength += 0.3
            elif dignity == "neutral":
                strength += 0.0
            elif dignity == "enemy":
                strength -= 0.3
            elif dignity == "debilitated":
                strength -= 0.8
                
            # Adjust for retrograde motion
            if is_retrograde:
                if planet_id in BENEFIC_PLANETS:
                    strength -= 0.2  # Retrograde weakens benefics
                elif planet_id in MALEFIC_PLANETS:
                    strength += 0.1  # Retrograde can strengthen malefics
                    
            # Adjust for combustion (too close to Sun)
            if planet_id != SUN and "combust" in planet_position.get("special_states", []):
                strength -= 0.4
                
            # Normalize strength
            strength = max(-1.0, min(1.0, strength))
            
            # Generate forecast for each sector
            for sector in sectors:
                # Determine trend
                if strength > 0.5:
                    trend = "Strong positive"
                elif strength > 0.2:
                    trend = "Positive"
                elif strength > -0.2:
                    trend = "Neutral"
                elif strength > -0.5:
                    trend = "Negative"
                else:
                    trend = "Strong negative"
                    
                # Determine volatility
                volatility = "Low"
                if is_retrograde:
                    volatility = "High"
                elif planet_id in [MARS, RAHU]:
                    volatility = "Moderate"
                    
                # Create sector forecast
                sector_forecasts[sector] = {
                    "trend": trend,
                    "strength": float(strength),
                    "volatility": volatility,
                    "influenced_by": get_planet_name(planet_id),
                    "planet_status": f"{dignity}{', retrograde' if is_retrograde else ''}"
                }
        
        # Special case: Technology sector influenced by Mercury-Rahu combinations
        if "technology" in sector_forecasts:
            mercury_rahu_aspect = False
            for yoga in yogas.get("key_yogas", []):
                if "Mercury" in yoga and "Rahu" in yoga:
                    mercury_rahu_aspect = True
                    break
                    
            if mercury_rahu_aspect:
                tech = sector_forecasts["technology"]
                tech["volatility"] = "High"
                tech["special_note"] = "Mercury-Rahu influence suggests innovation but high volatility"
                
        # Special case: Finance sector influenced by Jupiter-Saturn relationship
        if "finance" in sector_forecasts:
            jupiter_saturn_aspect = False
            for yoga in yogas.get("key_yogas", []):
                if "Jupiter" in yoga and "Saturn" in yoga:
                    jupiter_saturn_aspect = True
                    break
                    
            if jupiter_saturn_aspect:
                finance = sector_forecasts["finance"]
                finance["special_note"] = "Jupiter-Saturn influence suggests regulatory changes"
        
        return sector_forecasts
    
    def _generate_timing_signals(self, positions: Dict, yogas: Dict, dignities: Dict) -> Dict:
        """
        Generate market timing signals based on astrological factors.
        
        Args:
            positions: Planetary positions
            yogas: Analyzed yogas
            dignities: Planetary dignity calculations
            
        Returns:
            Dictionary with timing signals
        """
        # Initialize timing signals
        entry_signals = []
        exit_signals = []
        caution_signals = []
        
        # Check for retrograde Mercury (communication/trade disruptions)
        if positions.get(MERCURY, {}).get("is_retrograde", False):
            caution_signals.append("Mercury retrograde suggests potential communication/trade disruptions")
        
        # Check for Moon phase (market sentiment)
        moon_pos = positions.get(MOON, {}).get("longitude", 0)
        sun_pos = positions.get(SUN, {}).get("longitude", 0)
        
        # Calculate Moon phase angle
        moon_phase_angle = (moon_pos - sun_pos) % 360
        
        # New Moon: 0-45 degrees
        if 0 <= moon_phase_angle < 45:
            entry_signals.append("New Moon phase suggests beginning of new market cycle")
        # First Quarter: 45-135 degrees
        elif 45 <= moon_phase_angle < 135:
            entry_signals.append("First Quarter Moon suggests momentum building")
        # Full Moon: 135-225 degrees
        elif 135 <= moon_phase_angle < 225:
            caution_signals.append("Full Moon suggests potential market volatility or reversal")
        # Last Quarter: 225-315 degrees
        elif 225 <= moon_phase_angle < 315:
            exit_signals.append("Last Quarter Moon suggests consolidation or profit-taking")
        # Balsamic Moon: 315-360 degrees
        else:
            caution_signals.append("Balsamic Moon suggests preparing for new market cycle")
        
        # Check for applying aspects between key planets
        from .planetary_positions import get_planetary_aspects
        aspects = get_planetary_aspects(positions)
        
        for aspect in aspects:
            planet1 = aspect["planet1"]
            planet2 = aspect["planet2"]
            aspect_type = aspect["aspect_type"]
            applying = aspect.get("applying", False)
            
            # Jupiter-Sun favorable aspects (growth, optimism)
            if ((planet1 == JUPITER and planet2 == SUN) or (planet1 == SUN and planet2 == JUPITER)) and \
               aspect_type in ["conjunction", "trine", "sextile"] and applying:
                entry_signals.append(f"Applying {aspect_type} between Jupiter-Sun suggests favorable growth conditions")
            
            # Saturn-Mars challenging aspects (restrictions, frustrations)
            if ((planet1 == SATURN and planet2 == MARS) or (planet1 == MARS and planet2 == SATURN)) and \
               aspect_type in ["conjunction", "opposition", "square"] and applying:
                exit_signals.append(f"Applying {aspect_type} between Saturn-Mars suggests challenging market conditions")
            
            # Venus-Jupiter favorable aspects (financial expansion)
            if ((planet1 == VENUS and planet2 == JUPITER) or (planet1 == JUPITER and planet2 == VENUS)) and \
               aspect_type in ["conjunction", "trine", "sextile"] and applying:
                entry_signals.append(f"Applying {aspect_type} between Venus-Jupiter suggests favorable financial conditions")
        
        # Check for planets at critical degrees
        for planet_id, position in positions.items():
            longitude = position.get("longitude", 0)
            
            # Check for planets at 0, 15, or 29 degrees (critical degrees)
            degree_in_sign = longitude % 30
            if degree_in_sign < 1 or (14 < degree_in_sign < 16) or degree_in_sign > 29:
                planet_name = get_planet_name(planet_id)
                if planet_id in BENEFIC_PLANETS:
                    entry_signals.append(f"{planet_name} at critical degree suggests potential favorable turning point")
                elif planet_id in MALEFIC_PLANETS:
                    caution_signals.append(f"{planet_name} at critical degree suggests potential challenging turning point")
                else:
                    caution_signals.append(f"{planet_name} at critical degree suggests potential market turning point")
        
        # Check for yogas related to timing
        for yoga in yogas.get("key_yogas", []):
            if "Gaja Kesari" in yoga:  # Jupiter-Moon yoga (prosperity)
                entry_signals.append("Gaja Kesari Yoga suggests favorable timing for entries")
            elif "Viparita Raja" in yoga:  # Reversal yoga
                entry_signals.append("Viparita Raja Yoga suggests potential for positive reversals")
            elif "Dainya" in yoga:  # Challenging yoga
                exit_signals.append("Dainya Yoga suggests caution and potential exits")
        
        # Determine overall timing signal
        if len(entry_signals) > len(exit_signals) + len(caution_signals):
            overall_timing = "favorable"
        elif len(exit_signals) > len(entry_signals):
            overall_timing = "unfavorable"
        elif len(caution_signals) > len(entry_signals) + len(exit_signals):
            overall_timing = "cautious"
        else:
            overall_timing = "neutral"
        
        # Create timing signals dictionary
        timing_signals = {
            "entry_signals": entry_signals,
            "exit_signals": exit_signals,
            "caution_signals": caution_signals,
            "overall_timing": overall_timing
        }
        
        return timing_signals
    
    def analyze_date_range(self, start_date: Union[str, datetime],
                          end_date: Union[str, datetime],
                          birth_date: Optional[Union[str, datetime]] = None,
                          interval_days: int = 1) -> pd.DataFrame:
        """
        Analyze a range of dates and return the results as a DataFrame.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            birth_date: Optional birth date for dasha calculations
            interval_days: Interval between analysis dates in days
            
        Returns:
            DataFrame with analysis results for each date
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
            
        # Generate date range
        current_date = start_date
        results = []
        
        while current_date <= end_date:
            # Analyze current date
            analysis = self.analyze_date(current_date, birth_date)
            
            # Extract key metrics for DataFrame
            result = {
                "date": current_date,
                "trend": analysis["integrated_forecast"]["trend"],
                "trend_score": analysis["integrated_forecast"]["trend_score"],
                "volatility": analysis["integrated_forecast"]["volatility"],
                "volatility_score": analysis["integrated_forecast"]["volatility_score"],
                "confidence": analysis["integrated_forecast"]["confidence"],
                "timing": analysis["timing_signals"]["overall_timing"]
            }
            
            results.append(result)
            
            # Move to next date
            current_date += timedelta(days=interval_days)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        return df
    
    def plot_forecast_trends(self, df: pd.DataFrame, 
                           market_data: Optional[pd.DataFrame] = None,
                           title: str = "Vedic Astrological Market Forecast"):
        """
        Plot forecast trends from analyze_date_range results.
        
        Args:
            df: DataFrame from analyze_date_range
            market_data: Optional market price data for comparison
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        # Create trend score plot
        ax1 = plt.subplot(211)
        ax1.plot(df["date"], df["trend_score"], 'b-', label="Astrological Trend")
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axhline(y=0.2, color='g', linestyle='--', alpha=0.3)
        ax1.axhline(y=-0.2, color='r', linestyle='--', alpha=0.3)
        ax1.set_ylabel("Trend Score")
        ax1.set_title(title)
        ax1.legend(loc="upper left")
        
        # Add market data if provided
        if market_data is not None:
            # Ensure dates align
            market_data = market_data.set_index("date").reindex(df["date"]).reset_index()
            
            # Create twin axis for price
            ax2 = ax1.twinx()
            ax2.plot(market_data["date"], market_data["close"], 'r-', alpha=0.7, label="Market Price")
            ax2.set_ylabel("Price")
            ax2.legend(loc="upper right")
        
        # Create volatility plot
        ax3 = plt.subplot(212)
        ax3.plot(df["date"], df["volatility_score"], 'g-', label="Volatility Score")
        ax3.axhline(y=0.4, color='y', linestyle='--', alpha=0.3)
        ax3.axhline(y=0.7, color='r', linestyle='--', alpha=0.3)
        ax3.set_ylabel("Volatility Score")
        ax3.set_xlabel("Date")
        ax3.legend(loc="upper left")
        
        # Add timing signals
        timing_colors = {
            "favorable": "g",
            "unfavorable": "r",
            "cautious": "y",
            "neutral": "b"
        }
        
        for i, (_, row) in enumerate(df.iterrows()):
            timing = row["timing"]
            ax3.scatter(row["date"], 0.1, color=timing_colors.get(timing, "b"), s=50, alpha=0.7)
        
        plt.tight_layout()
        plt.show()
