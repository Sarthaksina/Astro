# Cosmic Market Oracle - Signal Generator

"""
This module provides specialized signal generators for trading strategies
based on Vedic astrological principles and market data.

It includes:
- Vedic astrological signal generators
- Planetary cycle signal generators
- Combined signal generators that integrate multiple sources
- Signal filtering and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

from src.astro_engine.planetary_positions import (
    PlanetaryCalculator, analyze_market_trend, analyze_financial_yogas
)
# Assuming get_planet_name might be needed elsewhere or by other classes in this file.
from ..astro_engine.constants import (
    get_planet_name, NAKSHATRA_PROPERTIES, PLANET_FINANCIAL_SIGNIFICANCE,
    SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU
)
from src.feature_engineering.astrological_features import AstrologicalFeatureGenerator
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("signal_generator")


class SignalGenerator:
    """Base class for all signal generators."""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the signal generator.
        
        Args:
            name: Signal generator name
            description: Signal generator description
        """
        self.name = name
        self.description = description
        self.calculator = PlanetaryCalculator()
        self.feature_generator = AstrologicalFeatureGenerator()
    
    def generate_signals(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market and planetary data.
        
        Args:
            market_data: DataFrame containing market data
            planetary_data: DataFrame containing planetary data
            
        Returns:
            DataFrame with trading signals
        """
        # Base implementation returns no signals
        signals = pd.DataFrame(index=market_data.index)
        signals["signal"] = 0  # 0: no signal, 1: buy, -1: sell
        signals["strength"] = 0.0
        signals["reason"] = ""
        
        return signals


class VedicNakshatraSignalGenerator(SignalGenerator):
    """Signal generator based on Moon nakshatra positions."""
    
    def __init__(self, name: str = "Nakshatra Signal Generator", 
                description: str = "Generates signals based on Moon nakshatra positions"):
        """
        Initialize the Nakshatra signal generator.
        
        Args:
            name: Signal generator name
            description: Signal generator description
        """
        super().__init__(name, description)
        
        # Nakshatra financial classifications will now be derived from NAKSHATRA_PROPERTIES
        
        # Signal strength modifiers based on nakshatra pada (quarter)
        self.pada_strength_modifiers = {
            1: 0.7,  # First pada - moderate strength
            2: 0.9,  # Second pada - strong
            3: 1.0,  # Third pada - strongest
            4: 0.8   # Fourth pada - strong but waning
        }
    
    def generate_signals(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Moon nakshatra positions.
        
        Args:
            market_data: DataFrame containing market data
            planetary_data: DataFrame containing planetary data
            
        Returns:
            DataFrame with trading signals
        """
        if market_data.empty or planetary_data.empty:
            logger.warning("Empty data provided for signal generation")
            return pd.DataFrame()
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=market_data.index)
        signals["signal"] = 0  # 0: no signal, 1: buy, -1: sell
        signals["strength"] = 0.0
        signals["reason"] = ""
        
        # Process each day
        for date in signals.index:
            # Skip if no planetary data for this date
            if date not in planetary_data.index:
                continue
            
            # Get planetary data for this date
            planets_for_date = planetary_data.loc[date]
            
            # Check if we have Moon nakshatra information
            if "moon_nakshatra" in planets_for_date: # This gives the name
                nakshatra_name_from_data = planets_for_date["moon_nakshatra"]
                pada = planets_for_date.get("moon_nakshatra_pada", 1) # Default to pada 1 if not found
                if not isinstance(pada, int) or not (1 <= pada <= 4): # Basic validation for pada
                    pada = 1
                    logger.debug(f"Invalid pada value for {nakshatra_name_from_data} on {date}, defaulting to 1.")

                moon_nak_num = None
                financial_nature = "neutral" # Default
                nakshatra_name_found = nakshatra_name_from_data # Use provided name initially for reason string

                for num, props in NAKSHATRA_PROPERTIES.items():
                    if props.get("name") == nakshatra_name_from_data:
                        moon_nak_num = num
                        financial_nature = props.get("financial_nature", "neutral")
                        nakshatra_name_found = props.get("name") # Use name from constants for consistency
                        break
                
                if moon_nak_num is None:
                    logger.warning(f"Moon Nakshatra name '{nakshatra_name_from_data}' not found in NAKSHATRA_PROPERTIES for date {date}. Skipping Nakshatra signals.")
                    signals.loc[date, "reason"] = f"Unknown Moon nakshatra: {nakshatra_name_from_data}"
                    continue

                signal_direction = 0
                base_strength = 0.0
                reason = ""

                if financial_nature == "bullish":
                    signal_direction = 1
                    base_strength = 0.7
                    reason = f"Bullish Moon nakshatra: {nakshatra_name_found}"
                elif financial_nature == "bearish":
                    signal_direction = -1
                    base_strength = 0.7
                    reason = f"Bearish Moon nakshatra: {nakshatra_name_found}"
                elif financial_nature == "volatile":
                    base_strength = 0.3
                    reason = f"Volatile Moon nakshatra: {nakshatra_name_found}"
                else: # Neutral or unknown
                    reason = f"Neutral Moon nakshatra: {nakshatra_name_found}"

                signal_strength = base_strength
                if signal_direction != 0:
                    signal_strength *= self.pada_strength_modifiers.get(pada, 0.7)
                    reason += f", Pada {pada}"
                elif financial_nature == "volatile": # If volatile and no direction, strength is just base_strength for volatile
                     signal_strength = base_strength
                     reason += f", Pada {pada}"
                
                if abs(signal_direction) > 0 and signal_strength >= 0.5:
                    signals.loc[date, "signal"] = signal_direction
                    signals.loc[date, "strength"] = signal_strength
                    signals.loc[date, "reason"] = reason
                elif financial_nature == "volatile" and signal_strength >= 0.3:
                    signals.loc[date, "signal"] = 0
                    signals.loc[date, "strength"] = signal_strength
                    signals.loc[date, "reason"] = reason
                else:
                    signals.loc[date, "signal"] = 0
                    signals.loc[date, "strength"] = 0.0
                    signals.loc[date, "reason"] = reason if reason else f"Neutral Moon nakshatra: {nakshatra_name_found}, Pada {pada}"
            else:
                signals.loc[date, "reason"] = "Moon nakshatra data not found"
        
        return signals


class VedicYogaSignalGenerator(SignalGenerator):
    """Signal generator based on Vedic astrological yogas (planetary combinations)."""
    
    def __init__(self, name: str = "Yoga Signal Generator", 
                description: str = "Generates signals based on Vedic astrological yogas"):
        """
        Initialize the Yoga signal generator.
        
        Args:
            name: Signal generator name
            description: Signal generator description
        """
        super().__init__(name, description)
        
        # Minimum yoga strength to generate a signal
        self.min_yoga_strength = 0.6
    
    def generate_signals(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Vedic astrological yogas.
        
        Args:
            market_data: DataFrame containing market data
            planetary_data: DataFrame containing planetary data
            
        Returns:
            DataFrame with trading signals
        """
        if market_data.empty or planetary_data.empty:
            logger.warning("Empty data provided for signal generation")
            return pd.DataFrame()
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=market_data.index)
        signals["signal"] = 0  # 0: no signal, 1: buy, -1: sell
        signals["strength"] = 0.0
        signals["reason"] = ""
        
        # Process each day
        for date in signals.index:
            # Skip if no planetary data for this date
            if date not in planetary_data.index:
                continue
            
            # Get planetary data for this date
            planets_for_date = planetary_data.loc[date]
            
            # Check for financial yogas
            bullish_yogas = planets_for_date.get("bullish_yoga_count", 0)
            bearish_yogas = planets_for_date.get("bearish_yoga_count", 0)
            
            # Initialize signal components
            signal_direction = 0
            signal_strength = 0.0
            reason = ""
            
            # Determine signal based on yoga counts
            if bullish_yogas > bearish_yogas and bullish_yogas > 0:
                # Bullish signal
                signal_direction = 1
                signal_strength = min(bullish_yogas * 0.2, 0.9)  # Cap at 0.9
                reason = f"Bullish yogas: {bullish_yogas}"
                
                # Add specific yoga details if available
                if "bullish_yogas" in planets_for_date:
                    yoga_list = planets_for_date["bullish_yogas"]
                    if isinstance(yoga_list, list) and len(yoga_list) > 0:
                        reason += f" ({', '.join(yoga_list[:3])})"
                    elif isinstance(yoga_list, str):
                        reason += f" ({yoga_list})"
                
            elif bearish_yogas > bullish_yogas and bearish_yogas > 0:
                # Bearish signal
                signal_direction = -1
                signal_strength = min(bearish_yogas * 0.2, 0.9)  # Cap at 0.9
                reason = f"Bearish yogas: {bearish_yogas}"
                
                # Add specific yoga details if available
                if "bearish_yogas" in planets_for_date:
                    yoga_list = planets_for_date["bearish_yogas"]
                    if isinstance(yoga_list, list) and len(yoga_list) > 0:
                        reason += f" ({', '.join(yoga_list[:3])})"
                    elif isinstance(yoga_list, str):
                        reason += f" ({yoga_list})"
            
            # Record signal if strong enough
            if abs(signal_direction) > 0 and signal_strength >= self.min_yoga_strength:
                signals.loc[date, "signal"] = signal_direction
                signals.loc[date, "strength"] = signal_strength
                signals.loc[date, "reason"] = reason
        
        return signals


class VedicDashaSignalGenerator(SignalGenerator):
    """Signal generator based on Vimshottari Dasha periods."""
    
    def __init__(self, name: str = "Dasha Signal Generator", 
                description: str = "Generates signals based on Vimshottari Dasha periods"):
        """
        Initialize the Dasha signal generator.
        
        Args:
            name: Signal generator name
            description: Signal generator description
        """
        super().__init__(name, description)
        
        # self.dasha_financial_map is now removed. PLANET_FINANCIAL_SIGNIFICANCE will be used directly.
        
        # Minimum signal strength to generate a trade
        self.min_signal_strength = 0.6
    
    def generate_signals(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Vimshottari Dasha periods.
        
        Args:
            market_data: DataFrame containing market data
            planetary_data: DataFrame containing planetary data
            
        Returns:
            DataFrame with trading signals
        """
        if market_data.empty or planetary_data.empty:
            logger.warning("Empty data provided for signal generation")
            return pd.DataFrame()
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=market_data.index)
        signals["signal"] = 0  # 0: no signal, 1: buy, -1: sell
        signals["strength"] = 0.0
        signals["reason"] = ""
        
        # Process each day
        for date in signals.index:
            # Skip if no planetary data for this date
            if date not in planetary_data.index:
                continue
            
            # Get planetary data for this date
            planets_for_date = planetary_data.loc[date]
            
            # Check if we have dasha information
            if "current_dasha_lord" in planets_for_date:
                dasha_lord_name = planets_for_date["current_dasha_lord"]
                antardasha_lord_name = planets_for_date.get("current_antardasha_lord")

                PLANET_NAME_TO_ID = {
                    "Sun": SUN, "Moon": MOON, "Mars": MARS, "Mercury": MERCURY,
                    "Jupiter": JUPITER, "Venus": VENUS, "Saturn": SATURN,
                    "Rahu": RAHU, "Ketu": KETU
                }
                default_financial_props = {"nature": "neutral", "strength": 0.5, "volatility": "moderate"}

                dasha_lord_id = PLANET_NAME_TO_ID.get(dasha_lord_name)
                dasha_info = PLANET_FINANCIAL_SIGNIFICANCE.get(dasha_lord_id, default_financial_props) if dasha_lord_id is not None else default_financial_props
                
                antardasha_info = None
                if antardasha_lord_name:
                    antardasha_lord_id = PLANET_NAME_TO_ID.get(antardasha_lord_name)
                    antardasha_info = PLANET_FINANCIAL_SIGNIFICANCE.get(antardasha_lord_id, default_financial_props) if antardasha_lord_id is not None else default_financial_props
                
                # Determine signal direction and strength
                if dasha_info["nature"] == "bullish":
                    signal_direction = 1
                    signal_strength = dasha_info["strength"]
                    reason = f"Bullish dasha lord: {dasha_lord_name}"
                elif dasha_info["nature"] == "bearish":
                    signal_direction = -1
                    signal_strength = dasha_info["strength"]
                    reason = f"Bearish dasha lord: {dasha_lord_name}"
                else:
                    signal_direction = 0
                    signal_strength = 0.0
                    reason = f"Neutral dasha lord: {dasha_lord_name}"
                
                # Adjust signal strength based on antardasha if available
                if antardasha_info and antardasha_lord_name: # Check if antardasha_info was successfully retrieved
                    if antardasha_info["nature"] == dasha_info["nature"]:
                        # Reinforcing antardasha
                        signal_strength = min(signal_strength + 0.1, 0.95)
                        reason += f", reinforced by {antardasha_lord_name} antardasha"
                    elif antardasha_info["nature"] != "neutral" and dasha_info["nature"] != "neutral" and antardasha_info["nature"] != dasha_info["nature"]:
                        # Contradicting antardasha
                        signal_strength = max(signal_strength - 0.2, 0.0)
                        reason += f", weakened by {antardasha_lord_name} antardasha"
                
                # Record signal if strong enough
                if abs(signal_direction) > 0 and signal_strength >= self.min_signal_strength:
                    signals.loc[date, "signal"] = signal_direction
                    signals.loc[date, "strength"] = signal_strength
                    signals.loc[date, "reason"] = reason
        
        return signals


class CombinedSignalGenerator(SignalGenerator):
    """Combines signals from multiple generators with weighted aggregation."""
    
    def __init__(self, name: str = "Combined Signal Generator", 
                description: str = "Combines signals from multiple generators"):
        """
        Initialize the Combined signal generator.
        
        Args:
            name: Signal generator name
            description: Signal generator description
        """
        super().__init__(name, description)
        
        # Initialize signal generators
        self.generators = []
        self.weights = {}
        
        # Default generators
        self.add_generator(VedicNakshatraSignalGenerator(), 0.3)
        self.add_generator(VedicYogaSignalGenerator(), 0.4)
        self.add_generator(VedicDashaSignalGenerator(), 0.3)
        
        # Minimum signal strength to generate a trade
        self.min_signal_strength = 0.6
    
    def add_generator(self, generator: SignalGenerator, weight: float = 1.0):
        """
        Add a signal generator to the combination.
        
        Args:
            generator: Signal generator to add
            weight: Weight of this generator in the combination
        """
        self.generators.append(generator)
        self.weights[generator.name] = weight
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total_weight
    
    def generate_signals(self, market_data: pd.DataFrame, planetary_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals by combining multiple signal generators.
        
        Args:
            market_data: DataFrame containing market data
            planetary_data: DataFrame containing planetary data
            
        Returns:
            DataFrame with trading signals
        """
        if market_data.empty or planetary_data.empty:
            logger.warning("Empty data provided for signal generation")
            return pd.DataFrame()
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=market_data.index)
        signals["signal"] = 0  # 0: no signal, 1: buy, -1: sell
        signals["strength"] = 0.0
        signals["reason"] = ""
        
        # Generate signals from each generator
        all_signals = {}
        for generator in self.generators:
            all_signals[generator.name] = generator.generate_signals(market_data, planetary_data)
        
        # Combine signals for each day
        for date in signals.index:
            bullish_signals = []
            bearish_signals = []
            
            # Collect signals from all generators
            for generator in self.generators:
                gen_signals = all_signals[generator.name]
                
                if date in gen_signals.index:
                    signal = gen_signals.loc[date, "signal"]
                    strength = gen_signals.loc[date, "strength"]
                    reason = gen_signals.loc[date, "reason"]
                    weight = self.weights[generator.name]
                    
                    if signal > 0 and strength > 0:
                        bullish_signals.append({
                            "generator": generator.name,
                            "strength": strength,
                            "weighted_strength": strength * weight,
                            "reason": reason
                        })
                    elif signal < 0 and strength > 0:
                        bearish_signals.append({
                            "generator": generator.name,
                            "strength": strength,
                            "weighted_strength": strength * weight,
                            "reason": reason
                        })
            
            # Determine final signal
            if bullish_signals and not bearish_signals:
                # Only bullish signals
                total_weighted_strength = sum(s["weighted_strength"] for s in bullish_signals)
                reasons = [f"{s['generator']}: {s['reason']}" for s in bullish_signals]
                
                signals.loc[date, "signal"] = 1
                signals.loc[date, "strength"] = total_weighted_strength
                signals.loc[date, "reason"] = " | ".join(reasons)
                
            elif bearish_signals and not bullish_signals:
                # Only bearish signals
                total_weighted_strength = sum(s["weighted_strength"] for s in bearish_signals)
                reasons = [f"{s['generator']}: {s['reason']}" for s in bearish_signals]
                
                signals.loc[date, "signal"] = -1
                signals.loc[date, "strength"] = total_weighted_strength
                signals.loc[date, "reason"] = " | ".join(reasons)
                
            elif bullish_signals and bearish_signals:
                # Both bullish and bearish signals
                bullish_strength = sum(s["weighted_strength"] for s in bullish_signals)
                bearish_strength = sum(s["weighted_strength"] for s in bearish_signals)
                
                if bullish_strength > bearish_strength:
                    # Net bullish
                    net_strength = bullish_strength - bearish_strength
                    bullish_reasons = [f"{s['generator']}: {s['reason']}" for s in bullish_signals]
                    
                    signals.loc[date, "signal"] = 1
                    signals.loc[date, "strength"] = net_strength
                    signals.loc[date, "reason"] = " | ".join(bullish_reasons)
                    
                elif bearish_strength > bullish_strength:
                    # Net bearish
                    net_strength = bearish_strength - bullish_strength
                    bearish_reasons = [f"{s['generator']}: {s['reason']}" for s in bearish_signals]
                    
                    signals.loc[date, "signal"] = -1
                    signals.loc[date, "strength"] = net_strength
                    signals.loc[date, "reason"] = " | ".join(bearish_reasons)
        
        # Filter signals by minimum strength
        signals.loc[signals["strength"] < self.min_signal_strength, "signal"] = 0
        
        return signals


class SignalFilter:
    """Filters and validates trading signals."""
    
    def __init__(self):
        """Initialize the signal filter."""
        pass
    
    def filter_signals(self, signals: pd.DataFrame, market_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Filter trading signals based on various criteria.
        
        Args:
            signals: DataFrame with trading signals
            market_data: Optional market data for additional filtering
            
        Returns:
            Filtered signals DataFrame
        """
        if signals.empty:
            return signals
        
        # Make a copy to avoid modifying the original
        filtered = signals.copy()
        
        # 1. Remove weak signals
        filtered.loc[filtered["strength"] < 0.5, "signal"] = 0
        
        # 2. Remove signals on consecutive days (keep only the first)
        prev_signal = 0
        for date in filtered.index:
            current_signal = filtered.loc[date, "signal"]
            
            if current_signal == prev_signal and current_signal != 0:
                filtered.loc[date, "signal"] = 0
            
            prev_signal = current_signal if current_signal != 0 else prev_signal
        
        # 3. Apply market-based filters if market data is provided
        if market_data is not None and not market_data.empty:
            # Filter out signals against strong market momentum
            if "Close" in market_data.columns:
                # Calculate 5-day momentum
                market_data["momentum_5d"] = market_data["Close"].pct_change(5)
                
                for date in filtered.index:
                    if date in market_data.index:
                        momentum = market_data.loc[date, "momentum_5d"]
                        signal = filtered.loc[date, "signal"]
                        
                        # If strong momentum against signal, reduce signal strength
                        if (momentum > 0.03 and signal < 0) or (momentum < -0.03 and signal > 0):
                            filtered.loc[date, "strength"] *= 0.5
                            
                            # If strength falls below threshold, remove signal
                            if filtered.loc[date, "strength"] < 0.5:
                                filtered.loc[date, "signal"] = 0
        
        return filtered
