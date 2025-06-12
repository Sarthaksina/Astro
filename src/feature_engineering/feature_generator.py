# Cosmic Market Oracle - Feature Generator Module

"""
This module integrates market and astrological data to generate features for the prediction model.
It combines features from both domains and creates interaction features that capture the
relationships between market conditions and astrological configurations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
# import logging # Removed
from datetime import datetime, timedelta
from src.utils.logger import get_logger # Added

from .astrological_features import AstrologicalFeatureGenerator
from .market_features import MarketFeatureGenerator
from ..astro_engine.planetary_positions import PlanetaryCalculator
from ..astro_engine.constants import get_planet_name, SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Removed
logger = get_logger(__name__) # Changed


class FeatureGenerator:
    """Integrates market and astrological features for model training."""
    
    def __init__(self, 
                 astro_calculator: Optional[PlanetaryCalculator] = None,
                 lookback_periods: List[int] = [1, 5, 10, 20, 50, 200]):
        """
        Initialize the feature generator.
        
        Args:
            astro_calculator: Optional PlanetaryCalculator instance
            lookback_periods: List of lookback periods for market features
        """
        self.astro_feature_generator = AstrologicalFeatureGenerator(calculator=astro_calculator)
        self.market_feature_generator = MarketFeatureGenerator(lookback_periods=lookback_periods)
        self.lookback_periods = lookback_periods
        
    def generate_features(self, integrated_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from integrated market and astrological data.
        
        Args:
            integrated_data: DataFrame with market data and planetary positions
            
        Returns:
            DataFrame with generated features
        """
        logger.info(f"Generating features for {len(integrated_data)} records")
        
        # Create a copy of the data to avoid modifying the original
        df = integrated_data.copy()
        
        # Ensure data is sorted by date
        df = df.sort_values('timestamp')
        
        # Generate market features
        market_features = self.market_feature_generator.generate_features(df)
        
        # Generate astrological features for each date
        dates = df['timestamp'].unique()
        astro_features = self.astro_feature_generator.generate_features_for_dates(dates)
        
        # Merge features
        result = pd.merge(
            market_features,
            astro_features,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Generate interaction features
        result = self._generate_interaction_features(result)
        
        logger.info(f"Generated {result.shape[1]} features for {len(result)} records")
        return result
    
    def _generate_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate interaction features between market and astrological features.
        
        Args:
            features: DataFrame with market and astrological features
            
        Returns:
            DataFrame with additional interaction features
        """
        df = features.copy()
        
        # Select key market features
        market_cols = [
            'daily_return', 'volatility', 'rsi', 'macd',
            'return_5d', 'return_20d', 'volume_change_ratio'
        ]
        
        # Select key astrological features using planet names
        astro_cols = [
            f'{get_planet_name(planet1)}_{get_planet_name(planet2)}_angle'
            for planet1 in [SUN, JUPITER, SATURN]
            for planet2 in [MOON, VENUS, MARS]
            if planet1 < planet2  # Avoid duplicate pairs
        ]
        
        # Add additional important astrological features
        astro_cols.extend([
            'mars_saturn_angle', 'element_fire', 'element_earth',
            'modality_cardinal', 'modality_fixed'
        ])
        
        # Create interaction features for key combinations
        for market_col in market_cols:
            if market_col in df.columns:
                for astro_col in astro_cols:
                    if astro_col in df.columns:
                        # Create interaction feature
                        df[f'interaction_{market_col}_{astro_col}'] = df[market_col] * df[astro_col]
        
        return df


def generate_features(integrated_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to generate features from integrated data.
    
    Args:
        integrated_data: DataFrame with market data and planetary positions
        
    Returns:
        DataFrame with generated features
    """
    generator = FeatureGenerator()
    return generator.generate_features(integrated_data)
