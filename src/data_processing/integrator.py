# Cosmic Market Oracle - Data Integration Module

"""
This module integrates market data with planetary positions for analysis.
It handles the synchronization of different data sources and prepares
the integrated dataset for feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta

from ..astro_engine.planetary_positions import PlanetaryCalculator
from ..astro_engine.constants import get_planet_name
from src.utils.logger import get_logger # Added import

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Removed
logger = get_logger(__name__) # Changed


def integrate_market_astro_data(
    market_data: pd.DataFrame,
    planetary_data: Optional[pd.DataFrame] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Integrate market data with planetary positions.
    
    Args:
        market_data: DataFrame with market data
        planetary_data: Optional DataFrame with pre-calculated planetary data
        start_date: Optional start date for filtering (format: YYYY-MM-DD)
        end_date: Optional end date for filtering (format: YYYY-MM-DD)
        
    Returns:
        DataFrame with integrated market and planetary data
    """
    # Make a copy to avoid modifying the original
    market_df = market_data.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in market_df.columns and not pd.api.types.is_datetime64_any_dtype(market_df['timestamp']):
        market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
    
    # Filter by date range if specified
    if start_date:
        start = pd.to_datetime(start_date)
        market_df = market_df[market_df['timestamp'] >= start]
    
    if end_date:
        end = pd.to_datetime(end_date)
        market_df = market_df[market_df['timestamp'] <= end]
    
    # If planetary data is not provided, calculate it
    if planetary_data is None:
        logger.info("Calculating planetary positions for market dates")
        planetary_df = calculate_planetary_data(market_df['timestamp'].unique())
    else:
        planetary_df = planetary_data.copy()
        # Ensure timestamp is datetime
        if 'timestamp' in planetary_df.columns and not pd.api.types.is_datetime64_any_dtype(planetary_df['timestamp']):
            planetary_df['timestamp'] = pd.to_datetime(planetary_df['timestamp'])
    
    # Merge market data with planetary data
    logger.info("Merging market data with planetary positions")
    integrated_df = pd.merge(
        market_df,
        planetary_df,
        on='timestamp',
        how='inner'
    )
    
    logger.info(f"Integrated dataset contains {len(integrated_df)} records")
    return integrated_df


def calculate_planetary_data(dates: Union[List[datetime], np.ndarray, pd.Series]) -> pd.DataFrame:
    """
    Calculate planetary positions for a list of dates.
    
    Args:
        dates: List or array of dates
        
    Returns:
        DataFrame with planetary positions for each date
    """
    # Initialize planetary calculator
    calculator = PlanetaryCalculator()
    
    # Convert dates to list if necessary
    if isinstance(dates, (np.ndarray, pd.Series)):
        dates = dates.tolist()
    
    # Initialize lists to store results
    results = []
    
    logger.info(f"Calculating planetary positions for {len(dates)} dates")
    
    # Calculate positions for each date
    for date in dates:
        # Get all planet positions
        planets = calculator.get_all_planets(date)
        
        # Create a record for this date
        record = {'timestamp': date}
        
        # Add planetary positions
        for planet_id, position in planets.items():
            planet_name = get_planet_name(planet_id)
            for key, value in position.items():
                record[f"{planet_name}_{key}"] = value
        
        results.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Close calculator to free resources
    calculator.close()
    
    return df


# Removed local get_planet_name function
