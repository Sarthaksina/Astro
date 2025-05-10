#!/usr/bin/env python3
"""
TimescaleDB Connector Usage Example

This module demonstrates how to use the TimescaleConnector class for the Cosmic Market Oracle project.
It shows how to set up the database, insert data, create optimized views, and perform correlation analysis
between market data and planetary positions.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.utils.db.timescale_connector import get_timescale_connector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("timescale_example")


def generate_sample_market_data(days=30, symbol="DJI"):
    """Generate sample market data for demonstration purposes.

    Args:
        days: Number of days of data to generate.
        symbol: Market symbol to use.

    Returns:
        DataFrame containing sample market data.
    """
    # Create date range
    end_date = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Generate random market data with a trend
    np.random.seed(42)  # For reproducibility
    close_prices = np.linspace(30000, 32000, len(dates)) + np.random.normal(0, 500, len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": symbol,
        "open": close_prices * 0.99,
        "high": close_prices * 1.02,
        "low": close_prices * 0.98,
        "close": close_prices,
        "volume": np.random.randint(1000000, 5000000, len(dates)),
        "source": "example"
    })
    
    return df


def generate_sample_planetary_data(days=30):
    """Generate sample planetary position data for demonstration purposes.

    Args:
        days: Number of days of data to generate.

    Returns:
        DataFrame containing sample planetary position data.
    """
    # Create date range
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # List of planets
    planets = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
    
    # Generate data for each planet
    data = []
    for planet in planets:
        # Different speeds and starting positions for each planet
        if planet == "Moon":
            speed = 13.0  # degrees per day
            start_pos = 0
        elif planet in ["Mercury", "Venus"]:
            speed = 1.0
            start_pos = 30 if planet == "Mercury" else 45
        elif planet == "Sun":
            speed = 1.0
            start_pos = 15
        elif planet == "Mars":
            speed = 0.5
            start_pos = 60
        elif planet == "Jupiter":
            speed = 0.08
            start_pos = 120
        elif planet == "Saturn":
            speed = 0.03
            start_pos = 180
        else:  # Outer planets
            speed = 0.01
            start_pos = 200 + planets.index(planet) * 20
        
        # Calculate positions
        for i, date in enumerate(dates):
            longitude = (start_pos + i * speed) % 360
            is_retrograde = False
            
            # Add some retrograde periods for certain planets
            if planet in ["Mercury", "Venus", "Mars", "Jupiter", "Saturn"] and i % 10 < 3:
                is_retrograde = True
                speed = -speed
            
            data.append({
                "timestamp": date,
                "planet": planet,
                "longitude": longitude,
                "latitude": np.random.uniform(-5, 5),
                "speed": speed * (-1 if is_retrograde else 1),
                "house": int(longitude / 30) + 1,
                "sign": int(longitude / 30) + 1,
                "nakshatra": int(longitude / (360/27)) + 1,
                "is_retrograde": is_retrograde
            })
    
    return pd.DataFrame(data)


def generate_sample_features(days=30):
    """Generate sample feature data for demonstration purposes.

    Args:
        days: Number of days of data to generate.

    Returns:
        DataFrame containing sample feature data.
    """
    # Create date range
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # List of features
    features = [
        {"name": "jupiter_saturn_angle", "category": "planetary_aspect"},
        {"name": "mars_sun_angle", "category": "planetary_aspect"},
        {"name": "moon_phase", "category": "lunar"},
        {"name": "mercury_retrograde_intensity", "category": "retrograde"},
        {"name": "venus_mars_harmony", "category": "planetary_aspect"}
    ]
    
    # Generate data for each feature
    data = []
    for feature in features:
        for date in dates:
            # Generate a cyclical pattern with some noise
            day_of_year = date.dayofyear
            if feature["name"] == "moon_phase":
                # Moon phase cycles every 29.5 days
                cycle = np.sin(2 * np.pi * day_of_year / 29.5)
            elif feature["name"] == "mercury_retrograde_intensity":
                # Mercury retrograde intensity (higher during retrograde periods)
                cycle = np.sin(2 * np.pi * day_of_year / 88) * 0.5 + 0.5
            elif feature["name"] == "jupiter_saturn_angle":
                # Jupiter-Saturn angle changes slowly
                cycle = np.sin(2 * np.pi * day_of_year / 365) * 0.7 + 0.3
            else:
                # Other features with different cycles
                cycle = np.sin(2 * np.pi * day_of_year / (180 + features.index(feature) * 30))
            
            # Add some noise
            value = cycle + np.random.normal(0, 0.1)
            
            data.append({
                "timestamp": date,
                "feature_name": feature["name"],
                "feature_value": value,
                "feature_category": feature["category"]
            })
    
    return pd.DataFrame(data)


def main():
    """Main function to demonstrate TimescaleConnector usage."""
    # Get connector (using default settings for local development)
    connector = get_timescale_connector()
    
    try:
        # Connect to the database
        logger.info("Connecting to TimescaleDB...")
        connector.connect()
        
        # Set up the complete environment
        logger.info("Setting up TimescaleDB environment...")
        connector.setup_complete_timescale_environment()
        
        # Generate and insert sample data
        days_of_data = 90  # Generate 90 days of data
        
        # Market data
        logger.info("Generating and inserting sample market data...")
        market_data = generate_sample_market_data(days=days_of_data)
        inserted_rows = connector.insert_market_data(market_data)
        logger.info(f"Inserted {inserted_rows} rows of market data")
        
        # Planetary positions
        logger.info("Generating and inserting sample planetary positions...")
        planetary_data = generate_sample_planetary_data(days=days_of_data)
        inserted_rows = connector.insert_planetary_positions(planetary_data)
        logger.info(f"Inserted {inserted_rows} rows of planetary positions")
        
        # Features
        logger.info("Generating and inserting sample features...")
        feature_data = generate_sample_features(days=days_of_data)
        inserted_rows = connector.insert_features(feature_data)
        logger.info(f"Inserted {inserted_rows} rows of features")
        
        # Optimize storage with compression
        logger.info("Optimizing storage with compression...")
        connector.enable_compression("market_data", compress_after="30 days")
        connector.enable_compression("planetary_positions", compress_after="30 days")
        connector.enable_compression("features", compress_after="30 days")
        
        # Set retention policies
        logger.info("Setting retention policies...")
        connector.set_retention_policy("market_data", drop_after="2 years")
        connector.set_retention_policy("planetary_positions", drop_after="2 years")
        connector.set_retention_policy("features", drop_after="2 years")
        
        # Optimize chunk time intervals
        logger.info("Optimizing chunk time intervals...")
        connector.optimize_chunk_time_interval("market_data", new_interval="7 days")
        connector.optimize_chunk_time_interval("planetary_positions", new_interval="7 days")
        connector.optimize_chunk_time_interval("features", new_interval="7 days")
        
        # Create synchronized views for integrated analysis
        logger.info("Creating synchronized views...")
        connector.create_synchronized_view("market_planetary_daily", interval="1 day")
        connector.create_synchronized_view("market_planetary_weekly", interval="1 week")
        connector.create_market_planetary_features_view("cosmic_market_integrated_daily", interval="1 day")
        
        # Perform correlation analysis
        logger.info("Performing correlation analysis...")
        planets_to_analyze = ["Jupiter", "Saturn", "Mars", "Mercury"]
        for planet in planets_to_analyze:
            connector.create_correlation_analysis(
                output_table="market_planetary_correlations",
                market_symbol="DJI",
                planet=planet,
                window="60 days"
            )
            logger.info(f"Created correlation analysis for {planet}")
        
        # Query and display some results
        logger.info("Querying market data...")
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        market_results = connector.query_market_data("DJI", start_date, end_date)
        logger.info(f"Retrieved {len(market_results)} rows of market data")
        
        logger.info("Querying planetary positions...")
        planetary_results = connector.query_planetary_positions(
            ["Jupiter", "Saturn"], start_date, end_date
        )
        logger.info(f"Retrieved {len(planetary_results)} rows of planetary positions")
        
        logger.info("Querying features...")
        feature_results = connector.query_features(
            ["jupiter_saturn_angle", "moon_phase"], start_date, end_date
        )
        logger.info(f"Retrieved {len(feature_results)} rows of features")
        
        logger.info("TimescaleDB connector demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in TimescaleDB connector demonstration: {e}")
    finally:
        # Always disconnect
        connector.disconnect()
        logger.info("Disconnected from TimescaleDB")


if __name__ == "__main__":
    main()