# Cosmic Market Oracle - Database Writer Module

"""
This module handles writing processed data and features to the TimescaleDB database.
It provides functions for efficient batch inserts and updates.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from config.database import get_db, engine
from .models import MarketData, PlanetaryData, PlanetaryAspect, MarketPrediction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def write_market_data_to_db(data: pd.DataFrame, batch_size: int = 1000) -> bool:
    """
    Write market data to the database in batches.
    
    Args:
        data: DataFrame with market data
        batch_size: Number of records to insert in each batch
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure required columns are present
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Required column '{col}' not found in market data")
                return False
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Process in batches
        total_records = len(data)
        num_batches = (total_records + batch_size - 1) // batch_size
        
        logger.info(f"Writing {total_records} market data records in {num_batches} batches")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_records)
            batch = data.iloc[start_idx:end_idx]
            
            # Create database session
            with Session(engine) as session:
                # Convert batch to list of model instances
                records = []
                for _, row in batch.iterrows():
                    record = MarketData(
                        timestamp=row['timestamp'],
                        symbol=row['symbol'],
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row.get('volume'),
                        adjusted_close=row.get('adjusted_close'),
                        volatility=row.get('volatility'),
                        rsi=row.get('rsi'),
                        macd=row.get('macd')
                    )
                    records.append(record)
                
                # Add all records to session
                session.add_all(records)
                
                # Commit the batch
                session.commit()
            
            logger.info(f"Batch {i+1}/{num_batches} written successfully")
        
        return True
    
    except SQLAlchemyError as e:
        logger.error(f"Database error writing market data: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error writing market data to database: {str(e)}")
        return False


def write_planetary_data_to_db(data: pd.DataFrame, batch_size: int = 1000) -> bool:
    """
    Write planetary data to the database in batches.
    
    Args:
        data: DataFrame with planetary data
        batch_size: Number of records to insert in each batch
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Process in batches
        total_records = len(data)
        num_batches = (total_records + batch_size - 1) // batch_size
        
        logger.info(f"Writing {total_records} planetary data records in {num_batches} batches")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_records)
            batch = data.iloc[start_idx:end_idx]
            
            # Create database session
            with Session(engine) as session:
                # Convert batch to list of model instances
                records = []
                for _, row in batch.iterrows():
                    # Extract planet data from columns
                    for planet_id in range(0, 12):  # Assuming planets 0-11
                        planet_prefix = f"planet_{planet_id}"
                        
                        # Check if this planet exists in the data
                        if f"{planet_prefix}_longitude" in row:
                            record = PlanetaryData(
                                timestamp=row['timestamp'],
                                planet_id=planet_id,
                                planet_name=get_planet_name(planet_id),
                                longitude=row[f"{planet_prefix}_longitude"],
                                latitude=row.get(f"{planet_prefix}_latitude", 0.0),
                                distance=row.get(f"{planet_prefix}_distance", 0.0),
                                longitude_speed=row.get(f"{planet_prefix}_longitude_speed", 0.0),
                                is_retrograde=row.get(f"{planet_prefix}_is_retrograde", False),
                                nakshatra=row.get(f"{planet_prefix}_nakshatra", 1),
                                nakshatra_degree=row.get(f"{planet_prefix}_nakshatra_degree", 0.0)
                            )
                            records.append(record)
                
                # Add all records to session
                session.add_all(records)
                
                # Commit the batch
                session.commit()
            
            logger.info(f"Batch {i+1}/{num_batches} written successfully")
        
        return True
    
    except SQLAlchemyError as e:
        logger.error(f"Database error writing planetary data: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error writing planetary data to database: {str(e)}")
        return False


def write_aspects_to_db(aspects_data: List[Dict], batch_size: int = 1000) -> bool:
    """
    Write planetary aspects to the database in batches.
    
    Args:
        aspects_data: List of dictionaries with aspect data
        batch_size: Number of records to insert in each batch
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Process in batches
        total_records = len(aspects_data)
        num_batches = (total_records + batch_size - 1) // batch_size
        
        logger.info(f"Writing {total_records} aspect records in {num_batches} batches")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_records)
            batch = aspects_data[start_idx:end_idx]
            
            # Create database session
            with Session(engine) as session:
                # Convert batch to list of model instances
                records = []
                for aspect in batch:
                    record = PlanetaryAspect(
                        timestamp=aspect['timestamp'],
                        planet1_id=aspect['planet1'],
                        planet2_id=aspect['planet2'],
                        aspect_type=aspect['aspect_type'],
                        aspect_angle=aspect['aspect_angle'],
                        actual_angle=aspect['actual_angle'],
                        orb=aspect['orb'],
                        is_applying=aspect['applying']
                    )
                    records.append(record)
                
                # Add all records to session
                session.add_all(records)
                
                # Commit the batch
                session.commit()
            
            logger.info(f"Batch {i+1}/{num_batches} written successfully")
        
        return True
    
    except SQLAlchemyError as e:
        logger.error(f"Database error writing aspect data: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error writing aspect data to database: {str(e)}")
        return False


def write_predictions_to_db(predictions: pd.DataFrame, model_version: str, batch_size: int = 1000) -> bool:
    """
    Write model predictions to the database in batches.
    
    Args:
        predictions: DataFrame with prediction data
        model_version: Version identifier for the model used
        batch_size: Number of records to insert in each batch
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure required columns are present
        required_columns = ['timestamp', 'target_date', 'symbol', 'prediction_value', 'confidence']
        for col in required_columns:
            if col not in predictions.columns:
                logger.error(f"Required column '{col}' not found in predictions data")
                return False
        
        # Process in batches
        total_records = len(predictions)
        num_batches = (total_records + batch_size - 1) // batch_size
        
        logger.info(f"Writing {total_records} prediction records in {num_batches} batches")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_records)
            batch = predictions.iloc[start_idx:end_idx]
            
            # Create database session
            with Session(engine) as session:
                # Convert batch to list of model instances
                records = []
                for _, row in batch.iterrows():
                    record = MarketPrediction(
                        timestamp=row['timestamp'],
                        target_date=row['target_date'],
                        symbol=row['symbol'],
                        prediction_value=row['prediction_value'],
                        confidence=row['confidence'],
                        model_version=model_version
                    )
                    records.append(record)
                
                # Add all records to session
                session.add_all(records)
                
                # Commit the batch
                session.commit()
            
            logger.info(f"Batch {i+1}/{num_batches} written successfully")
        
        return True
    
    except SQLAlchemyError as e:
        logger.error(f"Database error writing prediction data: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error writing prediction data to database: {str(e)}")
        return False


def write_features_to_db(features: pd.DataFrame) -> bool:
    """
    Write generated features to the database.
    This is a placeholder function that would be implemented based on the specific
    feature storage schema.
    
    Args:
        features: DataFrame with feature data
        
    Returns:
        True if successful, False otherwise
    """
    # This would be implemented based on how features are stored in the database
    # For now, just log the feature names
    logger.info(f"Feature columns: {features.columns.tolist()}")
    logger.info(f"Total features: {len(features.columns)}")
    logger.info(f"Total records: {len(features)}")
    
    # In a real implementation, features would be stored in a dedicated table
    # or in a format suitable for model training
    
    return True


def get_planet_name(planet_id: int) -> str:
    """
    Get the name of a planet from its ID.
    
    Args:
        planet_id: Planet ID
        
    Returns:
        Planet name as string
    """
    # This is a simplified version - in production, would import from astro_engine
    planet_names = {
        0: "sun",
        1: "moon",
        2: "mercury",
        3: "venus",
        4: "mars",
        5: "jupiter",
        6: "saturn",
        7: "uranus",
        8: "neptune",
        9: "pluto",
        10: "rahu",
        11: "ketu"
    }
    
    return planet_names.get(planet_id, f"planet_{planet_id}")
