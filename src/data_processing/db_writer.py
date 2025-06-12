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

# Updated imports to use consolidated database configuration
# from config.database import get_db, get_engine # Base removed # Already removed get_db
from src.data_integration.db_manager import (
    FinancialData as MarketData,
    PlanetaryPosition as PlanetaryData,
    PlanetaryAspect,
    Prediction as MarketPrediction,
    DatabaseManager # Added DatabaseManager
)
# Removed: from .models import MarketData, PlanetaryData, PlanetaryAspect, MarketPrediction

# Configure logging
from src.utils.logger import get_logger # Changed from setup_logger
logger = get_logger(__name__) # Changed from setup_logger

# Instantiate DatabaseManager
# engine = get_engine() # Removed
db_manager = DatabaseManager() # Added

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
            session = db_manager.Session()
            try:
                # Convert batch to list of model instances
                records = []
                for _, row in batch.iterrows():
                    record = MarketData( # FinancialData model from db_manager
                        timestamp=row['timestamp'],
                        symbol=row['symbol'],
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row.get('volume'),
                        adjusted_close=row.get('adjusted_close'),
                        # Volatility, rsi, macd are not in db_manager.FinancialData
                        # They were in data_processing.models.MarketData.
                        # This part of data model is lost if not added to FinancialData.
                        # For now, removing them based on current FinancialData structure.
                        # volatility=row.get('volatility'),
                        # rsi=row.get('rsi'),
                        # macd=row.get('macd')
                        source=row.get('source', 'unknown'), # FinancialData has 'source'
                        metadata=row.get('metadata', {})   # FinancialData has 'metadata'
                    )
                    records.append(record)
                
                session.add_all(records)
                session.commit()
                logger.info(f"Batch {i+1}/{num_batches} written successfully")
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Database error writing market data: {str(e)}")
                return False # Propagate failure for the batch
            except Exception as e:
                session.rollback()
                logger.error(f"Error writing market data batch: {str(e)}")
                return False # Propagate failure for the batch
            finally:
                session.close()
        
        return True
    
    # Outer exception handling remains the same for non-batch related errors
    except Exception as e:
        logger.error(f"General error in write_market_data_to_db: {str(e)}")
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
            session = db_manager.Session()
            try:
                # Convert batch to list of model instances
                records = []
                for _, row in batch.iterrows():
                    # Extract planet data from columns
                    # Assuming row directly contains fields for PlanetaryPosition model
                    # The old logic iterated 0-11 for planet_id and constructed prefix.
                    # The new PlanetaryPosition model in db_manager expects direct fields.
                    # This function needs to be called with data already structured for PlanetaryPosition.
                    # If the input 'data' DataFrame is flat (planet_0_longitude etc), it needs preprocessing
                    # *before* calling this writer function, or this function needs to be smarter.
                    # For now, assuming 'row' has direct fields for PlanetaryPosition.
                    # If 'planet_id' is not in row, this will fail.
                    # This function's call signature might need to change or data pre-formatted.
                    # The original code was trying to infer multiple planet records from one row.
                    # The new model PlanetaryPosition expects one record per planet.
                    # This means the input `data` DataFrame needs to be pre-processed
                    # to have one row per planet per timestamp.
                    # For now, I will assume 'data' is already in the correct format.
                    record = PlanetaryData( # PlanetaryPosition model from db_manager
                        timestamp=row['timestamp'],
                        planet_id=row['planet_id'], # Assumes this column exists
                        longitude=row['longitude'],
                        latitude=row.get('latitude', 0.0),
                        distance=row.get('distance', 0.0),
                        speed=row.get('longitude_speed', 0.0), # map longitude_speed to speed
                        is_retrograde=row.get('is_retrograde', False),
                        sign=row.get('sign', 0), # PlanetaryPosition has 'sign'
                        nakshatra=row.get('nakshatra', 0), # PlanetaryPosition has 'nakshatra'
                        house=row.get('house'), # PlanetaryPosition has 'house'
                        dignity=row.get('dignity'), # PlanetaryPosition has 'dignity'
                        aspects=row.get('aspects', {}) # PlanetaryPosition has 'aspects'
                    )
                    records.append(record)
                
                session.add_all(records)
                session.commit()
                logger.info(f"Batch {i+1}/{num_batches} written successfully")
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Database error writing planetary data: {str(e)}")
                return False # Propagate failure for the batch
            except Exception as e:
                session.rollback()
                logger.error(f"Error writing planetary data batch: {str(e)}")
                return False # Propagate failure for the batch
            finally:
                session.close()
        
        return True

    # Outer exception handling remains the same
    except Exception as e:
        logger.error(f"General error in write_planetary_data_to_db: {str(e)}")
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
            session = db_manager.Session()
            try:
                # Convert batch to list of model instances
                records = []
                for aspect in batch: # Assuming batch is a list of dicts
                    record = PlanetaryAspect( # PlanetaryAspect model from db_manager
                        timestamp=aspect['timestamp'],
                        planet1_id=aspect['planet1_id'], # schema uses planet1_id
                        planet2_id=aspect['planet2_id'], # schema uses planet2_id
                        aspect_type=aspect['aspect_type'],
                        aspect_angle=aspect['aspect_angle'],
                        actual_angle=aspect['actual_angle'],
                        orb=aspect['orb'],
                        is_applying=aspect['is_applying']
                    )
                    records.append(record)
                
                session.add_all(records)
                session.commit()
                logger.info(f"Batch {i+1}/{num_batches} written successfully")
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Database error writing aspect data: {str(e)}")
                return False # Propagate failure for the batch
            except Exception as e:
                session.rollback()
                logger.error(f"Error writing aspect data batch: {str(e)}")
                return False # Propagate failure for the batch
            finally:
                session.close()
        
        return True

    # Outer exception handling remains the same
    except Exception as e:
        logger.error(f"General error in write_aspects_to_db: {str(e)}")
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
            session = db_manager.Session()
            try:
                # Convert batch to list of model instances
                records = []
                for _, row in batch.iterrows():
                    record = MarketPrediction( # Prediction model from db_manager
                        timestamp=row['timestamp'],
                        target_timestamp=row['target_date'], # db_manager.Prediction uses target_timestamp
                        symbol=row['symbol'],
                        value=row['prediction_value'],    # db_manager.Prediction uses value
                        confidence=row['confidence'],
                        model_id=model_version, # db_manager.Prediction uses model_id
                        model_version=model_version, # Also add to new model_version field
                        # prediction_type needs to be sourced or defaulted
                        prediction_type=row.get('prediction_type', 'unknown')
                    )
                    records.append(record)
                
                session.add_all(records)
                session.commit()
                logger.info(f"Batch {i+1}/{num_batches} written successfully")
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Database error writing prediction data: {str(e)}")
                return False # Propagate failure for the batch
            except Exception as e:
                session.rollback()
                logger.error(f"Error writing prediction data batch: {str(e)}")
                return False # Propagate failure for the batch
            finally:
                session.close()

        return True

    # Outer exception handling remains the same
    except Exception as e:
        logger.error(f"General error in write_predictions_to_db: {str(e)}")
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


# Removed local get_planet_name function, should be imported from constants if needed by this module's logic,
# but it's not used by the refactored writer functions directly anymore.
# The PlanetaryData model itself doesn't store planet_name.
