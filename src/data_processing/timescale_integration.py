# Cosmic Market Oracle - TimescaleDB Integration Module

"""
This module provides high-performance time-series data storage and retrieval
using TimescaleDB for the Cosmic Market Oracle project.

It implements specialized schemas and functions for storing and querying:
- Historical market data
- Planetary positions
- Astrological calculations
- Integrated financial-astrological datasets
"""

import os
import logging
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

from src.utils.config import Config
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("timescale_integration")


class TimescaleConnector:
    """Handles connections and operations with TimescaleDB."""
    
    def __init__(self, config_path: str = "config/database_config.yaml"):
        """
        Initialize the TimescaleDB connector.
        
        Args:
            config_path: Path to the database configuration file.
        """
        self.config = Config(config_path)
        self.conn = None
        self.cursor = None
        
        # Connection parameters
        self.db_params = {
            "host": self.config.get("host", "localhost"),
            "port": self.config.get("port", 5432),
            "database": self.config.get("database", "cosmic_oracle"),
            "user": self.config.get("user", "postgres"),
            "password": self.config.get("password", "")
        }
        
        # Table names
        self.market_data_table = self.config.get("market_data_table", "market_data")
        self.planetary_data_table = self.config.get("planetary_data_table", "planetary_data")
        self.integrated_data_table = self.config.get("integrated_data_table", "integrated_data")
        
        # Initialize database if needed
        if self.config.get("auto_init", True):
            self.initialize_database()
    
    def connect(self):
        """Establish connection to the TimescaleDB database."""
        try:
            self.conn = psycopg2.connect(**self.db_params)
            self.cursor = self.conn.cursor()
            logger.info("Connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Error connecting to TimescaleDB: {str(e)}")
            raise
    
    def disconnect(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from TimescaleDB")
        
        self.cursor = None
        self.conn = None
    
    def initialize_database(self):
        """Initialize the database schema if it doesn't exist."""
        try:
            self.connect()
            
            # Create extension if not exists
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            
            # Create market data table
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.market_data_table} (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume BIGINT,
                    adjusted_close DOUBLE PRECISION,
                    market_cap BIGINT,
                    dividend DOUBLE PRECISION,
                    split DOUBLE PRECISION,
                    source VARCHAR(50),
                    PRIMARY KEY (time, symbol)
                );
            """)
            
            # Create planetary data table
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.planetary_data_table} (
                    time TIMESTAMPTZ NOT NULL,
                    planet_id INTEGER NOT NULL,
                    longitude DOUBLE PRECISION,
                    latitude DOUBLE PRECISION,
                    distance DOUBLE PRECISION,
                    longitude_speed DOUBLE PRECISION,
                    latitude_speed DOUBLE PRECISION,
                    distance_speed DOUBLE PRECISION,
                    is_retrograde BOOLEAN,
                    nakshatra INTEGER,
                    nakshatra_degree DOUBLE PRECISION,
                    dignity_state VARCHAR(20),
                    dignity_strength DOUBLE PRECISION,
                    is_combust BOOLEAN,
                    combustion_degree DOUBLE PRECISION,
                    shadbala_ratio DOUBLE PRECISION,
                    financial_strength DOUBLE PRECISION,
                    PRIMARY KEY (time, planet_id)
                );
            """)
            
            # Create integrated data table
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.integrated_data_table} (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    market_trend VARCHAR(20),
                    trend_strength DOUBLE PRECISION,
                    reversal_probability DOUBLE PRECISION,
                    support_level DOUBLE PRECISION,
                    resistance_level DOUBLE PRECISION,
                    bullish_yoga_count INTEGER,
                    bearish_yoga_count INTEGER,
                    volatile_yoga_count INTEGER,
                    moon_nakshatra INTEGER,
                    moon_nakshatra_pada INTEGER,
                    moon_nakshatra_financial VARCHAR(20),
                    current_dasha_lord VARCHAR(20),
                    dasha_end_date TIMESTAMPTZ,
                    PRIMARY KEY (time, symbol)
                );
            """)
            
            # Convert tables to TimescaleDB hypertables
            self.cursor.execute(f"SELECT create_hypertable('{self.market_data_table}', 'time', if_not_exists => TRUE);")
            self.cursor.execute(f"SELECT create_hypertable('{self.planetary_data_table}', 'time', if_not_exists => TRUE);")
            self.cursor.execute(f"SELECT create_hypertable('{self.integrated_data_table}', 'time', if_not_exists => TRUE);")
            
            # Create indexes for efficient queries
            self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.market_data_table}_symbol ON {self.market_data_table} (symbol, time DESC);")
            self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.planetary_data_table}_planet ON {self.planetary_data_table} (planet_id, time DESC);")
            self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.integrated_data_table}_symbol ON {self.integrated_data_table} (symbol, time DESC);")
            
            self.conn.commit()
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error initializing database: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def store_market_data(self, data: pd.DataFrame, symbol: str, source: str = "yahoo"):
        """
        Store market data in TimescaleDB.
        
        Args:
            data: DataFrame containing market data with DatetimeIndex
            symbol: Market symbol (e.g., "^DJI")
            source: Data source (e.g., "yahoo", "bloomberg")
        """
        if data.empty:
            logger.warning(f"No market data to store for {symbol}")
            return
        
        try:
            self.connect()
            
            # Prepare data for insertion
            records = []
            for idx, row in data.iterrows():
                record = (
                    idx,  # time
                    symbol,
                    row.get('Open', None),
                    row.get('High', None),
                    row.get('Low', None),
                    row.get('Close', None),
                    row.get('Volume', None),
                    row.get('Adj Close', None),
                    row.get('Market Cap', None),
                    row.get('Dividend', None),
                    row.get('Split', None),
                    source
                )
                records.append(record)
            
            # Insert data
            query = f"""
                INSERT INTO {self.market_data_table} 
                (time, symbol, open, high, low, close, volume, adjusted_close, market_cap, dividend, split, source)
                VALUES %s
                ON CONFLICT (time, symbol) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    adjusted_close = EXCLUDED.adjusted_close,
                    market_cap = EXCLUDED.market_cap,
                    dividend = EXCLUDED.dividend,
                    split = EXCLUDED.split,
                    source = EXCLUDED.source;
            """
            
            execute_values(self.cursor, query, records)
            self.conn.commit()
            
            logger.info(f"Stored {len(records)} market data records for {symbol}")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing market data: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def store_planetary_data(self, data: pd.DataFrame):
        """
        Store planetary data in TimescaleDB.
        
        Args:
            data: DataFrame containing planetary data with columns:
                - time: Timestamp
                - planet_id: Planet ID
                - longitude, latitude, distance: Positional data
                - is_retrograde: Retrograde status
                - Other planetary attributes
        """
        if data.empty:
            logger.warning("No planetary data to store")
            return
        
        try:
            self.connect()
            
            # Prepare data for insertion
            records = []
            for _, row in data.iterrows():
                record = (
                    row['time'],
                    row['planet_id'],
                    row.get('longitude', None),
                    row.get('latitude', None),
                    row.get('distance', None),
                    row.get('longitude_speed', None),
                    row.get('latitude_speed', None),
                    row.get('distance_speed', None),
                    row.get('is_retrograde', None),
                    row.get('nakshatra', None),
                    row.get('nakshatra_degree', None),
                    row.get('dignity_state', None),
                    row.get('dignity_strength', None),
                    row.get('is_combust', None),
                    row.get('combustion_degree', None),
                    row.get('shadbala_ratio', None),
                    row.get('financial_strength', None)
                )
                records.append(record)
            
            # Insert data
            query = f"""
                INSERT INTO {self.planetary_data_table} 
                (time, planet_id, longitude, latitude, distance, longitude_speed, latitude_speed, 
                distance_speed, is_retrograde, nakshatra, nakshatra_degree, dignity_state, 
                dignity_strength, is_combust, combustion_degree, shadbala_ratio, financial_strength)
                VALUES %s
                ON CONFLICT (time, planet_id) DO UPDATE SET
                    longitude = EXCLUDED.longitude,
                    latitude = EXCLUDED.latitude,
                    distance = EXCLUDED.distance,
                    longitude_speed = EXCLUDED.longitude_speed,
                    latitude_speed = EXCLUDED.latitude_speed,
                    distance_speed = EXCLUDED.distance_speed,
                    is_retrograde = EXCLUDED.is_retrograde,
                    nakshatra = EXCLUDED.nakshatra,
                    nakshatra_degree = EXCLUDED.nakshatra_degree,
                    dignity_state = EXCLUDED.dignity_state,
                    dignity_strength = EXCLUDED.dignity_strength,
                    is_combust = EXCLUDED.is_combust,
                    combustion_degree = EXCLUDED.combustion_degree,
                    shadbala_ratio = EXCLUDED.shadbala_ratio,
                    financial_strength = EXCLUDED.financial_strength;
            """
            
            execute_values(self.cursor, query, records)
            self.conn.commit()
            
            logger.info(f"Stored {len(records)} planetary data records")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing planetary data: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def store_integrated_data(self, data: pd.DataFrame):
        """
        Store integrated market-astrological data in TimescaleDB.
        
        Args:
            data: DataFrame containing integrated data with columns:
                - time: Timestamp
                - symbol: Market symbol
                - market_trend: Predicted market trend
                - Other integrated features
        """
        if data.empty:
            logger.warning("No integrated data to store")
            return
        
        try:
            self.connect()
            
            # Prepare data for insertion
            records = []
            for _, row in data.iterrows():
                record = (
                    row['time'],
                    row['symbol'],
                    row.get('market_trend', None),
                    row.get('trend_strength', None),
                    row.get('reversal_probability', None),
                    row.get('support_level', None),
                    row.get('resistance_level', None),
                    row.get('bullish_yoga_count', None),
                    row.get('bearish_yoga_count', None),
                    row.get('volatile_yoga_count', None),
                    row.get('moon_nakshatra', None),
                    row.get('moon_nakshatra_pada', None),
                    row.get('moon_nakshatra_financial', None),
                    row.get('current_dasha_lord', None),
                    row.get('dasha_end_date', None)
                )
                records.append(record)
            
            # Insert data
            query = f"""
                INSERT INTO {self.integrated_data_table} 
                (time, symbol, market_trend, trend_strength, reversal_probability, 
                support_level, resistance_level, bullish_yoga_count, bearish_yoga_count, 
                volatile_yoga_count, moon_nakshatra, moon_nakshatra_pada, 
                moon_nakshatra_financial, current_dasha_lord, dasha_end_date)
                VALUES %s
                ON CONFLICT (time, symbol) DO UPDATE SET
                    market_trend = EXCLUDED.market_trend,
                    trend_strength = EXCLUDED.trend_strength,
                    reversal_probability = EXCLUDED.reversal_probability,
                    support_level = EXCLUDED.support_level,
                    resistance_level = EXCLUDED.resistance_level,
                    bullish_yoga_count = EXCLUDED.bullish_yoga_count,
                    bearish_yoga_count = EXCLUDED.bearish_yoga_count,
                    volatile_yoga_count = EXCLUDED.volatile_yoga_count,
                    moon_nakshatra = EXCLUDED.moon_nakshatra,
                    moon_nakshatra_pada = EXCLUDED.moon_nakshatra_pada,
                    moon_nakshatra_financial = EXCLUDED.moon_nakshatra_financial,
                    current_dasha_lord = EXCLUDED.current_dasha_lord,
                    dasha_end_date = EXCLUDED.dasha_end_date;
            """
            
            execute_values(self.cursor, query, records)
            self.conn.commit()
            
            logger.info(f"Stored {len(records)} integrated data records")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing integrated data: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def get_market_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieve market data from TimescaleDB.
        
        Args:
            symbol: Market symbol (e.g., "^DJI")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame containing market data
        """
        try:
            self.connect()
            
            query = f"""
                SELECT time, open, high, low, close, volume, adjusted_close
                FROM {self.market_data_table}
                WHERE symbol = %s AND time BETWEEN %s AND %s
                ORDER BY time;
            """
            
            self.cursor.execute(query, (symbol, start_date, end_date))
            columns = [desc[0] for desc in self.cursor.description]
            data = self.cursor.fetchall()
            
            df = pd.DataFrame(data, columns=columns)
            if not df.empty:
                df.set_index('time', inplace=True)
                # Rename columns to match yfinance format
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'adjusted_close': 'Adj Close'
                }, inplace=True)
            
            logger.info(f"Retrieved {len(df)} market data records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving market data: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def get_planetary_data(self, start_date: str, end_date: str, planet_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Retrieve planetary data from TimescaleDB.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            planet_ids: Optional list of planet IDs to filter by
            
        Returns:
            DataFrame containing planetary data
        """
        try:
            self.connect()
            
            if planet_ids:
                query = f"""
                    SELECT *
                    FROM {self.planetary_data_table}
                    WHERE time BETWEEN %s AND %s AND planet_id = ANY(%s)
                    ORDER BY time, planet_id;
                """
                self.cursor.execute(query, (start_date, end_date, planet_ids))
            else:
                query = f"""
                    SELECT *
                    FROM {self.planetary_data_table}
                    WHERE time BETWEEN %s AND %s
                    ORDER BY time, planet_id;
                """
                self.cursor.execute(query, (start_date, end_date))
            
            columns = [desc[0] for desc in self.cursor.description]
            data = self.cursor.fetchall()
            
            df = pd.DataFrame(data, columns=columns)
            
            logger.info(f"Retrieved {len(df)} planetary data records")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving planetary data: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def get_integrated_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieve integrated data from TimescaleDB.
        
        Args:
            symbol: Market symbol (e.g., "^DJI")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame containing integrated data
        """
        try:
            self.connect()
            
            query = f"""
                SELECT *
                FROM {self.integrated_data_table}
                WHERE symbol = %s AND time BETWEEN %s AND %s
                ORDER BY time;
            """
            
            self.cursor.execute(query, (symbol, start_date, end_date))
            columns = [desc[0] for desc in self.cursor.description]
            data = self.cursor.fetchall()
            
            df = pd.DataFrame(data, columns=columns)
            if not df.empty:
                df.set_index('time', inplace=True)
            
            logger.info(f"Retrieved {len(df)} integrated data records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving integrated data: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def run_custom_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Run a custom SQL query against the TimescaleDB.
        
        Args:
            query: SQL query string
            params: Optional tuple of parameters for the query
            
        Returns:
            DataFrame containing query results
        """
        try:
            self.connect()
            
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            columns = [desc[0] for desc in self.cursor.description]
            data = self.cursor.fetchall()
            
            df = pd.DataFrame(data, columns=columns)
            
            logger.info(f"Custom query returned {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error running custom query: {str(e)}")
            raise
        finally:
            self.disconnect()
