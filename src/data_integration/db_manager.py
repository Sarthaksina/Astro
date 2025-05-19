"""
Unified Database Manager for the Cosmic Market Oracle.

This module provides a consolidated interface for database operations,
combining the functionality from both timescale_schema.py and timescale_integration.py
into a single, consistent API.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Tuple
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, ForeignKey, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.schema import CreateSchema
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.utils.config import Config
from src.utils.logger import setup_logger

# Set up logging
logger = setup_logger("db_manager")

# Define base class for SQLAlchemy models
Base = declarative_base()

# Define constants
HYPERTABLE_CHUNK_INTERVAL = "1 month"  # Default chunk interval for hypertables
DEFAULT_RETENTION_PERIOD = "5 years"   # Default data retention period


class DatabaseManager:
    """Unified manager for TimescaleDB database operations."""
    
    def __init__(self, 
                 config_path: str = "config/database_config.yaml",
                 host: str = "localhost", 
                 port: int = 5432, 
                 user: str = "postgres", 
                 password: str = "postgres", 
                 database: str = "cosmic_oracle",
                 schema: str = "cosmic_data"):
        """
        Initialize the database manager.
        
        Args:
            config_path: Path to the database configuration file (optional)
            host: Database host
            port: Database port
            user: Database user
            password: Database password
            database: Database name
            schema: Schema name
        """
        # Try to load from config file first
        self.config = Config(config_path) if os.path.exists(config_path) else None
        
        # Set connection parameters, prioritizing config file if available
        self.host = self.config.get("host", host) if self.config else host
        self.port = self.config.get("port", port) if self.config else port
        self.user = self.config.get("user", user) if self.config else user
        self.password = self.config.get("password", password) if self.config else password
        self.database = self.config.get("database", database) if self.config else database
        self.schema = self.config.get("schema", schema) if self.config else schema
        
        # SQLAlchemy specific
        self.engine = None
        self.metadata = MetaData(schema=schema)
        self.Session = None
        
        # psycopg2 specific
        self.conn = None
        self.cursor = None
        
        # Table names
        self.market_data_table = self.config.get("market_data_table", "financial_data") if self.config else "financial_data"
        self.planetary_data_table = self.config.get("planetary_data_table", "planetary_positions") if self.config else "planetary_positions"
        self.integrated_data_table = self.config.get("integrated_data_table", "integrated_data") if self.config else "integrated_data"
        
        # Initialize database if needed
        if self.config and self.config.get("auto_init", True):
            self.create_database()
            self.connect_sqlalchemy()
            self.create_tables()
        
    def create_database(self) -> None:
        """Create the database if it doesn't exist."""
        # Connect to default database to create new database
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.database,))
        exists = cursor.fetchone()
        
        if not exists:
            logger.info(f"Creating database {self.database}")
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.database)))
            logger.info(f"Database {self.database} created successfully")
        else:
            logger.info(f"Database {self.database} already exists")
            
        cursor.close()
        conn.close()
        
    def connect_sqlalchemy(self) -> None:
        """Connect to the database using SQLAlchemy and create schema if it doesn't exist."""
        # Create connection string
        conn_str = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        # Create engine
        self.engine = create_engine(conn_str)
        
        # Create session factory
        self.Session = sessionmaker(bind=self.engine)
        
        # Create schema if it doesn't exist
        with self.engine.connect() as conn:
            if not conn.dialect.has_schema(conn, self.schema):
                logger.info(f"Creating schema {self.schema}")
                conn.execute(CreateSchema(self.schema))
                logger.info(f"Schema {self.schema} created successfully")
            else:
                logger.info(f"Schema {self.schema} already exists")
                
        # Create TimescaleDB extension if it doesn't exist
        with self.engine.connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            logger.info("TimescaleDB extension created or already exists")
            
        logger.info(f"Connected to database {self.database} using SQLAlchemy")
    
    def connect_psycopg2(self) -> None:
        """Establish connection to the TimescaleDB database using psycopg2."""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.conn.cursor()
            logger.info("Connected to TimescaleDB using psycopg2")
        except Exception as e:
            logger.error(f"Error connecting to TimescaleDB: {str(e)}")
            raise
    
    def disconnect_psycopg2(self) -> None:
        """Close the psycopg2 database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from TimescaleDB")
        
        self.cursor = None
        self.conn = None
    
    def create_tables(self) -> None:
        """Create all tables defined in the models."""
        Base.metadata.create_all(self.engine)
        logger.info("Created all tables")
        
        # Convert tables to hypertables
        self._convert_to_hypertables()
    
    def _convert_to_hypertables(self) -> None:
        """Convert regular tables to TimescaleDB hypertables."""
        # Connect using psycopg2 for direct SQL execution
        self.connect_psycopg2()
        
        try:
            # Get all table models that should be hypertables (those with timestamp column)
            hypertable_models = [
                model for model in Base.__subclasses__() 
                if hasattr(model, '__tablename__') and 
                any(column.name == 'timestamp' for column in model.__table__.columns)
            ]
            
            for model in hypertable_models:
                table_name = model.__tablename__
                schema_name = model.__table_args__.get('schema', self.schema)
                qualified_table_name = f"{schema_name}.{table_name}"
                
                # Check if table is already a hypertable
                self.cursor.execute("""
                    SELECT 1 FROM timescaledb_information.hypertables
                    WHERE hypertable_schema = %s AND hypertable_name = %s
                """, (schema_name, table_name))
                
                is_hypertable = self.cursor.fetchone() is not None
                
                if not is_hypertable:
                    # Convert to hypertable
                    self.cursor.execute(f"""
                        SELECT create_hypertable(
                            '{qualified_table_name}', 'timestamp',
                            chunk_time_interval => INTERVAL '{HYPERTABLE_CHUNK_INTERVAL}'
                        );
                    """)
                    
                    # Set retention policy
                    self.cursor.execute(f"""
                        SELECT add_retention_policy(
                            '{qualified_table_name}',
                            INTERVAL '{DEFAULT_RETENTION_PERIOD}'
                        );
                    """)
                    
                    self.conn.commit()
                    logger.info(f"Converted {qualified_table_name} to hypertable with retention policy")
                else:
                    logger.info(f"Table {qualified_table_name} is already a hypertable")
                    
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error converting tables to hypertables: {str(e)}")
            raise
        finally:
            self.disconnect_psycopg2()
    
    def drop_tables(self) -> None:
        """Drop all tables."""
        Base.metadata.drop_all(self.engine)
        logger.info("Dropped all tables")
    
    # Data storage methods
    
    def store_market_data(self, data: pd.DataFrame, symbol: str, source: str = "yahoo") -> None:
        """
        Store market data in the database.
        
        Args:
            data: DataFrame containing market data with DatetimeIndex
            symbol: Market symbol (e.g., "^DJI")
            source: Data source (e.g., "yahoo", "bloomberg")
        """
        if data.empty:
            logger.warning("No market data to store")
            return
            
        # Ensure data has a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("Market data must have a DatetimeIndex")
            raise ValueError("Market data must have a DatetimeIndex")
            
        # Create a session
        session = self.Session()
        
        try:
            # Convert DataFrame to list of FinancialData objects
            financial_data_objects = []
            
            for timestamp, row in data.iterrows():
                financial_data = FinancialData(
                    timestamp=timestamp,
                    symbol=symbol,
                    open=row.get('open', None),
                    high=row.get('high', None),
                    low=row.get('low', None),
                    close=row.get('close', None),
                    volume=row.get('volume', None),
                    adjusted_close=row.get('adj_close', None) or row.get('adjusted_close', None),
                    source=source,
                    metadata={}  # Additional metadata can be added here
                )
                financial_data_objects.append(financial_data)
                
            # Add all objects to the session
            session.add_all(financial_data_objects)
            
            # Commit the transaction
            session.commit()
            
            logger.info(f"Stored {len(financial_data_objects)} market data records for {symbol}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing market data: {str(e)}")
            raise
        finally:
            session.close()
    
    def store_planetary_data(self, data: pd.DataFrame) -> None:
        """
        Store planetary data in the database.
        
        Args:
            data: DataFrame containing planetary data with columns:
                - timestamp: Timestamp
                - planet_id: Planet ID
                - longitude, latitude, distance: Positional data
                - is_retrograde: Retrograde status
                - Other planetary attributes
        """
        if data.empty:
            logger.warning("No planetary data to store")
            return
            
        # Create a session
        session = self.Session()
        
        try:
            # Convert DataFrame to list of PlanetaryPosition objects
            planetary_position_objects = []
            
            for _, row in data.iterrows():
                planetary_position = PlanetaryPosition(
                    timestamp=row['timestamp'],
                    planet_id=row['planet_id'],
                    longitude=row['longitude'],
                    latitude=row.get('latitude', 0.0),
                    distance=row.get('distance', 0.0),
                    speed=row.get('speed', 0.0),
                    is_retrograde=row.get('is_retrograde', False),
                    sign=row.get('sign', 0),
                    nakshatra=row.get('nakshatra', 0),
                    house=row.get('house', None),
                    dignity=row.get('dignity', None),
                    aspects=row.get('aspects', {})
                )
                planetary_position_objects.append(planetary_position)
                
            # Add all objects to the session
            session.add_all(planetary_position_objects)
            
            # Commit the transaction
            session.commit()
            
            logger.info(f"Stored {len(planetary_position_objects)} planetary data records")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing planetary data: {str(e)}")
            raise
        finally:
            session.close()
    
    # Data retrieval methods
    
    def get_market_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieve market data from the database.
        
        Args:
            symbol: Market symbol (e.g., "^DJI")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame containing market data
        """
        # Create a session
        session = self.Session()
        
        try:
            # Query financial data
            query = session.query(FinancialData).filter(
                FinancialData.symbol == symbol,
                FinancialData.timestamp >= start_date,
                FinancialData.timestamp <= end_date
            ).order_by(FinancialData.timestamp)
            
            # Convert query results to DataFrame
            results = query.all()
            
            if not results:
                logger.warning(f"No market data found for {symbol} between {start_date} and {end_date}")
                return pd.DataFrame()
                
            # Create DataFrame from results
            data = []
            for result in results:
                data.append({
                    'timestamp': result.timestamp,
                    'open': result.open,
                    'high': result.high,
                    'low': result.low,
                    'close': result.close,
                    'volume': result.volume,
                    'adjusted_close': result.adjusted_close,
                    'source': result.source
                })
                
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Retrieved {len(df)} market data records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving market data: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_planetary_data(self, start_date: str, end_date: str, planet_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Retrieve planetary data from the database.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            planet_ids: Optional list of planet IDs to filter by
            
        Returns:
            DataFrame containing planetary data
        """
        # Create a session
        session = self.Session()
        
        try:
            # Build query
            query = session.query(PlanetaryPosition).filter(
                PlanetaryPosition.timestamp >= start_date,
                PlanetaryPosition.timestamp <= end_date
            )
            
            # Add planet filter if specified
            if planet_ids:
                query = query.filter(PlanetaryPosition.planet_id.in_(planet_ids))
                
            # Order by timestamp and planet_id
            query = query.order_by(PlanetaryPosition.timestamp, PlanetaryPosition.planet_id)
            
            # Execute query
            results = query.all()
            
            if not results:
                logger.warning(f"No planetary data found between {start_date} and {end_date}")
                return pd.DataFrame()
                
            # Create DataFrame from results
            data = []
            for result in results:
                data.append({
                    'timestamp': result.timestamp,
                    'planet_id': result.planet_id,
                    'longitude': result.longitude,
                    'latitude': result.latitude,
                    'distance': result.distance,
                    'speed': result.speed,
                    'is_retrograde': result.is_retrograde,
                    'sign': result.sign,
                    'nakshatra': result.nakshatra,
                    'house': result.house,
                    'dignity': result.dignity,
                    'aspects': result.aspects
                })
                
            df = pd.DataFrame(data)
            
            logger.info(f"Retrieved {len(df)} planetary data records")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving planetary data: {str(e)}")
            raise
        finally:
            session.close()
    
    def run_custom_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Run a custom SQL query against the database.
        
        Args:
            query: SQL query string
            params: Optional tuple of parameters for the query
            
        Returns:
            DataFrame containing query results
        """
        self.connect_psycopg2()
        
        try:
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
            self.disconnect_psycopg2()


# Define SQLAlchemy models

class PlanetaryPosition(Base):
    """Model for planetary positions."""
    
    __tablename__ = "planetary_positions"
    __table_args__ = {"schema": "cosmic_data"}
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    planet_id = Column(Integer, nullable=False, index=True)
    longitude = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)
    distance = Column(Float, nullable=False)
    speed = Column(Float, nullable=False)
    is_retrograde = Column(Boolean, nullable=False)
    sign = Column(Integer, nullable=False)
    nakshatra = Column(Integer, nullable=False)
    house = Column(Integer, nullable=True)
    dignity = Column(String(50), nullable=True)
    aspects = Column(JSONB, nullable=True)
    
    def __repr__(self):
        return f"<PlanetaryPosition(timestamp={self.timestamp}, planet_id={self.planet_id})>"


class FinancialData(Base):
    """Model for financial data."""
    
    __tablename__ = "financial_data"
    __table_args__ = {"schema": "cosmic_data"}
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    adjusted_close = Column(Float, nullable=True)
    source = Column(String(50), nullable=False)
    metadata = Column(JSONB, nullable=True)
    
    def __repr__(self):
        return f"<FinancialData(timestamp={self.timestamp}, symbol={self.symbol})>"


class MarketRegime(Base):
    """Model for market regime labels."""
    
    __tablename__ = "market_regimes"
    __table_args__ = {"schema": "cosmic_data"}
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    regime_hmm = Column(Integer, nullable=True)
    regime_kmeans = Column(Integer, nullable=True)
    regime_rule_based = Column(Integer, nullable=True)
    regime_consensus = Column(Integer, nullable=True)
    volatility = Column(Float, nullable=True)
    trend = Column(Float, nullable=True)
    metadata = Column(JSONB, nullable=True)
    
    def __repr__(self):
        return f"<MarketRegime(timestamp={self.timestamp}, symbol={self.symbol})>"


class AstrologicalEvent(Base):
    """Model for significant astrological events."""
    
    __tablename__ = "astrological_events"
    __table_args__ = {"schema": "cosmic_data"}
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    planets_involved = Column(JSONB, nullable=False)
    description = Column(String(500), nullable=False)
    strength = Column(Float, nullable=True)
    market_impact = Column(String(50), nullable=True)
    metadata = Column(JSONB, nullable=True)
    
    def __repr__(self):
        return f"<AstrologicalEvent(timestamp={self.timestamp}, event_type={self.event_type})>"


class TechnicalIndicator(Base):
    """Model for technical indicators."""
    
    __tablename__ = "technical_indicators"
    __table_args__ = {"schema": "cosmic_data"}
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    indicator_type = Column(String(50), nullable=False, index=True)
    value = Column(Float, nullable=False)
    parameters = Column(JSONB, nullable=True)
    metadata = Column(JSONB, nullable=True)
    
    def __repr__(self):
        return f"<TechnicalIndicator(timestamp={self.timestamp}, symbol={self.symbol}, indicator_type={self.indicator_type})>"


class Prediction(Base):
    """Model for model predictions."""
    
    __tablename__ = "predictions"
    __table_args__ = {"schema": "cosmic_data"}
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    model_id = Column(String(100), nullable=False, index=True)
    prediction_type = Column(String(50), nullable=False)
    target_timestamp = Column(DateTime, nullable=False)
    value = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    features_used = Column(JSONB, nullable=True)
    metadata = Column(JSONB, nullable=True)
    
    def __repr__(self):
        return f"<Prediction(timestamp={self.timestamp}, symbol={self.symbol}, model_id={self.model_id})>"


class DataSyncLog(Base):
    """Model for tracking data synchronization."""
    
    __tablename__ = "data_sync_log"
    __table_args__ = {"schema": "cosmic_data"}
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    data_type = Column(String(50), nullable=False, index=True)
    source = Column(String(100), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    records_processed = Column(Integer, nullable=False)
    success = Column(Boolean, nullable=False)
    error_message = Column(String(500), nullable=True)
    metadata = Column(JSONB, nullable=True)
    
    def __repr__(self):
        return f"<DataSyncLog(timestamp={self.timestamp}, data_type={self.data_type})>"


# Example usage
if __name__ == "__main__":
    # Create database manager
    db_manager = DatabaseManager(
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        database="cosmic_oracle",
        schema="cosmic_data"
    )
    
    # Create database and connect
    db_manager.create_database()
    db_manager.connect_sqlalchemy()
    
    # Create tables
    db_manager.create_tables()
    
    logger.info("Database setup completed successfully")
