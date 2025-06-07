#!/usr/bin/env python3
"""
TimescaleDB Connector for Cosmic Market Oracle

This module provides utilities for connecting to TimescaleDB and managing time-series data
for the Cosmic Market Oracle project. It includes functions for creating hypertables,
optimizing queries, and handling time-series specific operations.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Configure logging
from src.utils.logging_config import setup_logging
logger = setup_logging(__name__)


class TimescaleConnector:
    """Connector for TimescaleDB operations."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "cosmic_market_oracle",
        user: str = "cosmic",
        password: str = "cosmic_password",
        schema: str = "public"
    ):
        """Initialize the TimescaleDB connector.

        Args:
            host: Database host.
            port: Database port.
            database: Database name.
            user: Database user.
            password: Database password.
            schema: Database schema.
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.schema = schema
        self.engine = None
        self.conn = None

    def connect(self) -> None:
        """Connect to the TimescaleDB database."""
        try:
            # Create SQLAlchemy engine
            connection_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            self.engine = create_engine(connection_string)
            
            # Create psycopg2 connection for operations not supported by SQLAlchemy
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            
            logger.info(f"Connected to TimescaleDB at {self.host}:{self.port}/{self.database}")
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from the TimescaleDB database."""
        if self.conn:
            self.conn.close()
            self.conn = None
        
        if self.engine:
            self.engine.dispose()
            self.engine = None
        
        logger.info("Disconnected from TimescaleDB")

    def create_hypertable(
        self,
        table_name: str,
        time_column: str = "timestamp",
        chunk_time_interval: str = "1 day",
        if_not_exists: bool = True
    ) -> bool:
        """Convert a regular table to a TimescaleDB hypertable.

        Args:
            table_name: Name of the table to convert.
            time_column: Name of the time column.
            chunk_time_interval: Interval for chunking data.
            if_not_exists: Only create if the hypertable doesn't exist.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with self.conn.cursor() as cur:
                # Create the hypertable
                query = sql.SQL("SELECT create_hypertable({}, {}, if_not_exists => {}, chunk_time_interval => {})").format(
                    sql.Literal(table_name),
                    sql.Literal(time_column),
                    sql.Literal(if_not_exists),
                    sql.Literal(chunk_time_interval)
                )
                cur.execute(query)
                self.conn.commit()
                
                logger.info(f"Created hypertable {table_name} with time column {time_column}")
                return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to create hypertable {table_name}: {e}")
            return False

    def create_market_data_table(self) -> bool:
        """Create the market_data table for storing historical market data.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with self.conn.cursor() as cur:
                # Create the table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume BIGINT,
                    source VARCHAR(50),
                    PRIMARY KEY (timestamp, symbol)
                )
                """)
                self.conn.commit()
                
                # Convert to hypertable
                return self.create_hypertable("market_data", "timestamp")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to create market_data table: {e}")
            return False

    def create_planetary_positions_table(self) -> bool:
        """Create the planetary_positions table for storing astrological data.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with self.conn.cursor() as cur:
                # Create the table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS planetary_positions (
                    timestamp TIMESTAMPTZ NOT NULL,
                    planet VARCHAR(20) NOT NULL,
                    longitude DOUBLE PRECISION,
                    latitude DOUBLE PRECISION,
                    speed DOUBLE PRECISION,
                    house INT,
                    sign INT,
                    nakshatra INT,
                    is_retrograde BOOLEAN,
                    PRIMARY KEY (timestamp, planet)
                )
                """)
                self.conn.commit()
                
                # Convert to hypertable
                return self.create_hypertable("planetary_positions", "timestamp")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to create planetary_positions table: {e}")
            return False

    def create_features_table(self) -> bool:
        """Create the features table for storing generated features.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with self.conn.cursor() as cur:
                # Create the table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    timestamp TIMESTAMPTZ NOT NULL,
                    feature_name VARCHAR(100) NOT NULL,
                    feature_value DOUBLE PRECISION,
                    feature_category VARCHAR(50),
                    PRIMARY KEY (timestamp, feature_name)
                )
                """)
                self.conn.commit()
                
                # Convert to hypertable
                return self.create_hypertable("features", "timestamp")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to create features table: {e}")
            return False

    def insert_market_data(
        self,
        data: Union[pd.DataFrame, List[Dict]]
    ) -> int:
        """Insert market data into the market_data table.

        Args:
            data: DataFrame or list of dictionaries containing market data.
                Must have columns/keys: timestamp, symbol, open, high, low, close, volume, source.

        Returns:
            Number of rows inserted.
        """
        try:
            # Convert DataFrame to list of tuples if necessary
            if isinstance(data, pd.DataFrame):
                # Ensure timestamp is in the correct format
                if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                # Convert to list of tuples
                data_tuples = [
                    (row['timestamp'], row['symbol'], row['open'], row['high'], 
                     row['low'], row['close'], row['volume'], row.get('source', 'unknown'))
                    for _, row in data.iterrows()
                ]
            else:
                # Convert list of dictionaries to list of tuples
                data_tuples = [
                    (item['timestamp'], item['symbol'], item['open'], item['high'], 
                     item['low'], item['close'], item['volume'], item.get('source', 'unknown'))
                    for item in data
                ]
            
            with self.conn.cursor() as cur:
                # Insert data using execute_values for better performance
                execute_values(
                    cur,
                    """
                    INSERT INTO market_data (timestamp, symbol, open, high, low, close, volume, source)
                    VALUES %s
                    ON CONFLICT (timestamp, symbol) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        source = EXCLUDED.source
                    """,
                    data_tuples
                )
                self.conn.commit()
                
                logger.info(f"Inserted {len(data_tuples)} rows into market_data table")
                return len(data_tuples)
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to insert market data: {e}")
            raise

    def insert_planetary_positions(
        self,
        data: Union[pd.DataFrame, List[Dict]]
    ) -> int:
        """Insert planetary positions into the planetary_positions table.

        Args:
            data: DataFrame or list of dictionaries containing planetary positions.
                Must have columns/keys: timestamp, planet, longitude, latitude, speed, 
                house, sign, nakshatra, is_retrograde.

        Returns:
            Number of rows inserted.
        """
        try:
            # Convert DataFrame to list of tuples if necessary
            if isinstance(data, pd.DataFrame):
                # Ensure timestamp is in the correct format
                if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                # Convert to list of tuples
                data_tuples = [
                    (row['timestamp'], row['planet'], row['longitude'], row['latitude'], 
                     row['speed'], row['house'], row['sign'], row['nakshatra'], row['is_retrograde'])
                    for _, row in data.iterrows()
                ]
            else:
                # Convert list of dictionaries to list of tuples
                data_tuples = [
                    (item['timestamp'], item['planet'], item['longitude'], item['latitude'], 
                     item['speed'], item['house'], item['sign'], item['nakshatra'], item['is_retrograde'])
                    for item in data
                ]
            
            with self.conn.cursor() as cur:
                # Insert data using execute_values for better performance
                execute_values(
                    cur,
                    """
                    INSERT INTO planetary_positions 
                    (timestamp, planet, longitude, latitude, speed, house, sign, nakshatra, is_retrograde)
                    VALUES %s
                    ON CONFLICT (timestamp, planet) DO UPDATE SET
                        longitude = EXCLUDED.longitude,
                        latitude = EXCLUDED.latitude,
                        speed = EXCLUDED.speed,
                        house = EXCLUDED.house,
                        sign = EXCLUDED.sign,
                        nakshatra = EXCLUDED.nakshatra,
                        is_retrograde = EXCLUDED.is_retrograde
                    """,
                    data_tuples
                )
                self.conn.commit()
                
                logger.info(f"Inserted {len(data_tuples)} rows into planetary_positions table")
                return len(data_tuples)
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to insert planetary positions: {e}")
            raise

    def insert_features(
        self,
        data: Union[pd.DataFrame, List[Dict]]
    ) -> int:
        """Insert features into the features table.

        Args:
            data: DataFrame or list of dictionaries containing features.
                Must have columns/keys: timestamp, feature_name, feature_value, feature_category.

        Returns:
            Number of rows inserted.
        """
        try:
            # Convert DataFrame to list of tuples if necessary
            if isinstance(data, pd.DataFrame):
                # Ensure timestamp is in the correct format
                if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                # Convert to list of tuples
                data_tuples = [
                    (row['timestamp'], row['feature_name'], row['feature_value'], row.get('feature_category', None))
                    for _, row in data.iterrows()
                ]
            else:
                # Convert list of dictionaries to list of tuples
                data_tuples = [
                    (item['timestamp'], item['feature_name'], item['feature_value'], item.get('feature_category', None))
                    for item in data
                ]
            
            with self.conn.cursor() as cur:
                # Insert data using execute_values for better performance
                execute_values(
                    cur,
                    """
                    INSERT INTO features (timestamp, feature_name, feature_value, feature_category)
                    VALUES %s
                    ON CONFLICT (timestamp, feature_name) DO UPDATE SET
                        feature_value = EXCLUDED.feature_value,
                        feature_category = EXCLUDED.feature_category
                    """,
                    data_tuples
                )
                self.conn.commit()
                
                logger.info(f"Inserted {len(data_tuples)} rows into features table")
                return len(data_tuples)
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to insert features: {e}")
            raise

    def query_market_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1 day"
    ) -> pd.DataFrame:
        """Query market data for a specific symbol and time range.

        Args:
            symbol: Market symbol to query.
            start_date: Start date for the query.
            end_date: End date for the query.
            interval: Time interval for aggregation (e.g., '1 day', '1 hour').

        Returns:
            DataFrame containing the queried market data.
        """
        try:
            # Convert dates to datetime if they are strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            # Create the query
            query = f"""
            SELECT
                time_bucket('{interval}', timestamp) AS time,
                symbol,
                FIRST(open, timestamp) AS open,
                MAX(high) AS high,
                MIN(low) AS low,
                LAST(close, timestamp) AS close,
                SUM(volume) AS volume
            FROM market_data
            WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
            GROUP BY time, symbol
            ORDER BY time ASC
            """
            
            # Execute the query and return the results as a DataFrame
            return pd.read_sql_query(
                query,
                self.engine,
                params=(symbol, start_date, end_date)
            )
        except Exception as e:
            logger.error(f"Failed to query market data: {e}")
            raise

    def query_planetary_positions(
        self,
        planets: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1 day"
    ) -> pd.DataFrame:
        """Query planetary positions for specific planets and time range.

        Args:
            planets: List of planets to query.
            start_date: Start date for the query.
            end_date: End date for the query.
            interval: Time interval for aggregation (e.g., '1 day', '1 hour').

        Returns:
            DataFrame containing the queried planetary positions.
        """
        try:
            # Convert dates to datetime if they are strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            # Create the query
            query = f"""
            SELECT
                time_bucket('{interval}', timestamp) AS time,
                planet,
                LAST(longitude, timestamp) AS longitude,
                LAST(latitude, timestamp) AS latitude,
                LAST(speed, timestamp) AS speed,
                LAST(house, timestamp) AS house,
                LAST(sign, timestamp) AS sign,
                LAST(nakshatra, timestamp) AS nakshatra,
                LAST(is_retrograde, timestamp) AS is_retrograde
            FROM planetary_positions
            WHERE planet = ANY(%s) AND timestamp >= %s AND timestamp <= %s
            GROUP BY time, planet
            ORDER BY time ASC, planet ASC
            """
            
            # Execute the query and return the results as a DataFrame
            return pd.read_sql_query(
                query,
                self.engine,
                params=(planets, start_date, end_date)
            )
        except Exception as e:
            logger.error(f"Failed to query planetary positions: {e}")
            raise

    def query_features(
        self,
        feature_names: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1 day",
        category: Optional[str] = None
    ) -> pd.DataFrame:
        """Query features for specific feature names and time range.

        Args:
            feature_names: List of feature names to query.
            start_date: Start date for the query.
            end_date: End date for the query.
            interval: Time interval for aggregation (e.g., '1 day', '1 hour').
            category: Optional category filter.

        Returns:
            DataFrame containing the queried features.
        """
        try:
            # Convert dates to datetime if they are strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            # Create the base query
            query = f"""
            SELECT
                time_bucket('{interval}', timestamp) AS time,
                feature_name,
                AVG(feature_value) AS feature_value,
                feature_category
            FROM features
            WHERE feature_name = ANY(%s) AND timestamp >= %s AND timestamp <= %s
            """
            
            # Add category filter if provided
            params = [feature_names, start_date, end_date]
            if category:
                query += " AND feature_category = %s"
                params.append(category)
            
            # Complete the query
            query += """
            GROUP BY time, feature_name, feature_category
            ORDER BY time ASC, feature_name ASC
            """
            
            # Execute the query and return the results as a DataFrame
            return pd.read_sql_query(
                query,
                self.engine,
                params=params
            )
        except Exception as e:
            logger.error(f"Failed to query features: {e}")
            raise

    def create_continuous_aggregate(
        self,
        view_name: str,
        source_table: str,
        time_column: str = "timestamp",
        time_bucket: str = "1 day",
        group_columns: List[str] = None,
        aggregates: Dict[str, str] = None,
        start_offset: str = "1 year",
        end_offset: str = "1 hour"
    ) -> bool:
        """Create a continuous aggregate view for efficient querying of aggregated data.

        Args:
            view_name: Name of the continuous aggregate view.
            source_table: Source table for the view.
            time_column: Name of the time column.
            time_bucket: Time bucket for aggregation.
            group_columns: Columns to group by (besides time).
            aggregates: Dictionary mapping column names to aggregate functions.
            start_offset: Start offset for the refresh window.
            end_offset: End offset for the refresh window.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Set default values if not provided
            if group_columns is None:
                group_columns = []
            
            if aggregates is None:
                aggregates = {}
            
            # Build the SELECT clause
            select_clause = f"time_bucket('{time_bucket}', {time_column}) AS bucket"
            
            # Add group columns
            for col in group_columns:
                select_clause += f", {col}"
            
            # Add aggregates
            for col, agg_func in aggregates.items():
                select_clause += f", {agg_func}({col}) AS {col}_{agg_func.lower()}"
            
            # Build the GROUP BY clause
            group_by_clause = "bucket"
            for col in group_columns:
                group_by_clause += f", {col}"
            
            # Create the view
            with self.conn.cursor() as cur:
                # Drop the view if it exists
                cur.execute(f"DROP MATERIALIZED VIEW IF EXISTS {view_name} CASCADE")
                
                # Create the continuous aggregate view
                query = f"""
                CREATE MATERIALIZED VIEW {view_name}
                WITH (timescaledb.continuous) AS
                SELECT {select_clause}
                FROM {source_table}
                GROUP BY {group_by_clause}
                WITH NO DATA
                """
                cur.execute(query)
                
                # Set the refresh policy
                query = f"""
                SELECT add_continuous_aggregate_policy('{view_name}',
                    start_offset => INTERVAL '{start_offset}',
                    end_offset => INTERVAL '{end_offset}',
                    schedule_interval => INTERVAL '1 hour')
                """
                cur.execute(query)
                
                # Refresh the view with initial data
                cur.execute(f"CALL refresh_continuous_aggregate('{view_name}', NULL, NULL)")
                
                self.conn.commit()
                logger.info(f"Created continuous aggregate view {view_name}")
                return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to create continuous aggregate view {view_name}: {e}")
            return False

    def create_default_continuous_aggregates(self) -> bool:
        """Create default continuous aggregate views for common queries.

        Returns:
            True if all views were created successfully, False otherwise.
        """
        try:
            # Create continuous aggregate for market data
            market_data_success = self.create_continuous_aggregate(
                view_name="market_data_daily",
                source_table="market_data",
                time_column="timestamp",
                time_bucket="1 day",
                group_columns=["symbol"],
                aggregates={
                    "open": "FIRST",
                    "high": "MAX",
                    "low": "MIN",
                    "close": "LAST",
                    "volume": "SUM"
                }
            )
            
            # Create continuous aggregate for planetary positions
            planetary_positions_success = self.create_continuous_aggregate(
                view_name="planetary_positions_daily",
                source_table="planetary_positions",
                time_column="timestamp",
                time_bucket="1 day",
                group_columns=["planet"],
                aggregates={
                    "longitude": "LAST",
                    "latitude": "LAST",
                    "speed": "LAST",
                    "house": "LAST",
                    "sign": "LAST",
                    "nakshatra": "LAST",
                    "is_retrograde": "LAST"
                }
            )
            
            # Create continuous aggregate for features
            features_success = self.create_continuous_aggregate(
                view_name="features_daily",
                source_table="features",
                time_column="timestamp",
                time_bucket="1 day",
                group_columns=["feature_name", "feature_category"],
                aggregates={
                    "feature_value": "AVG"
                }
            )
            
            return market_data_success and planetary_positions_success and features_success
        except Exception as e:
            logger.error(f"Failed to create default continuous aggregates: {e}")
            return False


def get_timescale_connector(
    config_file: Optional[str] = None,
    env_prefix: str = "TIMESCALE"
) -> TimescaleConnector:
    """Get a TimescaleConnector instance from configuration.

    Args:
        config_file: Path to configuration file. If None, uses environment variables.
        env_prefix: Prefix for environment variables.

    Returns:
        TimescaleConnector instance.
    """
    # Try to load configuration from file
    if config_file and os.path.exists(config_file):
        import yaml
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        # Create connector from file configuration
        connector = TimescaleConnector(
            host=config.get("host", "localhost"),
            port=config.get("port", 5432),
            database=config.get("database", "cosmic_market_oracle"),
            user=config.get("user", "cosmic"),
            password=config.get("password", "cosmic_password"),
            schema=config.get("schema", "public")
        )
    else:
        # Create connector from environment variables
        connector = TimescaleConnector(
            host=os.environ.get(f"{env_prefix}_HOST", "localhost"),
            port=int(os.environ.get(f"{env_prefix}_PORT", "5432")),
            database=os.environ.get(f"{env_prefix}_DATABASE", "cosmic_market_oracle"),
            user=os.environ.get(f"{env_prefix}_USER", "cosmic"),
            password=os.environ.get(f"{env_prefix}_PASSWORD", "cosmic_password"),
            schema=os.environ.get(f"{env_prefix}_SCHEMA", "public")
        )
    
    return connector


    def enable_compression(self, table_name: str, compress_after: str = "7 days") -> bool:
        """Enable TimescaleDB native compression on a hypertable.

        Args:
            table_name: Name of the hypertable to compress.
            compress_after: Time interval after which chunks are compressed.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with self.conn.cursor() as cur:
                # Enable compression
                query = sql.SQL("ALTER TABLE {} SET (timescaledb.compress = true)").format(
                    sql.Identifier(table_name)
                )
                cur.execute(query)
                
                # Set compression policy
                query = sql.SQL("""
                SELECT add_compression_policy({}, INTERVAL {})
                """).format(
                    sql.Literal(table_name),
                    sql.Literal(compress_after)
                )
                cur.execute(query)
                
                self.conn.commit()
                logger.info(f"Enabled compression on {table_name} with compress_after={compress_after}")
                return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to enable compression on {table_name}: {e}")
            return False

    def set_retention_policy(self, table_name: str, drop_after: str = "1 year") -> bool:
        """Set a retention policy on a hypertable to automatically drop old chunks.

        Args:
            table_name: Name of the hypertable.
            drop_after: Time interval after which chunks are dropped.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with self.conn.cursor() as cur:
                # Set retention policy
                query = sql.SQL("""
                SELECT add_retention_policy({}, INTERVAL {})
                """).format(
                    sql.Literal(table_name),
                    sql.Literal(drop_after)
                )
                cur.execute(query)
                
                self.conn.commit()
                logger.info(f"Set retention policy on {table_name} with drop_after={drop_after}")
                return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to set retention policy on {table_name}: {e}")
            return False

    def optimize_chunk_time_interval(self, table_name: str, new_interval: str) -> bool:
        """Optimize the chunk time interval for a hypertable.

        Args:
            table_name: Name of the hypertable.
            new_interval: New chunk time interval.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with self.conn.cursor() as cur:
                # Set new chunk time interval
                query = sql.SQL("""
                SELECT set_chunk_time_interval({}, INTERVAL {})
                """).format(
                    sql.Literal(table_name),
                    sql.Literal(new_interval)
                )
                cur.execute(query)
                
                self.conn.commit()
                logger.info(f"Set chunk time interval on {table_name} to {new_interval}")
                return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to set chunk time interval on {table_name}: {e}")
            return False

    def create_synchronized_view(self, view_name: str, interval: str = "1 day") -> bool:
        """Create a view that synchronizes market data with planetary positions.

        This creates a view that joins market data with planetary positions on the same timestamp,
        allowing for integrated analysis of financial and astrological data.

        Args:
            view_name: Name of the view to create.
            interval: Time interval for bucketing data.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with self.conn.cursor() as cur:
                # Create the synchronized view
                query = f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT
                    time_bucket('{interval}', m.timestamp) AS time,
                    m.symbol,
                    m.open,
                    m.high,
                    m.low,
                    m.close,
                    m.volume,
                    p.planet,
                    p.longitude,
                    p.latitude,
                    p.speed,
                    p.house,
                    p.sign,
                    p.nakshatra,
                    p.is_retrograde
                FROM
                    market_data m
                JOIN
                    planetary_positions p
                ON
                    time_bucket('{interval}', m.timestamp) = time_bucket('{interval}', p.timestamp)
                ORDER BY
                    time, m.symbol, p.planet
                """
                cur.execute(query)
                
                self.conn.commit()
                logger.info(f"Created synchronized view {view_name}")
                return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to create synchronized view {view_name}: {e}")
            return False

    def create_market_planetary_features_view(self, view_name: str, interval: str = "1 day") -> bool:
        """Create a view that combines market data, planetary positions, and features.

        This creates a comprehensive view that joins all three main data types for integrated analysis.

        Args:
            view_name: Name of the view to create.
            interval: Time interval for bucketing data.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with self.conn.cursor() as cur:
                # Create the integrated view
                query = f"""
                CREATE OR REPLACE VIEW {view_name} AS
                WITH market_agg AS (
                    SELECT
                        time_bucket('{interval}', timestamp) AS time,
                        symbol,
                        FIRST(open, timestamp) AS open,
                        MAX(high) AS high,
                        MIN(low) AS low,
                        LAST(close, timestamp) AS close,
                        SUM(volume) AS volume
                    FROM market_data
                    GROUP BY time, symbol
                ),
                planetary_agg AS (
                    SELECT
                        time_bucket('{interval}', timestamp) AS time,
                        planet,
                        LAST(longitude, timestamp) AS longitude,
                        LAST(latitude, timestamp) AS latitude,
                        LAST(speed, timestamp) AS speed,
                        LAST(house, timestamp) AS house,
                        LAST(sign, timestamp) AS sign,
                        LAST(nakshatra, timestamp) AS nakshatra,
                        LAST(is_retrograde, timestamp) AS is_retrograde
                    FROM planetary_positions
                    GROUP BY time, planet
                ),
                feature_agg AS (
                    SELECT
                        time_bucket('{interval}', timestamp) AS time,
                        feature_name,
                        AVG(feature_value) AS feature_value,
                        feature_category
                    FROM features
                    GROUP BY time, feature_name, feature_category
                )
                SELECT
                    m.time,
                    m.symbol,
                    m.open,
                    m.high,
                    m.low,
                    m.close,
                    m.volume,
                    p.planet,
                    p.longitude,
                    p.latitude,
                    p.speed,
                    p.house,
                    p.sign,
                    p.nakshatra,
                    p.is_retrograde,
                    f.feature_name,
                    f.feature_value,
                    f.feature_category
                FROM
                    market_agg m
                FULL OUTER JOIN
                    planetary_agg p
                ON
                    m.time = p.time
                FULL OUTER JOIN
                    feature_agg f
                ON
                    m.time = f.time
                ORDER BY
                    m.time, m.symbol, p.planet, f.feature_name
                """
                cur.execute(query)
                
                self.conn.commit()
                logger.info(f"Created integrated view {view_name}")
                return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to create integrated view {view_name}: {e}")
            return False

    def create_correlation_analysis(self, output_table: str, market_symbol: str, planet: str, 
                                  feature_name: Optional[str] = None, window: str = "30 days") -> bool:
        """Create a table with correlation analysis between market data and planetary positions.

        Args:
            output_table: Name of the output table to store correlation results.
            market_symbol: Market symbol to analyze.
            planet: Planet to analyze.
            feature_name: Optional feature to include in analysis.
            window: Time window for rolling correlation.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with self.conn.cursor() as cur:
                # Create the output table if it doesn't exist
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {output_table} (
                    timestamp TIMESTAMPTZ NOT NULL,
                    market_symbol VARCHAR(20) NOT NULL,
                    planet VARCHAR(20) NOT NULL,
                    feature_name VARCHAR(100),
                    correlation_price DOUBLE PRECISION,
                    correlation_volatility DOUBLE PRECISION,
                    correlation_volume DOUBLE PRECISION,
                    p_value DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, market_symbol, planet, feature_name)
                )
                """
                cur.execute(create_table_query)
                
                # Make it a hypertable if it's not already
                self.create_hypertable(output_table, "timestamp", if_not_exists=True)
                
                # Build the correlation analysis query
                feature_join = ""
                feature_select = ""
                feature_where = ""
                if feature_name:
                    feature_join = f"""
                    LEFT JOIN features f 
                    ON time_bucket('1 day', m.timestamp) = time_bucket('1 day', f.timestamp)
                    """
                    feature_select = ", f.feature_value"
                    feature_where = f"AND f.feature_name = '{feature_name}'"
                
                # Insert correlation results
                analysis_query = f"""
                INSERT INTO {output_table} (timestamp, market_symbol, planet, feature_name, 
                                          correlation_price, correlation_volatility, correlation_volume, p_value)
                WITH rolling_data AS (
                    SELECT
                        time_bucket('1 day', m.timestamp) AS time,
                        m.symbol,
                        p.planet,
                        {'NULL' if not feature_name else "'" + feature_name + "'"} AS feature_name,
                        m.close,
                        (m.high - m.low) / m.low AS volatility,
                        m.volume,
                        p.longitude,
                        p.speed,
                        p.latitude{feature_select}
                    FROM
                        market_data m
                    JOIN
                        planetary_positions p
                    ON
                        time_bucket('1 day', m.timestamp) = time_bucket('1 day', p.timestamp)
                    {feature_join}
                    WHERE
                        m.symbol = '{market_symbol}'
                        AND p.planet = '{planet}'
                        {feature_where}
                    ORDER BY
                        time
                )
                SELECT
                    now() AS timestamp,
                    '{market_symbol}' AS market_symbol,
                    '{planet}' AS planet,
                    {'NULL' if not feature_name else "'" + feature_name + "'"} AS feature_name,
                    corr(close, longitude) AS correlation_price,
                    corr(volatility, longitude) AS correlation_volatility,
                    corr(volume, longitude) AS correlation_volume,
                    0.05 AS p_value  -- Placeholder, would need more complex stats calculation
                FROM
                    rolling_data
                WHERE
                    time >= now() - INTERVAL '{window}'
                GROUP BY
                    market_symbol, planet, feature_name
                ON CONFLICT (timestamp, market_symbol, planet, feature_name) DO UPDATE SET
                    correlation_price = EXCLUDED.correlation_price,
                    correlation_volatility = EXCLUDED.correlation_volatility,
                    correlation_volume = EXCLUDED.correlation_volume,
                    p_value = EXCLUDED.p_value
                """
                cur.execute(analysis_query)
                
                self.conn.commit()
                logger.info(f"Created correlation analysis in {output_table}")
                return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to create correlation analysis: {e}")
            return False

    def setup_complete_timescale_environment(self) -> bool:
        """Set up a complete TimescaleDB environment with all tables, views, and optimizations.

        This is a convenience method that sets up everything needed for the Cosmic Market Oracle project.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Create tables
            self.create_market_data_table()
            self.create_planetary_positions_table()
            self.create_features_table()
            
            # Create continuous aggregates
            self.create_default_continuous_aggregates()
            
            # Enable compression
            self.enable_compression("market_data")
            self.enable_compression("planetary_positions")
            self.enable_compression("features")
            
            # Set retention policies
            self.set_retention_policy("market_data", "5 years")
            self.set_retention_policy("planetary_positions", "5 years")
            self.set_retention_policy("features", "5 years")
            
            # Create synchronized views
            self.create_synchronized_view("market_planetary_sync")
            self.create_market_planetary_features_view("cosmic_market_integrated")
            
            logger.info("Set up complete TimescaleDB environment")
            return True
        except Exception as e:
            logger.error(f"Failed to set up complete TimescaleDB environment: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    connector = get_timescale_connector()
    connector.connect()
    
    # Set up complete environment
    connector.setup_complete_timescale_environment()
    
    # Example correlation analysis
    connector.create_correlation_analysis(
        output_table="market_planetary_correlations",
        market_symbol="DJI",
        planet="Jupiter",
        window="90 days"
    )
    
    connector.disconnect()