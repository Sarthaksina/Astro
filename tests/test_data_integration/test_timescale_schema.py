"""
Tests for the TimescaleDB schema module.

This module tests the functionality of the TimescaleDB schema and database manager.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data_integration.timescale_schema import (
    DatabaseManager, PlanetaryPosition, FinancialData, MarketRegime,
    AstrologicalEvent, TechnicalIndicator, Prediction, DataSyncLog,
    Base
)


@pytest.fixture
def mock_engine():
    """Create a mock SQLAlchemy engine."""
    with patch('src.data_integration.timescale_schema.create_engine') as mock_create_engine:
        engine_mock = MagicMock()
        mock_create_engine.return_value = engine_mock
        yield engine_mock


@pytest.fixture
def mock_session():
    """Create a mock SQLAlchemy session."""
    with patch('src.data_integration.timescale_schema.sessionmaker') as mock_sessionmaker:
        session_factory = MagicMock()
        session_mock = MagicMock()
        session_factory.return_value = session_mock
        mock_sessionmaker.return_value = session_factory
        yield session_mock


@pytest.fixture
def mock_psycopg2_connect():
    """Create a mock psycopg2 connection."""
    with patch('src.data_integration.timescale_schema.psycopg2.connect') as mock_connect:
        conn_mock = MagicMock()
        cursor_mock = MagicMock()
        conn_mock.cursor.return_value = cursor_mock
        mock_connect.return_value = conn_mock
        yield conn_mock, cursor_mock


def test_database_manager_initialization():
    """Test DatabaseManager initialization."""
    db_manager = DatabaseManager(
        host="test_host",
        port=5432,
        user="test_user",
        password="test_password",
        database="test_db",
        schema="test_schema"
    )
    
    assert db_manager.host == "test_host"
    assert db_manager.port == 5432
    assert db_manager.user == "test_user"
    assert db_manager.password == "test_password"
    assert db_manager.database == "test_db"
    assert db_manager.schema == "test_schema"
    assert db_manager.engine is None
    assert db_manager.Session is None


def test_create_database(mock_psycopg2_connect):
    """Test creating a database."""
    conn_mock, cursor_mock = mock_psycopg2_connect
    
    # Mock database existence check
    cursor_mock.fetchone.return_value = None  # Database doesn't exist
    
    # Create database manager
    db_manager = DatabaseManager(
        host="test_host",
        port=5432,
        user="test_user",
        password="test_password",
        database="test_db",
        schema="test_schema"
    )
    
    # Call create_database
    db_manager.create_database()
    
    # Check that connect was called with correct parameters
    mock_psycopg2_connect.assert_called_once_with(
        host="test_host",
        port=5432,
        user="test_user",
        password="test_password"
    )
    
    # Check that cursor was created
    conn_mock.cursor.assert_called_once()
    
    # Check that database existence was checked
    cursor_mock.execute.assert_any_call("SELECT 1 FROM pg_database WHERE datname = %s", ("test_db",))
    
    # Check that database was created
    cursor_mock.execute.assert_any_call("CREATE DATABASE test_db")
    
    # Check that cursor and connection were closed
    cursor_mock.close.assert_called_once()
    conn_mock.close.assert_called_once()


def test_create_database_exists(mock_psycopg2_connect):
    """Test creating a database that already exists."""
    conn_mock, cursor_mock = mock_psycopg2_connect
    
    # Mock database existence check
    cursor_mock.fetchone.return_value = (1,)  # Database exists
    
    # Create database manager
    db_manager = DatabaseManager(
        host="test_host",
        port=5432,
        user="test_user",
        password="test_password",
        database="test_db",
        schema="test_schema"
    )
    
    # Call create_database
    db_manager.create_database()
    
    # Check that connect was called with correct parameters
    mock_psycopg2_connect.assert_called_once_with(
        host="test_host",
        port=5432,
        user="test_user",
        password="test_password"
    )
    
    # Check that cursor was created
    conn_mock.cursor.assert_called_once()
    
    # Check that database existence was checked
    cursor_mock.execute.assert_called_once_with("SELECT 1 FROM pg_database WHERE datname = %s", ("test_db",))
    
    # Check that database was not created
    assert "CREATE DATABASE" not in str(cursor_mock.execute.call_args_list)
    
    # Check that cursor and connection were closed
    cursor_mock.close.assert_called_once()
    conn_mock.close.assert_called_once()


def test_connect(mock_engine):
    """Test connecting to the database."""
    # Create database manager
    db_manager = DatabaseManager(
        host="test_host",
        port=5432,
        user="test_user",
        password="test_password",
        database="test_db",
        schema="test_schema"
    )
    
    # Mock engine.connect context manager
    conn_mock = MagicMock()
    mock_engine.__enter__.return_value = conn_mock
    
    # Mock schema existence check
    conn_mock.dialect.has_schema.return_value = False
    
    # Call connect
    with patch('src.data_integration.timescale_schema.sessionmaker') as mock_sessionmaker:
        session_factory = MagicMock()
        mock_sessionmaker.return_value = session_factory
        
        db_manager.connect()
        
        # Check that engine was created with correct connection string
        from src.data_integration.timescale_schema import create_engine
        create_engine.assert_called_once_with(
            "postgresql://test_user:test_password@test_host:5432/test_db"
        )
        
        # Check that session factory was created
        mock_sessionmaker.assert_called_once_with(bind=mock_engine)
        
        # Check that schema existence was checked
        conn_mock.dialect.has_schema.assert_called_once_with(conn_mock, "test_schema")
        
        # Check that schema was created
        conn_mock.execute.assert_any_call("CREATE SCHEMA test_schema")
        
        # Check that TimescaleDB extension was created
        conn_mock.execute.assert_any_call("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")


def test_connect_schema_exists(mock_engine):
    """Test connecting to the database with existing schema."""
    # Create database manager
    db_manager = DatabaseManager(
        host="test_host",
        port=5432,
        user="test_user",
        password="test_password",
        database="test_db",
        schema="test_schema"
    )
    
    # Mock engine.connect context manager
    conn_mock = MagicMock()
    mock_engine.__enter__.return_value = conn_mock
    
    # Mock schema existence check
    conn_mock.dialect.has_schema.return_value = True
    
    # Call connect
    with patch('src.data_integration.timescale_schema.sessionmaker') as mock_sessionmaker:
        session_factory = MagicMock()
        mock_sessionmaker.return_value = session_factory
        
        db_manager.connect()
        
        # Check that engine was created with correct connection string
        from src.data_integration.timescale_schema import create_engine
        create_engine.assert_called_once_with(
            "postgresql://test_user:test_password@test_host:5432/test_db"
        )
        
        # Check that session factory was created
        mock_sessionmaker.assert_called_once_with(bind=mock_engine)
        
        # Check that schema existence was checked
        conn_mock.dialect.has_schema.assert_called_once_with(conn_mock, "test_schema")
        
        # Check that schema was not created
        assert "CREATE SCHEMA" not in str(conn_mock.execute.call_args_list)
        
        # Check that TimescaleDB extension was created
        conn_mock.execute.assert_called_once_with("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")


def test_create_tables(mock_engine):
    """Test creating tables."""
    # Create database manager
    db_manager = DatabaseManager(
        host="test_host",
        port=5432,
        user="test_user",
        password="test_password",
        database="test_db",
        schema="test_schema"
    )
    
    # Set engine
    db_manager.engine = mock_engine
    
    # Call create_tables
    with patch.object(Base.metadata, 'create_all') as mock_create_all:
        with patch.object(db_manager, '_convert_to_hypertables') as mock_convert:
            db_manager.create_tables()
            
            # Check that create_all was called
            mock_create_all.assert_called_once_with(mock_engine)
            
            # Check that _convert_to_hypertables was called
            mock_convert.assert_called_once()


def test_convert_to_hypertables(mock_engine):
    """Test converting tables to hypertables."""
    # Create database manager
    db_manager = DatabaseManager(
        host="test_host",
        port=5432,
        user="test_user",
        password="test_password",
        database="test_db",
        schema="test_schema"
    )
    
    # Set engine
    db_manager.engine = mock_engine
    
    # Mock engine.connect context manager
    conn_mock = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = conn_mock
    
    # Mock hypertable existence check
    conn_mock.execute.return_value.fetchone.return_value = None  # Table is not a hypertable
    
    # Call _convert_to_hypertables
    db_manager._convert_to_hypertables()
    
    # Check that hypertable existence was checked for each table
    expected_tables = [
        "planetary_positions", "financial_data", "market_regimes",
        "astrological_events", "technical_indicators", "predictions"
    ]
    
    for table in expected_tables:
        # Check hypertable existence query
        query = f"""
                SELECT 1 FROM timescaledb_information.hypertables 
                WHERE hypertable_name = '{table}' AND hypertable_schema = 'test_schema'
                """
        conn_mock.execute.assert_any_call(query)
        
        # Check create_hypertable query
        query = f"""
                    SELECT create_hypertable(
                        'test_schema.{table}', 'timestamp',
                        chunk_time_interval => interval '1 month'
                    )
                    """
        conn_mock.execute.assert_any_call(query)
        
        # Check retention policy query
        query = f"""
                SELECT add_retention_policy(
                    'test_schema.{table}', interval '5 years'
                )
                """
        conn_mock.execute.assert_any_call(query)
        
        # Check index creation query
        query = f"""
                CREATE INDEX IF NOT EXISTS {table}_timestamp_idx ON test_schema.{table} (timestamp DESC)
                """
        conn_mock.execute.assert_any_call(query)


def test_drop_tables(mock_engine):
    """Test dropping tables."""
    # Create database manager
    db_manager = DatabaseManager(
        host="test_host",
        port=5432,
        user="test_user",
        password="test_password",
        database="test_db",
        schema="test_schema"
    )
    
    # Set engine
    db_manager.engine = mock_engine
    
    # Call drop_tables
    with patch.object(Base.metadata, 'drop_all') as mock_drop_all:
        db_manager.drop_tables()
        
        # Check that drop_all was called
        mock_drop_all.assert_called_once_with(mock_engine)


def test_planetary_position_model():
    """Test PlanetaryPosition model."""
    # Create a PlanetaryPosition instance
    position = PlanetaryPosition(
        timestamp=datetime(2023, 1, 1),
        planet_id=1,
        longitude=45.0,
        latitude=0.0,
        distance=1.0,
        speed=1.0,
        is_retrograde=False,
        sign=2,
        nakshatra=5,
        house=1,
        dignity="exalted",
        aspects={"planet_3": 120.0}
    )
    
    # Check attributes
    assert position.timestamp == datetime(2023, 1, 1)
    assert position.planet_id == 1
    assert position.longitude == 45.0
    assert position.latitude == 0.0
    assert position.distance == 1.0
    assert position.speed == 1.0
    assert position.is_retrograde is False
    assert position.sign == 2
    assert position.nakshatra == 5
    assert position.house == 1
    assert position.dignity == "exalted"
    assert position.aspects == {"planet_3": 120.0}
    
    # Check __repr__
    assert "PlanetaryPosition" in repr(position)
    assert "planet_id=1" in repr(position)


def test_financial_data_model():
    """Test FinancialData model."""
    # Create a FinancialData instance
    data = FinancialData(
        timestamp=datetime(2023, 1, 1),
        symbol="SPY",
        open=400.0,
        high=410.0,
        low=395.0,
        close=405.0,
        volume=1000000,
        adjusted_close=405.0,
        source="yahoo_finance",
        metadata={"dividend": 0.0}
    )
    
    # Check attributes
    assert data.timestamp == datetime(2023, 1, 1)
    assert data.symbol == "SPY"
    assert data.open == 400.0
    assert data.high == 410.0
    assert data.low == 395.0
    assert data.close == 405.0
    assert data.volume == 1000000
    assert data.adjusted_close == 405.0
    assert data.source == "yahoo_finance"
    assert data.metadata == {"dividend": 0.0}
    
    # Check __repr__
    assert "FinancialData" in repr(data)
    assert "symbol=SPY" in repr(data)


def test_market_regime_model():
    """Test MarketRegime model."""
    # Create a MarketRegime instance
    regime = MarketRegime(
        timestamp=datetime(2023, 1, 1),
        symbol="SPY",
        regime_hmm=1,
        regime_kmeans=2,
        regime_rule_based=1,
        regime_consensus=1,
        volatility=0.2,
        trend=0.1,
        metadata={"hmm_probability": 0.8}
    )
    
    # Check attributes
    assert regime.timestamp == datetime(2023, 1, 1)
    assert regime.symbol == "SPY"
    assert regime.regime_hmm == 1
    assert regime.regime_kmeans == 2
    assert regime.regime_rule_based == 1
    assert regime.regime_consensus == 1
    assert regime.volatility == 0.2
    assert regime.trend == 0.1
    assert regime.metadata == {"hmm_probability": 0.8}
    
    # Check __repr__
    assert "MarketRegime" in repr(regime)
    assert "symbol=SPY" in repr(regime)


def test_astrological_event_model():
    """Test AstrologicalEvent model."""
    # Create an AstrologicalEvent instance
    event = AstrologicalEvent(
        timestamp=datetime(2023, 1, 1),
        event_type="conjunction",
        planets_involved=[1, 2],
        description="Sun-Moon conjunction",
        strength=0.9,
        market_impact="positive",
        metadata={"orb": 2.0}
    )
    
    # Check attributes
    assert event.timestamp == datetime(2023, 1, 1)
    assert event.event_type == "conjunction"
    assert event.planets_involved == [1, 2]
    assert event.description == "Sun-Moon conjunction"
    assert event.strength == 0.9
    assert event.market_impact == "positive"
    assert event.metadata == {"orb": 2.0}
    
    # Check __repr__
    assert "AstrologicalEvent" in repr(event)
    assert "event_type=conjunction" in repr(event)


def test_technical_indicator_model():
    """Test TechnicalIndicator model."""
    # Create a TechnicalIndicator instance
    indicator = TechnicalIndicator(
        timestamp=datetime(2023, 1, 1),
        symbol="SPY",
        indicator_type="rsi",
        value=65.0,
        parameters={"window": 14},
        metadata={"overbought": 70.0}
    )
    
    # Check attributes
    assert indicator.timestamp == datetime(2023, 1, 1)
    assert indicator.symbol == "SPY"
    assert indicator.indicator_type == "rsi"
    assert indicator.value == 65.0
    assert indicator.parameters == {"window": 14}
    assert indicator.metadata == {"overbought": 70.0}
    
    # Check __repr__
    assert "TechnicalIndicator" in repr(indicator)
    assert "symbol=SPY" in repr(indicator)
    assert "indicator_type=rsi" in repr(indicator)


def test_prediction_model():
    """Test Prediction model."""
    # Create a Prediction instance
    prediction = Prediction(
        timestamp=datetime(2023, 1, 1),
        symbol="SPY",
        model_id="lstm_v1",
        prediction_type="price",
        target_timestamp=datetime(2023, 1, 2),
        value=410.0,
        confidence=0.8,
        features_used={"technical": ["rsi", "macd"]},
        metadata={"training_date": "2023-01-01"}
    )
    
    # Check attributes
    assert prediction.timestamp == datetime(2023, 1, 1)
    assert prediction.symbol == "SPY"
    assert prediction.model_id == "lstm_v1"
    assert prediction.prediction_type == "price"
    assert prediction.target_timestamp == datetime(2023, 1, 2)
    assert prediction.value == 410.0
    assert prediction.confidence == 0.8
    assert prediction.features_used == {"technical": ["rsi", "macd"]}
    assert prediction.metadata == {"training_date": "2023-01-01"}
    
    # Check __repr__
    assert "Prediction" in repr(prediction)
    assert "symbol=SPY" in repr(prediction)
    assert "model_id=lstm_v1" in repr(prediction)


def test_data_sync_log_model():
    """Test DataSyncLog model."""
    # Create a DataSyncLog instance
    log = DataSyncLog(
        timestamp=datetime(2023, 1, 1),
        data_type="financial_data",
        source="yahoo_finance",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        records_processed=100,
        success=True,
        error_message=None,
        metadata={"symbols": ["SPY", "QQQ"]}
    )
    
    # Check attributes
    assert log.timestamp == datetime(2023, 1, 1)
    assert log.data_type == "financial_data"
    assert log.source == "yahoo_finance"
    assert log.start_date == datetime(2023, 1, 1)
    assert log.end_date == datetime(2023, 1, 31)
    assert log.records_processed == 100
    assert log.success is True
    assert log.error_message is None
    assert log.metadata == {"symbols": ["SPY", "QQQ"]}
    
    # Check __repr__
    assert "DataSyncLog" in repr(log)
    assert "data_type=financial_data" in repr(log)
