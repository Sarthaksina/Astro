#!/usr/bin/env python3
"""
Tests for TimescaleDB Connector

This module contains tests for the TimescaleConnector class, which provides utilities
for connecting to TimescaleDB and managing time-series data for the Cosmic Market Oracle project.
"""

import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.utils.db.timescale_connector import TimescaleConnector, get_timescale_connector


class TestTimescaleConnector(unittest.TestCase):
    """Test cases for TimescaleConnector class."""

    def setUp(self):
        """Set up test environment."""
        # Create a mock connector with mocked connection and cursor
        self.connector = TimescaleConnector(
            host="test_host",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_password"
        )
        
        # Mock the connection and cursor
        self.connector.conn = MagicMock()
        self.connector.engine = MagicMock()
        
        # Create a mock cursor
        self.mock_cursor = MagicMock()
        self.connector.conn.cursor.return_value.__enter__.return_value = self.mock_cursor

    def test_connect(self):
        """Test connect method."""
        with patch('psycopg2.connect') as mock_connect, \
             patch('sqlalchemy.create_engine') as mock_create_engine:
            # Reset connection attributes to None
            self.connector.conn = None
            self.connector.engine = None
            
            # Configure mocks
            mock_connect.return_value = MagicMock()
            mock_create_engine.return_value = MagicMock()
            
            # Call connect
            self.connector.connect()
            
            # Verify connections were established
            mock_connect.assert_called_once()
            mock_create_engine.assert_called_once()
            self.assertIsNotNone(self.connector.conn)
            self.assertIsNotNone(self.connector.engine)

    def test_disconnect(self):
        """Test disconnect method."""
        # Call disconnect
        self.connector.disconnect()
        
        # Verify connections were closed
        self.connector.conn.close.assert_called_once()
        self.connector.engine.dispose.assert_called_once()
        self.assertIsNone(self.connector.conn)
        self.assertIsNone(self.connector.engine)

    def test_create_hypertable(self):
        """Test create_hypertable method."""
        # Call create_hypertable
        result = self.connector.create_hypertable("test_table", "time_col", "2 days")
        
        # Verify SQL was executed
        self.mock_cursor.execute.assert_called_once()
        self.connector.conn.commit.assert_called_once()
        self.assertTrue(result)

    def test_enable_compression(self):
        """Test enable_compression method."""
        # Call enable_compression
        result = self.connector.enable_compression("test_table", "14 days")
        
        # Verify SQL was executed twice (enable compression + set policy)
        self.assertEqual(self.mock_cursor.execute.call_count, 2)
        self.connector.conn.commit.assert_called_once()
        self.assertTrue(result)

    def test_set_retention_policy(self):
        """Test set_retention_policy method."""
        # Call set_retention_policy
        result = self.connector.set_retention_policy("test_table", "2 years")
        
        # Verify SQL was executed
        self.mock_cursor.execute.assert_called_once()
        self.connector.conn.commit.assert_called_once()
        self.assertTrue(result)

    def test_create_synchronized_view(self):
        """Test create_synchronized_view method."""
        # Call create_synchronized_view
        result = self.connector.create_synchronized_view("test_sync_view", "1 hour")
        
        # Verify SQL was executed
        self.mock_cursor.execute.assert_called_once()
        self.connector.conn.commit.assert_called_once()
        self.assertTrue(result)

    def test_create_market_planetary_features_view(self):
        """Test create_market_planetary_features_view method."""
        # Call create_market_planetary_features_view
        result = self.connector.create_market_planetary_features_view("test_integrated_view")
        
        # Verify SQL was executed
        self.mock_cursor.execute.assert_called_once()
        self.connector.conn.commit.assert_called_once()
        self.assertTrue(result)

    def test_create_correlation_analysis(self):
        """Test create_correlation_analysis method."""
        # Call create_correlation_analysis
        result = self.connector.create_correlation_analysis(
            "test_correlations", "BTC", "Mars", window="60 days"
        )
        
        # Verify SQL was executed multiple times (create table + hypertable + analysis)
        self.assertTrue(self.mock_cursor.execute.call_count >= 2)
        self.connector.conn.commit.assert_called_once()
        self.assertTrue(result)

    def test_setup_complete_timescale_environment(self):
        """Test setup_complete_timescale_environment method."""
        # Mock all the methods called by setup_complete_timescale_environment
        self.connector.create_market_data_table = MagicMock(return_value=True)
        self.connector.create_planetary_positions_table = MagicMock(return_value=True)
        self.connector.create_features_table = MagicMock(return_value=True)
        self.connector.create_default_continuous_aggregates = MagicMock(return_value=True)
        self.connector.enable_compression = MagicMock(return_value=True)
        self.connector.set_retention_policy = MagicMock(return_value=True)
        self.connector.create_synchronized_view = MagicMock(return_value=True)
        self.connector.create_market_planetary_features_view = MagicMock(return_value=True)
        
        # Call setup_complete_timescale_environment
        result = self.connector.setup_complete_timescale_environment()
        
        # Verify all methods were called
        self.connector.create_market_data_table.assert_called_once()
        self.connector.create_planetary_positions_table.assert_called_once()
        self.connector.create_features_table.assert_called_once()
        self.connector.create_default_continuous_aggregates.assert_called_once()
        self.assertEqual(self.connector.enable_compression.call_count, 3)  # Called for 3 tables
        self.assertEqual(self.connector.set_retention_policy.call_count, 3)  # Called for 3 tables
        self.connector.create_synchronized_view.assert_called_once()
        self.connector.create_market_planetary_features_view.assert_called_once()
        self.assertTrue(result)

    def test_insert_and_query_market_data(self):
        """Test insert_market_data and query_market_data methods."""
        # Mock execute_values
        with patch('psycopg2.extras.execute_values') as mock_execute_values, \
             patch('pandas.read_sql_query') as mock_read_sql:
            # Create test data
            test_data = pd.DataFrame({
                'timestamp': [datetime.now()],
                'symbol': ['DJI'],
                'open': [100.0],
                'high': [105.0],
                'low': [95.0],
                'close': [102.0],
                'volume': [1000000],
                'source': ['test']
            })
            
            # Configure mock for query result
            mock_read_sql.return_value = test_data
            
            # Test insert_market_data
            result = self.connector.insert_market_data(test_data)
            mock_execute_values.assert_called_once()
            self.connector.conn.commit.assert_called_once()
            self.assertEqual(result, 1)  # 1 row inserted
            
            # Reset mocks
            self.connector.conn.commit.reset_mock()
            
            # Test query_market_data
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
            query_result = self.connector.query_market_data('DJI', start_date, end_date)
            mock_read_sql.assert_called_once()
            self.assertIsInstance(query_result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()