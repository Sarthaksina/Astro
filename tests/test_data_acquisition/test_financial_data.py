"""
Tests for the financial data acquisition module.

This module contains unit tests for the financial data components, including:
- Financial data sources (Yahoo Finance, CRSP, Bloomberg)
- Market data processing
- Market reconstruction
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import os
import tempfile
import shutil
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_acquisition.financial_data import (
    FinancialDataSource,
    YahooFinanceDataSource,
    CRSPDataSource,
    BloombergDataSource,
    MarketDataProcessor,
    MarketReconstructionEngine
)


class TestFinancialDataSource:
    """Tests for the base FinancialDataSource class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a concrete subclass for testing
        class ConcreteDataSource(FinancialDataSource):
            def fetch_data(self, **kwargs):
                return pd.DataFrame({
                    'open': [100, 101, 102],
                    'high': [105, 106, 107],
                    'low': [95, 96, 97],
                    'close': [102, 103, 104],
                    'volume': [1000000, 1100000, 1200000]
                }, index=pd.date_range(start='2020-01-01', periods=3))
        
        self.data_source = ConcreteDataSource("Test Source", self.temp_dir)
        self.test_data = self.data_source.fetch_data()
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test data source initialization."""
        assert self.data_source.name == "Test Source"
        assert self.data_source.data_dir == self.temp_dir
        assert os.path.exists(self.temp_dir)
    
    def test_save_and_load_csv(self):
        """Test saving and loading CSV data."""
        filename = "test_data.csv"
        self.data_source.save_data(self.test_data, filename)
        
        # Check if file was created
        filepath = os.path.join(self.temp_dir, filename)
        assert os.path.exists(filepath)
        
        # Load the data
        loaded_data = self.data_source.load_data(filename)
        
        # Check if data is the same
        pd.testing.assert_frame_equal(
            self.test_data, loaded_data, check_dtype=False)
    
    def test_save_and_load_parquet(self):
        """Test saving and loading Parquet data."""
        filename = "test_data.parquet"
        self.data_source.save_data(self.test_data, filename)
        
        # Check if file was created
        filepath = os.path.join(self.temp_dir, filename)
        assert os.path.exists(filepath)
        
        # Load the data
        loaded_data = self.data_source.load_data(filename)
        
        # Check if data is the same
        pd.testing.assert_frame_equal(self.test_data, loaded_data)
    
    def test_save_and_load_pickle(self):
        """Test saving and loading Pickle data."""
        filename = "test_data.pkl"
        self.data_source.save_data(self.test_data, filename)
        
        # Check if file was created
        filepath = os.path.join(self.temp_dir, filename)
        assert os.path.exists(filepath)
        
        # Load the data
        loaded_data = self.data_source.load_data(filename)
        
        # Check if data is the same
        pd.testing.assert_frame_equal(self.test_data, loaded_data)
    
    def test_save_default_format(self):
        """Test saving with default format."""
        filename = "test_data"  # No extension
        self.data_source.save_data(self.test_data, filename)
        
        # Check if file was created with .parquet extension
        filepath = os.path.join(self.temp_dir, filename + ".parquet")
        assert os.path.exists(filepath)
    
    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            self.data_source.load_data("nonexistent_file.csv")
    
    def test_load_unsupported_format(self):
        """Test loading an unsupported format."""
        # Create a file with unsupported extension
        filepath = os.path.join(self.temp_dir, "test_data.txt")
        with open(filepath, 'w') as f:
            f.write("test")
            
        with pytest.raises(ValueError):
            self.data_source.load_data("test_data.txt")


class TestYahooFinanceDataSource:
    """Tests for the YahooFinanceDataSource class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_source = YahooFinanceDataSource(self.temp_dir)
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch('yfinance.Ticker')
    def test_fetch_data(self, mock_ticker):
        """Test fetching data from Yahoo Finance."""
        # Mock the Ticker.history method
        mock_history = MagicMock()
        mock_history.return_value = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range(start='2020-01-01', periods=3))
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history = mock_history
        mock_ticker.return_value = mock_ticker_instance
        
        # Fetch data
        data = self.data_source.fetch_data(
            symbol="AAPL",
            start_date="2020-01-01",
            end_date="2020-01-03"
        )
        
        # Check if Ticker was called with correct arguments
        mock_ticker.assert_called_once_with("AAPL")
        mock_history.assert_called_once_with(
            start="2020-01-01",
            end="2020-01-03",
            interval="1d"
        )
        
        # Check if data was returned
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3
        assert 'open' in data.columns  # Column names should be lowercase
        
        # Check if metadata was saved
        metadata_files = [f for f in os.listdir(self.temp_dir) if f.endswith('_metadata.json')]
        assert len(metadata_files) == 1
        
        # Check metadata content
        with open(os.path.join(self.temp_dir, metadata_files[0]), 'r') as f:
            metadata = json.load(f)
            assert metadata['source'] == "Yahoo Finance"
            assert metadata['symbol'] == "AAPL"
            assert metadata['start_date'] == "2020-01-01"
            assert metadata['end_date'] == "2020-01-03"
    
    @patch('yfinance.Ticker')
    def test_fetch_multiple_symbols(self, mock_ticker):
        """Test fetching data for multiple symbols."""
        # Mock the Ticker.history method
        mock_history = MagicMock()
        mock_history.return_value = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range(start='2020-01-01', periods=3))
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history = mock_history
        mock_ticker.return_value = mock_ticker_instance
        
        # Fetch data for multiple symbols
        results = self.data_source.fetch_multiple_symbols(
            symbols=["AAPL", "MSFT", "GOOGL"],
            start_date="2020-01-01",
            end_date="2020-01-03"
        )
        
        # Check if Ticker was called for each symbol
        assert mock_ticker.call_count == 3
        assert mock_history.call_count == 3
        
        # Check if results contain data for each symbol
        assert len(results) == 3
        assert "AAPL" in results
        assert "MSFT" in results
        assert "GOOGL" in results
        
        # Check if data files were saved
        data_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.parquet')]
        assert len(data_files) == 3


class TestCRSPDataSource:
    """Tests for the CRSPDataSource class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_source = CRSPDataSource(api_key="test_key", data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_fetch_data(self):
        """Test fetching data from CRSP."""
        # Fetch data
        data = self.data_source.fetch_data(
            dataset="daily_stock_file",
            start_date="2020-01-01",
            end_date="2020-01-03",
            symbols=["AAPL", "MSFT"],
            fields=["PRC", "VOL"]
        )
        
        # Check if data was returned
        assert isinstance(data, pd.DataFrame)
        assert isinstance(data.index, pd.MultiIndex)
        assert data.index.names == ["date", "symbol"]
        assert "PRC" in data.columns
        assert "VOL" in data.columns
        
        # Check if metadata was saved
        metadata_files = [f for f in os.listdir(self.temp_dir) if f.endswith('_metadata.json')]
        assert len(metadata_files) == 1
        
        # Check metadata content
        with open(os.path.join(self.temp_dir, metadata_files[0]), 'r') as f:
            metadata = json.load(f)
            assert metadata['source'] == "CRSP"
            assert metadata['dataset'] == "daily_stock_file"
            assert metadata['start_date'] == "2020-01-01"
            assert metadata['end_date'] == "2020-01-03"
            assert metadata['symbols'] == ["AAPL", "MSFT"]
            assert metadata['fields'] == ["PRC", "VOL"]
    
    def test_fetch_data_no_api_key(self):
        """Test fetching data without an API key."""
        # Create a data source without an API key
        data_source = CRSPDataSource(api_key=None, data_dir=self.temp_dir)
        
        # Attempt to fetch data
        with pytest.raises(ValueError, match="CRSP API key is required"):
            data_source.fetch_data(
                dataset="daily_stock_file",
                start_date="2020-01-01"
            )


class TestBloombergDataSource:
    """Tests for the BloombergDataSource class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_source = BloombergDataSource(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_fetch_data(self):
        """Test fetching data from Bloomberg."""
        # Fetch data
        data = self.data_source.fetch_data(
            securities=["AAPL US Equity", "MSFT US Equity"],
            fields=["PX_LAST", "PX_VOLUME"],
            start_date="2020-01-01",
            end_date="2020-01-03"
        )
        
        # Check if data was returned
        assert isinstance(data, pd.DataFrame)
        assert isinstance(data.index, pd.MultiIndex)
        assert data.index.names == ["date", "security"]
        assert "PX_LAST" in data.columns
        assert "PX_VOLUME" in data.columns
        
        # Check if metadata was saved
        metadata_files = [f for f in os.listdir(self.temp_dir) if f.endswith('_metadata.json')]
        assert len(metadata_files) == 1
        
        # Check metadata content
        with open(os.path.join(self.temp_dir, metadata_files[0]), 'r') as f:
            metadata = json.load(f)
            assert metadata['source'] == "Bloomberg"
            assert metadata['securities'] == ["AAPL US Equity", "MSFT US Equity"]
            assert metadata['fields'] == ["PX_LAST", "PX_VOLUME"]
            assert metadata['start_date'] == "2020-01-01"
            assert metadata['end_date'] == "2020-01-03"
            assert metadata['periodicity'] == "DAILY"


class TestMarketDataProcessor:
    """Tests for the MarketDataProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.processor = MarketDataProcessor(self.temp_dir)
        
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=10)
        self.sample_data = pd.DataFrame({
            'open': [100 + i for i in range(10)],
            'high': [105 + i for i in range(10)],
            'low': [95 + i for i in range(10)],
            'close': [102 + i for i in range(10)],
            'volume': [1000000 + i * 100000 for i in range(10)]
        }, index=dates)
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_normalize_data(self):
        """Test normalizing data."""
        # Normalize with z-score
        z_normalized = self.processor.normalize_data(
            self.sample_data, method="zscore")
        
        # Check if data was normalized
        assert isinstance(z_normalized, pd.DataFrame)
        assert z_normalized.shape == self.sample_data.shape
        assert abs(z_normalized.mean().mean()) < 1e-10  # Mean should be close to 0
        assert abs(z_normalized.std().mean() - 1.0) < 1e-10  # Std should be close to 1
        
        # Normalize with min-max
        mm_normalized = self.processor.normalize_data(
            self.sample_data, method="minmax")
        
        # Check if data was normalized
        assert isinstance(mm_normalized, pd.DataFrame)
        assert mm_normalized.shape == self.sample_data.shape
        assert mm_normalized.min().min() >= 0.0  # Min should be >= 0
        assert mm_normalized.max().max() <= 1.0  # Max should be <= 1
        
        # Test with invalid method
        with pytest.raises(ValueError):
            self.processor.normalize_data(self.sample_data, method="invalid")
    
    def test_resample_data(self):
        """Test resampling data."""
        # Resample to weekly
        weekly = self.processor.resample_data(self.sample_data, freq="W")
        
        # Check if data was resampled
        assert isinstance(weekly, pd.DataFrame)
        assert len(weekly) < len(self.sample_data)  # Should have fewer rows
        assert set(weekly.columns) == set(self.sample_data.columns)  # Same columns
        
        # Resample to monthly
        monthly = self.processor.resample_data(self.sample_data, freq="M")
        
        # Check if data was resampled
        assert isinstance(monthly, pd.DataFrame)
        assert len(monthly) < len(weekly)  # Should have fewer rows than weekly
        
        # Test with non-datetime index
        data_with_int_index = self.sample_data.reset_index()
        with pytest.raises(ValueError):
            self.processor.resample_data(data_with_int_index, freq="W")
    
    def test_calculate_returns(self):
        """Test calculating returns."""
        # Calculate returns
        returns = self.processor.calculate_returns(
            self.sample_data, price_col="close", periods=[1, 5])
        
        # Check if returns were calculated
        assert isinstance(returns, pd.DataFrame)
        assert returns.shape[0] == self.sample_data.shape[0]
        assert "return_1d" in returns.columns
        assert "return_5d" in returns.columns
        
        # First return should be NaN
        assert pd.isna(returns["return_1d"].iloc[0])
        
        # Returns should be calculated correctly
        assert returns["return_1d"].iloc[1] == (
            self.sample_data["close"].iloc[1] / self.sample_data["close"].iloc[0] - 1)
    
    def test_calculate_technical_indicators(self):
        """Test calculating technical indicators."""
        # Calculate indicators
        indicators = self.processor.calculate_technical_indicators(self.sample_data)
        
        # Check if indicators were calculated
        assert isinstance(indicators, pd.DataFrame)
        assert indicators.shape[0] == self.sample_data.shape[0]
        
        # Check if specific indicators are present
        assert "ma_5" in indicators.columns
        assert "ma_10" in indicators.columns
        assert "rsi_14" in indicators.columns
        assert "bb_middle" in indicators.columns
        assert "bb_upper" in indicators.columns
        assert "bb_lower" in indicators.columns
        assert "macd" in indicators.columns
        assert "macd_signal" in indicators.columns
        assert "macd_hist" in indicators.columns
    
    def test_detect_outliers(self):
        """Test detecting outliers."""
        # Create data with outliers
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[data_with_outliers.index[5], "close"] = 200  # Outlier
        
        # Detect outliers with z-score
        outliers_z = self.processor.detect_outliers(
            data_with_outliers, columns=["close"], method="zscore")
        
        # Check if outliers were detected
        assert isinstance(outliers_z, pd.DataFrame)
        assert "close_outlier" in outliers_z.columns
        assert outliers_z["close_outlier"].iloc[5] == True  # Should detect the outlier
        
        # Detect outliers with IQR
        outliers_iqr = self.processor.detect_outliers(
            data_with_outliers, columns=["close"], method="iqr")
        
        # Check if outliers were detected
        assert isinstance(outliers_iqr, pd.DataFrame)
        assert "close_outlier" in outliers_iqr.columns
        assert outliers_iqr["close_outlier"].iloc[5] == True  # Should detect the outlier
        
        # Test with invalid method
        with pytest.raises(ValueError):
            self.processor.detect_outliers(
                data_with_outliers, method="invalid")
    
    def test_handle_missing_data(self):
        """Test handling missing data."""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[data_with_missing.index[3:5], "close"] = np.nan
        
        # Handle missing data with forward fill
        filled_ff = self.processor.handle_missing_data(
            data_with_missing, method="forward_fill")
        
        # Check if missing data was handled
        assert isinstance(filled_ff, pd.DataFrame)
        assert filled_ff.shape == data_with_missing.shape
        assert not filled_ff["close"].isna().any()  # No NaNs
        assert filled_ff["close"].iloc[3] == data_with_missing["close"].iloc[2]  # Forward filled
        
        # Handle missing data with backward fill
        filled_bf = self.processor.handle_missing_data(
            data_with_missing, method="backward_fill")
        
        # Check if missing data was handled
        assert isinstance(filled_bf, pd.DataFrame)
        assert not filled_bf["close"].isna().any()  # No NaNs
        assert filled_bf["close"].iloc[4] == data_with_missing["close"].iloc[5]  # Backward filled
        
        # Handle missing data with linear interpolation
        filled_li = self.processor.handle_missing_data(
            data_with_missing, method="linear_interpolation")
        
        # Check if missing data was handled
        assert isinstance(filled_li, pd.DataFrame)
        assert not filled_li["close"].isna().any()  # No NaNs
        
        # Handle missing data by dropping
        filled_drop = self.processor.handle_missing_data(
            data_with_missing, method="drop")
        
        # Check if missing data was handled
        assert isinstance(filled_drop, pd.DataFrame)
        assert len(filled_drop) < len(data_with_missing)  # Rows were dropped
        assert not filled_drop["close"].isna().any()  # No NaNs
        
        # Test with invalid method
        with pytest.raises(ValueError):
            self.processor.handle_missing_data(
                data_with_missing, method="invalid")
    
    def test_save_processed_data(self):
        """Test saving processed data."""
        # Save data in different formats
        self.processor.save_processed_data(
            self.sample_data, "test_processed.csv")
        self.processor.save_processed_data(
            self.sample_data, "test_processed.parquet")
        self.processor.save_processed_data(
            self.sample_data, "test_processed.pkl")
        self.processor.save_processed_data(
            self.sample_data, "test_processed_no_ext")
        
        # Check if files were created
        assert os.path.exists(os.path.join(self.temp_dir, "test_processed.csv"))
        assert os.path.exists(os.path.join(self.temp_dir, "test_processed.parquet"))
        assert os.path.exists(os.path.join(self.temp_dir, "test_processed.pkl"))
        assert os.path.exists(os.path.join(self.temp_dir, "test_processed_no_ext.parquet"))


class TestMarketReconstructionEngine:
    """Tests for the MarketReconstructionEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.engine = MarketReconstructionEngine(self.temp_dir)
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_reconstruct_pre_dji_data(self):
        """Test reconstructing pre-DJI data."""
        # Reconstruct data
        data = self.engine.reconstruct_pre_dji_data(
            start_year=1800,
            end_year=1850,
            resolution="monthly"
        )
        
        # Check if data was reconstructed
        assert isinstance(data, pd.DataFrame)
        assert "index_value" in data.columns
        assert "volume" in data.columns
        
        # Check date range
        assert data.index[0].year == 1800
        assert data.index[-1].year == 1850
        
        # Check if metadata was saved
        metadata_files = [f for f in os.listdir(self.temp_dir) if f.endswith('_metadata.json')]
        assert len(metadata_files) == 1
        
        # Check metadata content
        with open(os.path.join(self.temp_dir, metadata_files[0]), 'r') as f:
            metadata = json.load(f)
            assert metadata['method'] == "synthetic_reconstruction"
            assert metadata['start_year'] == 1800
            assert metadata['end_year'] == 1850
            assert metadata['resolution'] == "monthly"
    
    def test_reconstruct_with_different_resolutions(self):
        """Test reconstructing with different resolutions."""
        # Reconstruct with daily resolution
        daily = self.engine.reconstruct_pre_dji_data(
            start_year=1890,
            end_year=1895,
            resolution="daily"
        )
        
        # Reconstruct with weekly resolution
        weekly = self.engine.reconstruct_pre_dji_data(
            start_year=1890,
            end_year=1895,
            resolution="weekly"
        )
        
        # Reconstruct with monthly resolution
        monthly = self.engine.reconstruct_pre_dji_data(
            start_year=1890,
            end_year=1895,
            resolution="monthly"
        )
        
        # Check if data was reconstructed with different resolutions
        assert len(daily) > len(weekly)
        assert len(weekly) > len(monthly)
        
        # Test with invalid resolution
        with pytest.raises(ValueError):
            self.engine.reconstruct_pre_dji_data(
                start_year=1890,
                end_year=1895,
                resolution="invalid"
            )
    
    def test_save_reconstructed_data(self):
        """Test saving reconstructed data."""
        # Reconstruct data
        data = self.engine.reconstruct_pre_dji_data(
            start_year=1890,
            end_year=1895,
            resolution="monthly"
        )
        
        # Save data in different formats
        self.engine.save_reconstructed_data(
            data, "test_reconstructed.csv")
        self.engine.save_reconstructed_data(
            data, "test_reconstructed.parquet")
        self.engine.save_reconstructed_data(
            data, "test_reconstructed.pkl")
        self.engine.save_reconstructed_data(
            data, "test_reconstructed_no_ext")
        
        # Check if files were created
        assert os.path.exists(os.path.join(self.temp_dir, "test_reconstructed.csv"))
        assert os.path.exists(os.path.join(self.temp_dir, "test_reconstructed.parquet"))
        assert os.path.exists(os.path.join(self.temp_dir, "test_reconstructed.pkl"))
        assert os.path.exists(os.path.join(self.temp_dir, "test_reconstructed_no_ext.parquet"))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
