# Cosmic Market Oracle - Tests for Market Data Acquisition Module

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.data_acquisition.market_data import (
    fetch_historical_data,
    process_market_data,
    calculate_returns,
    detect_market_regime,
    get_market_volatility
)


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    # Create a DataFrame with 30 days of mock market data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    data = {
        'Date': dates,
        'Open': np.linspace(100, 110, len(dates)) + np.random.normal(0, 1, len(dates)),
        'High': np.linspace(102, 112, len(dates)) + np.random.normal(0, 1, len(dates)),
        'Low': np.linspace(98, 108, len(dates)) + np.random.normal(0, 1, len(dates)),
        'Close': np.linspace(101, 111, len(dates)) + np.random.normal(0, 1, len(dates)),
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }
    
    # Ensure High is always the highest and Low is always the lowest
    for i in range(len(dates)):
        data['High'][i] = max(data['Open'][i], data['Close'][i], data['High'][i])
        data['Low'][i] = min(data['Open'][i], data['Close'][i], data['Low'][i])
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df


class TestMarketDataAcquisition:
    """Tests for market data acquisition functions."""
    
    @patch('src.data_acquisition.market_data.yf.download')
    def test_fetch_historical_data_yahoo(self, mock_download):
        """Test fetching historical data from Yahoo Finance."""
        # Setup mock return value
        mock_data = sample_market_data()
        mock_download.return_value = mock_data
        
        # Call the function
        result = fetch_historical_data(
            start_date="2023-01-01",
            end_date="2023-01-31",
            symbol="^DJI",
            source="yahoo"
        )
        
        # Verify the function called yf.download with correct parameters
        mock_download.assert_called_once_with(
            tickers="^DJI",
            start="2023-01-01",
            end="2023-01-31",
            auto_adjust=True
        )
        
        # Verify the result is the same as the mock data
        pd.testing.assert_frame_equal(result, mock_data)
    
    def test_process_market_data(self, sample_market_data):
        """Test processing of raw market data."""
        df = sample_market_data.copy()
        
        # Call the function
        processed_df = process_market_data(df)
        
        # Verify the processed data has the expected columns
        expected_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'Log_Returns', 'Volatility_20d'
        ]
        for col in expected_columns:
            assert col in processed_df.columns
        
        # Verify calculations are correct
        assert len(processed_df) == len(df)
        
        # Check that returns are calculated correctly
        manual_returns = df['Close'].pct_change()
        pd.testing.assert_series_equal(
            processed_df['Returns'].dropna(),
            manual_returns.dropna(),
            check_names=False
        )
    
    def test_calculate_returns(self, sample_market_data):
        """Test calculation of various return metrics."""
        df = sample_market_data.copy()
        
        # Call the function
        returns_df = calculate_returns(df)
        
        # Verify the returns data has the expected columns
        expected_columns = [
            'Daily_Return', 'Log_Return', 'Cumulative_Return'
        ]
        for col in expected_columns:
            assert col in returns_df.columns
        
        # Verify calculations are correct
        assert len(returns_df) == len(df)
        
        # Check that daily returns are calculated correctly
        manual_returns = df['Close'].pct_change()
        pd.testing.assert_series_equal(
            returns_df['Daily_Return'].dropna(),
            manual_returns.dropna(),
            check_names=False
        )
        
        # Check that log returns are calculated correctly
        manual_log_returns = np.log(df['Close'] / df['Close'].shift(1))
        pd.testing.assert_series_equal(
            returns_df['Log_Return'].dropna(),
            manual_log_returns.dropna(),
            check_names=False,
            rtol=1e-10
        )
    
    def test_detect_market_regime(self, sample_market_data):
        """Test detection of market regimes (bull/bear/sideways)."""
        df = sample_market_data.copy()
        
        # Add trend column (upward trend for testing)
        df['SMA_50'] = np.linspace(95, 115, len(df))
        df['SMA_200'] = np.linspace(90, 105, len(df))
        
        # Call the function
        regime = detect_market_regime(df)
        
        # Verify the result is a string
        assert isinstance(regime, str)
        assert regime in ['bull', 'bear', 'sideways']
        
        # For this specific test data, it should detect a bull market
        # since we set up SMA_50 > SMA_200 and both are rising
        assert regime == 'bull'
        
        # Test bear market detection
        df['SMA_50'] = np.linspace(115, 95, len(df))  # Downward trend
        df['SMA_200'] = np.linspace(120, 100, len(df))
        bear_regime = detect_market_regime(df)
        assert bear_regime == 'bear'
    
    def test_get_market_volatility(self, sample_market_data):
        """Test calculation of market volatility."""
        df = sample_market_data.copy()
        
        # Calculate log returns for the test
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Call the function with different window sizes
        vol_20 = get_market_volatility(df, window=20)
        vol_10 = get_market_volatility(df, window=10)
        
        # Verify the results are Series or DataFrames with the expected length
        assert isinstance(vol_20, (pd.Series, pd.DataFrame))
        assert isinstance(vol_10, (pd.Series, pd.DataFrame))
        
        # Shorter window should have more non-NaN values
        assert vol_10.count() >= vol_20.count()
        
        # Verify the calculation is correct by comparing with manual calculation
        manual_vol_20 = df['Log_Returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
        pd.testing.assert_series_equal(
            vol_20.dropna(),
            manual_vol_20.dropna(),
            check_names=False
        )
