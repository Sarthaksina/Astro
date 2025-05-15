# Cosmic Market Oracle - Market Data Acquisition Module

"""
This module handles the acquisition of historical market data from various sources,
with a focus on the Dow Jones Industrial Average (DJI) over a 200-year period.

It provides functions to fetch data from different sources, normalize the data,
and prepare it for integration with astrological data.
"""

import datetime
import logging
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_historical_data(
    start_date: str,
    end_date: str,
    symbol: str = "^DJI",
    source: str = "yahoo"
) -> pd.DataFrame:
    """
    Fetch historical market data from the specified source.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        symbol: Market symbol to fetch (default: ^DJI for Dow Jones)
        source: Data source (default: yahoo)
        
    Returns:
        DataFrame with historical market data
    """
    logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
    
    if source.lower() == "yahoo":
        return _fetch_from_yahoo(symbol, start_date, end_date)
    elif source.lower() == "csv":
        return _fetch_from_csv(symbol, start_date, end_date)
    else:
        raise ValueError(f"Unsupported data source: {source}")


def _fetch_from_yahoo(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance.
    
    Args:
        symbol: Market symbol to fetch
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with historical market data
    """
    try:
        # Fetch data from Yahoo Finance
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            logger.warning(f"No data retrieved for {symbol} from Yahoo Finance")
            return pd.DataFrame()
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Rename columns to standardized format
        data = data.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adjusted_close',
            'Volume': 'volume'
        })
        
        # Add symbol column
        data['symbol'] = symbol
        
        # Calculate additional metrics
        data = calculate_market_metrics(data)
        
        logger.info(f"Successfully fetched {len(data)} records for {symbol}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data from Yahoo Finance: {str(e)}")
        raise


def _fetch_from_csv(symbol: str, start_date: str, end_date: str, file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch historical data from a CSV file.
    
    Args:
        symbol: Market symbol to fetch
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        file_path: Path to CSV file (default: None, will use a default path)
        
    Returns:
        DataFrame with historical market data
    """
    if file_path is None:
        file_path = f"data/historical/{symbol.replace('^', '')}.csv"
    
    try:
        # Read CSV file
        data = pd.read_csv(file_path, parse_dates=['Date'])
        
        # Filter by date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        data = data[(data['Date'] >= start) & (data['Date'] <= end)]
        
        # Rename columns to standardized format
        data = data.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adjusted_close',
            'Volume': 'volume'
        })
        
        # Add symbol column
        data['symbol'] = symbol
        
        # Calculate additional metrics
        data = calculate_market_metrics(data)
        
        logger.info(f"Successfully loaded {len(data)} records from CSV for {symbol}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from CSV: {str(e)}")
        raise


def calculate_market_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional market metrics from OHLCV data.
    
    Args:
        data: DataFrame with market data
        
    Returns:
        DataFrame with additional metrics calculated
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate volatility (20-day rolling standard deviation of returns)
    df['volatility'] = df['daily_return'].rolling(window=20).std()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    
    return df


def reconstruct_historical_dji(start_year: int = 1800, end_year: int = 1896) -> pd.DataFrame:
    """
    Reconstruct historical DJI-equivalent index for pre-1896 period.
    This is a placeholder for the actual implementation that would use constituent stocks.
    
    Args:
        start_year: Start year for reconstruction
        end_year: End year for reconstruction
        
    Returns:
        DataFrame with reconstructed historical data
    """
    logger.info(f"Reconstructing historical DJI from {start_year} to {end_year}")
    
    # This is a simplified placeholder implementation
    # In a real implementation, this would use constituent stocks and historical records
    
    # Create date range
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Create synthetic data
    np.random.seed(42)  # For reproducibility
    n_days = len(date_range)
    
    # Start with a base value and apply random walk with drift
    base_value = 40.0  # Starting value
    drift = 0.0001  # Small upward drift
    volatility = 0.01  # Daily volatility
    
    # Generate log returns with drift and volatility
    log_returns = np.random.normal(drift, volatility, n_days)
    
    # Calculate price series
    price_series = base_value * np.exp(np.cumsum(log_returns))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'open': price_series,
        'close': price_series,
        'high': price_series * (1 + np.random.uniform(0, 0.02, n_days)),
        'low': price_series * (1 - np.random.uniform(0, 0.02, n_days)),
        'volume': np.random.randint(100000, 1000000, n_days),
        'symbol': '^DJI'
    })
    
    # Ensure high >= open, close and low <= open, close
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    # Calculate additional metrics
    df = calculate_market_metrics(df)
    
    logger.info(f"Successfully reconstructed {len(df)} days of historical DJI data")
    return df


def merge_historical_data(modern_data: pd.DataFrame, reconstructed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge modern and reconstructed historical data into a single dataset.
    
    Args:
        modern_data: DataFrame with modern market data
        reconstructed_data: DataFrame with reconstructed historical data
        
    Returns:
        DataFrame with merged data
    """
    # Ensure no duplicate dates
    combined_data = pd.concat([reconstructed_data, modern_data]).drop_duplicates(subset=['timestamp'])
    
    # Sort by date
    combined_data = combined_data.sort_values('timestamp')
    
    logger.info(f"Combined dataset contains {len(combined_data)} records from {combined_data['timestamp'].min()} to {combined_data['timestamp'].max()}")
    return combined_data


def get_complete_dji_history(start_date: str = "1800-01-01", end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Get complete DJI history by combining reconstructed pre-1896 data with modern data.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (default: today)
        
    Returns:
        DataFrame with complete historical data
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    reconstruction_end = pd.to_datetime("1896-05-26")  # DJI was created on May 26, 1896
    
    # Determine if we need reconstructed data
    need_reconstruction = start < reconstruction_end
    
    # Get modern data (from 1896 onwards)
    modern_start = max(start, pd.to_datetime("1896-05-26"))
    if modern_start <= end:
        modern_data = fetch_historical_data(
            start_date=modern_start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            symbol="^DJI",
            source="yahoo"
        )
    else:
        modern_data = pd.DataFrame()
    
    # Get reconstructed data if needed
    if need_reconstruction:
        reconstruction_start_year = start.year
        reconstruction_end_year = min(1896, end.year)
        reconstructed_data = reconstruct_historical_dji(
            start_year=reconstruction_start_year,
            end_year=reconstruction_end_year
        )
        
        # Filter by exact dates
        reconstructed_data = reconstructed_data[
            (reconstructed_data['timestamp'] >= start) & 
            (reconstructed_data['timestamp'] <= min(end, reconstruction_end))
        ]
    else:
        reconstructed_data = pd.DataFrame()
    
    # Merge datasets if both exist
    if not modern_data.empty and not reconstructed_data.empty:
        return merge_historical_data(modern_data, reconstructed_data)
    elif not modern_data.empty:
        return modern_data
    elif not reconstructed_data.empty:
        return reconstructed_data
    else:
        logger.warning("No data retrieved for the specified date range")
        return pd.DataFrame()
