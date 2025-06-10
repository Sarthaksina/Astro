"""
Financial Data Acquisition Module for the Cosmic Market Oracle.

This module provides tools for acquiring and processing financial data from various sources:
1. Historical market data from institutional sources (CRSP, Global Financial Data)
2. Bloomberg Terminal data for DJI constituents
3. Reconstruction of pre-DJI market activity
4. Multi-resolution financial data processing
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import requests
import yfinance as yf
from pathlib import Path
import json

# Configure logging
from src.utils.logger import get_logger # Changed from setup_logger
logger = get_logger(__name__) # Changed from setup_logger
from src.utils.file_io import save_dataframe # Added import

class FinancialDataSource:
    """Base class for financial data sources."""
    
    def __init__(self, name: str, data_dir: str = "data/financial"):
        """
        Initialize the financial data source.
        
        Args:
            name: Name of the data source
            data_dir: Directory for storing data
        """
        self.name = name
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def fetch_data(self, **kwargs) -> pd.DataFrame:
        """
        Fetch data from the source.
        
        Args:
            **kwargs: Source-specific parameters
            
        Returns:
            DataFrame with the fetched data
        """
        raise NotImplementedError("Subclasses must implement fetch_data")
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """
        Save data to disk.
        
        Args:
            data: DataFrame to save
            filename: Name of the file
        """
        filepath_str = os.path.join(self.data_dir, filename)
        if save_dataframe(df=data, file_path_str=filepath_str, create_dirs=True):
            # Original logger message can be kept if desired, or rely on save_dataframe's logging.
            # logger.info(f"Saved data to {filepath_str} (via utility)")
            pass # save_dataframe already logs success
        else:
            logger.error(f"Failed to save data to {filepath_str} using utility.")
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from disk.
        
        Args:
            filename: Name of the file
            
        Returns:
            DataFrame with the loaded data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found")
        
        # Determine file format based on extension
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif filepath.endswith('.parquet'):
            data = pd.read_parquet(filepath)
        elif filepath.endswith('.pickle') or filepath.endswith('.pkl'):
            data = pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unsupported file format for {filepath}")
            
        logger.info(f"Loaded data from {filepath}")
        return data


class YahooFinanceDataSource(FinancialDataSource):
    """Yahoo Finance data source."""
    
    def __init__(self, data_dir: str = "data/financial/yahoo"):
        """
        Initialize the Yahoo Finance data source.
        
        Args:
            data_dir: Directory for storing data
        """
        super().__init__("Yahoo Finance", data_dir)
        
    def fetch_data(self, 
                 symbol: str, 
                 start_date: str, 
                 end_date: str = None,
                 interval: str = "1d") -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            interval: Data interval (1d, 1wk, 1mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Save metadata
            metadata = {
                "source": self.name,
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date or datetime.now().strftime("%Y-%m-%d"),
                "interval": interval,
                "rows": len(data),
                "columns": list(data.columns),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save metadata
            metadata_path = os.path.join(
                self.data_dir, 
                f"{symbol}_{start_date}_{end_date or 'now'}_metadata.json"
            )
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def fetch_multiple_symbols(self, 
                             symbols: List[str], 
                             start_date: str, 
                             end_date: str = None,
                             interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            interval: Data interval (1d, 1wk, 1mo)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )
                results[symbol] = data
                
                # Save individual data
                filename = f"{symbol}_{start_date}_{end_date or 'now'}.parquet"
                self.save_data(data, filename)
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
                
        return results


class CRSPDataSource(FinancialDataSource):
    """CRSP (Center for Research in Security Prices) data source."""
    
    def __init__(self, 
               api_key: str = None,
               data_dir: str = "data/financial/crsp"):
        """
        Initialize the CRSP data source.
        
        Args:
            api_key: API key for accessing CRSP data
            data_dir: Directory for storing data
        """
        super().__init__("CRSP", data_dir)
        self.api_key = api_key or os.environ.get("CRSP_API_KEY")
        
        if not self.api_key:
            logger.warning("No CRSP API key provided. Set CRSP_API_KEY environment variable.")
    
    def fetch_data(self, 
                 dataset: str,
                 start_date: str, 
                 end_date: str = None,
                 symbols: List[str] = None,
                 fields: List[str] = None) -> pd.DataFrame:
        """
        Fetch data from CRSP.
        
        Args:
            dataset: CRSP dataset name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            symbols: List of ticker symbols
            fields: List of fields to fetch
            
        Returns:
            DataFrame with requested data
        """
        if not self.api_key:
            raise ValueError("CRSP API key is required")
            
        logger.info(f"Fetching {dataset} data from CRSP")
        
        # This is a placeholder for the actual CRSP API implementation
        # In a real implementation, this would use the CRSP API client
        
        # For now, we'll simulate the data
        dates = pd.date_range(start=start_date, end=end_date or datetime.now())
        
        if not symbols:
            symbols = ["AAPL", "MSFT", "GOOGL"]
            
        if not fields:
            fields = ["PRC", "VOL", "RET"]
            
        # Create a multi-index DataFrame
        index = pd.MultiIndex.from_product(
            [dates, symbols],
            names=["date", "symbol"]
        )
        
        # Generate random data for each field
        data = {}
        for field in fields:
            if field == "PRC":  # Price
                data[field] = np.random.normal(100, 10, len(index))
            elif field == "VOL":  # Volume
                data[field] = np.random.lognormal(15, 1, len(index))
            elif field == "RET":  # Return
                data[field] = np.random.normal(0.0005, 0.02, len(index))
            else:
                data[field] = np.random.random(len(index))
                
        df = pd.DataFrame(data, index=index)
        
        # Save metadata
        metadata = {
            "source": self.name,
            "dataset": dataset,
            "start_date": start_date,
            "end_date": end_date or datetime.now().strftime("%Y-%m-%d"),
            "symbols": symbols,
            "fields": fields,
            "rows": len(df),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save metadata
        metadata_path = os.path.join(
            self.data_dir, 
            f"{dataset}_{start_date}_{end_date or 'now'}_metadata.json"
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return df


class BloombergDataSource(FinancialDataSource):
    """Bloomberg Terminal data source."""
    
    def __init__(self, 
               terminal_path: str = None,
               data_dir: str = "data/financial/bloomberg"):
        """
        Initialize the Bloomberg data source.
        
        Args:
            terminal_path: Path to Bloomberg Terminal executable
            data_dir: Directory for storing data
        """
        super().__init__("Bloomberg", data_dir)
        self.terminal_path = terminal_path
        
    def fetch_data(self, 
                 securities: List[str],
                 fields: List[str],
                 start_date: str, 
                 end_date: str = None,
                 periodicity: str = "DAILY") -> pd.DataFrame:
        """
        Fetch data from Bloomberg Terminal.
        
        Args:
            securities: List of Bloomberg securities
            fields: List of Bloomberg fields
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            periodicity: Data periodicity (DAILY, WEEKLY, MONTHLY)
            
        Returns:
            DataFrame with requested data
        """
        logger.info(f"Fetching Bloomberg data for {len(securities)} securities")
        
        # This is a placeholder for the actual Bloomberg API implementation
        # In a real implementation, this would use the Bloomberg API client
        
        # For now, we'll simulate the data
        dates = pd.date_range(start=start_date, end=end_date or datetime.now())
        
        # Create a multi-index DataFrame
        index = pd.MultiIndex.from_product(
            [dates, securities],
            names=["date", "security"]
        )
        
        # Generate random data for each field
        data = {}
        for field in fields:
            if field == "PX_LAST":  # Last price
                data[field] = np.random.normal(100, 10, len(index))
            elif field == "PX_VOLUME":  # Volume
                data[field] = np.random.lognormal(15, 1, len(index))
            elif field == "PX_HIGH":  # High price
                base = np.random.normal(100, 10, len(index))
                data[field] = base * (1 + np.random.random(len(index)) * 0.05)
            elif field == "PX_LOW":  # Low price
                base = np.random.normal(100, 10, len(index))
                data[field] = base * (1 - np.random.random(len(index)) * 0.05)
            else:
                data[field] = np.random.random(len(index))
                
        df = pd.DataFrame(data, index=index)
        
        # Save metadata
        metadata = {
            "source": self.name,
            "securities": securities,
            "fields": fields,
            "start_date": start_date,
            "end_date": end_date or datetime.now().strftime("%Y-%m-%d"),
            "periodicity": periodicity,
            "rows": len(df),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save metadata
        metadata_path = os.path.join(
            self.data_dir, 
            f"bloomberg_{start_date}_{end_date or 'now'}_metadata.json"
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return df


class MarketDataProcessor:
    """Processor for market data."""
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize the market data processor.
        
        Args:
            data_dir: Directory for storing processed data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def normalize_data(self, 
                     data: pd.DataFrame, 
                     method: str = "zscore") -> pd.DataFrame:
        """
        Normalize market data.
        
        Args:
            data: DataFrame with market data
            method: Normalization method (zscore, minmax)
            
        Returns:
            Normalized DataFrame
        """
        if method == "zscore":
            return (data - data.mean()) / data.std()
        elif method == "minmax":
            return (data - data.min()) / (data.max() - data.min())
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
    
    def resample_data(self, 
                    data: pd.DataFrame, 
                    freq: str) -> pd.DataFrame:
        """
        Resample data to a different frequency.
        
        Args:
            data: DataFrame with market data
            freq: Target frequency (D, W, M, Q, Y)
            
        Returns:
            Resampled DataFrame
        """
        # Ensure the index is a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex")
            
        # Define aggregation functions for OHLCV data
        agg_funcs = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Filter aggregation functions to include only columns in the data
        agg_funcs = {k: v for k, v in agg_funcs.items() if k in data.columns}
        
        # Add any other columns with a default aggregation of 'last'
        for col in data.columns:
            if col not in agg_funcs:
                agg_funcs[col] = 'last'
                
        # Resample the data
        resampled = data.resample(freq).agg(agg_funcs)
        
        return resampled
    
    def calculate_returns(self, 
                        data: pd.DataFrame, 
                        price_col: str = "close",
                        periods: List[int] = [1, 5, 20]) -> pd.DataFrame:
        """
        Calculate returns over different periods.
        
        Args:
            data: DataFrame with market data
            price_col: Column name for price data
            periods: List of periods for return calculation
            
        Returns:
            DataFrame with returns
        """
        result = data.copy()
        
        for period in periods:
            col_name = f"return_{period}d"
            result[col_name] = data[price_col].pct_change(period)
            
        return result
    
    def calculate_technical_indicators(self, 
                                     data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        result = data.copy()
        
        # Moving Averages
        for window in [5, 10, 20, 50, 200]:
            result[f'ma_{window}'] = data['close'].rolling(window=window).mean()
            
        # Relative Strength Index (RSI)
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        result['bb_middle'] = data['close'].rolling(window=20).mean()
        result['bb_std'] = data['close'].rolling(window=20).std()
        result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
        result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']
        
        # MACD
        result['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        result['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        return result
    
    def detect_outliers(self, 
                      data: pd.DataFrame, 
                      columns: List[str] = None,
                      method: str = "zscore",
                      threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in market data.
        
        Args:
            data: DataFrame with market data
            columns: Columns to check for outliers
            method: Outlier detection method (zscore, iqr)
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier flags
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
            
        result = pd.DataFrame(index=data.index)
        
        for col in columns:
            if method == "zscore":
                z_scores = (data[col] - data[col].mean()) / data[col].std()
                result[f"{col}_outlier"] = (abs(z_scores) > threshold)
            elif method == "iqr":
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                result[f"{col}_outlier"] = (
                    (data[col] < lower_bound) | (data[col] > upper_bound)
                )
            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")
                
        return result
    
    def handle_missing_data(self, 
                          data: pd.DataFrame, 
                          method: str = "forward_fill") -> pd.DataFrame:
        """
        Handle missing data in market data.
        
        Args:
            data: DataFrame with market data
            method: Method for handling missing data
            
        Returns:
            DataFrame with handled missing data
        """
        if method == "forward_fill":
            return data.ffill()
        elif method == "backward_fill":
            return data.bfill()
        elif method == "linear_interpolation":
            return data.interpolate(method='linear')
        elif method == "drop":
            return data.dropna()
        else:
            raise ValueError(f"Unsupported missing data handling method: {method}")
    
    def save_processed_data(self, 
                          data: pd.DataFrame, 
                          filename: str):
        """
        Save processed data to disk.
        
        Args:
            data: DataFrame to save
            filename: Name of the file
        """
        filepath_str = os.path.join(self.data_dir, filename)
        if save_dataframe(df=data, file_path_str=filepath_str, create_dirs=True):
            # logger.info(f"Saved processed data to {filepath_str} (via utility)")
            pass # save_dataframe already logs success
        else:
            logger.error(f"Failed to save processed data to {filepath_str} using utility.")


class MarketReconstructionEngine:
    """Engine for reconstructing historical market data."""
    
    def __init__(self, data_dir: str = "data/reconstructed"):
        """
        Initialize the market reconstruction engine.
        
        Args:
            data_dir: Directory for storing reconstructed data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def reconstruct_pre_dji_data(self, 
                               start_year: int, 
                               end_year: int = 1896,
                               resolution: str = "monthly") -> pd.DataFrame:
        """
        Reconstruct market data before the formal DJI existence.
        
        Args:
            start_year: Start year for reconstruction
            end_year: End year for reconstruction
            resolution: Data resolution (daily, weekly, monthly)
            
        Returns:
            DataFrame with reconstructed data
        """
        logger.info(f"Reconstructing market data from {start_year} to {end_year}")
        
        # This is a placeholder for the actual reconstruction algorithm
        # In a real implementation, this would use historical economic data,
        # commodity prices, and other proxies to estimate market activity
        
        # Generate dates based on resolution
        if resolution == "daily":
            dates = pd.date_range(
                start=f"{start_year}-01-01", 
                end=f"{end_year}-12-31", 
                freq='D'
            )
        elif resolution == "weekly":
            dates = pd.date_range(
                start=f"{start_year}-01-01", 
                end=f"{end_year}-12-31", 
                freq='W'
            )
        elif resolution == "monthly":
            dates = pd.date_range(
                start=f"{start_year}-01-01", 
                end=f"{end_year}-12-31", 
                freq='M'
            )
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")
            
        # Generate synthetic data
        # We'll use a simple model with trend, seasonality, and noise
        
        # Parameters
        trend_growth = 0.03  # 3% annual growth
        seasonality_amplitude = 0.05  # 5% seasonal variation
        noise_level = 0.02  # 2% random noise
        
        # Time components
        t = np.arange(len(dates)) / 365  # Time in years
        
        # Trend component (exponential growth)
        trend = np.exp(trend_growth * t)
        
        # Seasonality component (annual cycle)
        month_of_year = np.array([d.month for d in dates])
        seasonality = 1 + seasonality_amplitude * np.sin(2 * np.pi * month_of_year / 12)
        
        # Noise component
        noise = 1 + np.random.normal(0, noise_level, len(dates))
        
        # Combine components
        index_value = 10 * trend * seasonality * noise  # Start at 10
        
        # Create DataFrame
        data = pd.DataFrame({
            'index_value': index_value,
            'volume': np.random.lognormal(10, 1, len(dates)) * trend
        }, index=dates)
        
        # Add metadata
        metadata = {
            "method": "synthetic_reconstruction",
            "start_year": start_year,
            "end_year": end_year,
            "resolution": resolution,
            "parameters": {
                "trend_growth": trend_growth,
                "seasonality_amplitude": seasonality_amplitude,
                "noise_level": noise_level
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save metadata
        metadata_path = os.path.join(
            self.data_dir, 
            f"reconstruction_{start_year}_{end_year}_metadata.json"
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return data
    
    def save_reconstructed_data(self, 
                              data: pd.DataFrame, 
                              filename: str):
        """
        Save reconstructed data to disk.
        
        Args:
            data: DataFrame to save
            filename: Name of the file
        """
        filepath_str = os.path.join(self.data_dir, filename)
        if save_dataframe(df=data, file_path_str=filepath_str, create_dirs=True):
            pass # save_dataframe already logs success
        else:
            logger.error(f"Failed to save reconstructed data to {filepath_str} using utility.")


# Additional utility functions for market data acquisition

def calculate_market_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional market metrics from OHLCV data.
    
    Args:
        data: DataFrame with market data
        
    Returns:
        DataFrame with additional metrics calculated
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Ensure we have the necessary columns
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col.lower() in df.columns for col in required_columns):
        # Try to standardize column names
        for old, new in zip(['Open', 'High', 'Low', 'Close'], required_columns):
            if old in df.columns:
                df[new] = df[old]
    
    # Calculate daily returns
    if 'close' in df.columns:
        df['daily_return'] = df['close'].pct_change()
        
        # Calculate volatility (20-day rolling standard deviation of returns)
        df['volatility_20d'] = df['daily_return'].rolling(window=20).std()
        
        # Calculate moving averages
        for window in [10, 50, 200]:
            df[f'ma_{window}d'] = df['close'].rolling(window=window).mean()
    
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
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    reconstruction_end = pd.to_datetime("1896-05-26")  # DJI was created on May 26, 1896
    
    # Determine if we need reconstructed data
    need_reconstruction = start < reconstruction_end
    
    # Get modern data (from 1896 onwards)
    modern_start = max(start, pd.to_datetime("1896-05-26"))
    if modern_start <= end:
        # Use YahooFinanceDataSource to fetch modern data
        yahoo_source = YahooFinanceDataSource()
        modern_data = yahoo_source.fetch_data(
            symbol="^DJI",
            start_date=modern_start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d")
        )
        
        # Calculate additional metrics
        if not modern_data.empty:
            modern_data = calculate_market_metrics(modern_data)
    else:
        modern_data = pd.DataFrame()
    
    # Get reconstructed data if needed
    if need_reconstruction:
        reconstruction_start_year = start.year
        reconstruction_end_year = min(1896, end.year)
        
        # Use MarketReconstructionEngine to reconstruct historical data
        reconstruction_engine = MarketReconstructionEngine()
        reconstructed_data = reconstruction_engine.reconstruct_pre_dji_data(
            start_year=reconstruction_start_year,
            end_year=reconstruction_end_year,
            resolution="daily"
        )
        
        # Filter by exact dates
        reconstructed_data = reconstructed_data[
            (reconstructed_data.index >= start) & 
            (reconstructed_data.index <= min(end, reconstruction_end))
        ]
        
        # Calculate additional metrics
        if not reconstructed_data.empty:
            reconstructed_data = calculate_market_metrics(reconstructed_data)
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


# Example usage
if __name__ == "__main__":
    # Yahoo Finance example
    yahoo_source = YahooFinanceDataSource()
    yahoo_data = yahoo_source.fetch_data(
        symbol="AAPL",
        start_date="2020-01-01",
        end_date="2020-12-31"
    )
    yahoo_source.save_data(yahoo_data, "AAPL_2020.parquet")
    
    # CRSP example (placeholder)
    crsp_source = CRSPDataSource(api_key="your_api_key")
    crsp_data = crsp_source.fetch_data(
        dataset="daily_stock_file",
        start_date="2020-01-01",
        end_date="2020-12-31",
        symbols=["AAPL", "MSFT", "GOOGL"],
        fields=["PRC", "VOL", "RET"]
    )
    crsp_source.save_data(crsp_data, "crsp_daily_2020.parquet")
    
    # Market data processing example
    processor = MarketDataProcessor()
    processed_data = processor.calculate_technical_indicators(yahoo_data)
    processor.save_processed_data(processed_data, "AAPL_2020_processed.parquet")
    
    # Market reconstruction example
    reconstruction_engine = MarketReconstructionEngine()
    reconstructed_data = reconstruction_engine.reconstruct_pre_dji_data(
        start_year=1800,
        end_year=1896,
        resolution="monthly"
    )
    reconstruction_engine.save_reconstructed_data(
        reconstructed_data, 
        "market_1800_1896_monthly.parquet"
    )
    
    # Get complete DJI history example
    complete_dji = get_complete_dji_history(
        start_date="1850-01-01",
        end_date="2023-01-01"
    )
    print(f"Complete DJI history contains {len(complete_dji)} records from {complete_dji.index.min()} to {complete_dji.index.max()}")

