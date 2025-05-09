# Cosmic Market Oracle - Market Features Module

"""
This module transforms raw market data into machine learning-compatible features.
It includes technical indicators, volatility measures, and other market-specific
transformations relevant to financial prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats


class MarketFeatureGenerator:
    """Generates machine learning features from market data."""
    
    def __init__(self):
        """
        Initialize the market feature generator.
        """
        pass
    
    def add_price_features(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Add basic price-derived features to the dataframe.
        
        Args:
            df: DataFrame with market data
            price_col: Name of the price column
            
        Returns:
            DataFrame with added features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate returns at different timeframes
        result['return_1d'] = result[price_col].pct_change(1)
        result['return_5d'] = result[price_col].pct_change(5)
        result['return_10d'] = result[price_col].pct_change(10)
        result['return_20d'] = result[price_col].pct_change(20)
        result['return_60d'] = result[price_col].pct_change(60)
        
        # Calculate log returns
        result['log_return_1d'] = np.log(result[price_col] / result[price_col].shift(1))
        
        # Price relative to moving averages
        for window in [10, 20, 50, 200]:
            ma_col = f'ma_{window}'
            result[ma_col] = result[price_col].rolling(window=window).mean()
            result[f'price_to_{ma_col}'] = result[price_col] / result[ma_col] - 1
        
        # Moving average crossovers (as binary indicators)
        result['ma_10_20_crossover'] = ((result['ma_10'] > result['ma_20']) & 
                                      (result['ma_10'].shift(1) <= result['ma_20'].shift(1))).astype(float)
        result['ma_10_20_crossunder'] = ((result['ma_10'] < result['ma_20']) & 
                                       (result['ma_10'].shift(1) >= result['ma_20'].shift(1))).astype(float)
        result['ma_50_200_crossover'] = ((result['ma_50'] > result['ma_200']) & 
                                       (result['ma_50'].shift(1) <= result['ma_200'].shift(1))).astype(float)
        result['ma_50_200_crossunder'] = ((result['ma_50'] < result['ma_200']) & 
                                        (result['ma_50'].shift(1) >= result['ma_200'].shift(1))).astype(float)
        
        # Price momentum
        result['momentum_20d'] = result[price_col] - result[price_col].shift(20)
        result['momentum_60d'] = result[price_col] - result[price_col].shift(60)
        
        return result
    
    def add_volatility_features(self, df: pd.DataFrame, returns_col: str = 'return_1d') -> pd.DataFrame:
        """
        Add volatility-related features to the dataframe.
        
        Args:
            df: DataFrame with market data
            returns_col: Name of the returns column
            
        Returns:
            DataFrame with added features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Historical volatility (standard deviation of returns)
        for window in [10, 20, 60]:
            result[f'volatility_{window}d'] = result[returns_col].rolling(window=window).std()
        
        # Normalized volatility (current volatility relative to longer-term)
        result['volatility_ratio_10_60'] = result['volatility_10d'] / result['volatility_60d']
        
        # Volatility trend
        result['volatility_trend_20d'] = result['volatility_20d'] - result['volatility_20d'].shift(20)
        
        # High-Low range volatility
        if 'high' in df.columns and 'low' in df.columns:
            result['daily_range'] = (df['high'] - df['low']) / df['close']
            result['daily_range_ma10'] = result['daily_range'].rolling(window=10).mean()
        
        return result
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to the dataframe.
        
        Args:
            df: DataFrame with market data (must have OHLC columns)
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Relative Strength Index (RSI)
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain_14 = gain.rolling(window=14).mean()
        avg_loss_14 = loss.rolling(window=14).mean()
        
        rs_14 = avg_gain_14 / avg_loss_14
        result['rsi_14'] = 100 - (100 / (1 + rs_14))
        
        # Moving Average Convergence Divergence (MACD)
        ema_12 = result['close'].ewm(span=12, adjust=False).mean()
        ema_26 = result['close'].ewm(span=26, adjust=False).mean()
        result['macd'] = ema_12 - ema_26
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        # Bollinger Bands
        result['bb_middle_20'] = result['close'].rolling(window=20).mean()
        result['bb_std_20'] = result['close'].rolling(window=20).std()
        result['bb_upper_20'] = result['bb_middle_20'] + (result['bb_std_20'] * 2)
        result['bb_lower_20'] = result['bb_middle_20'] - (result['bb_std_20'] * 2)
        result['bb_width_20'] = (result['bb_upper_20'] - result['bb_lower_20']) / result['bb_middle_20']
        
        # Stochastic Oscillator
        n = 14
        result['lowest_low'] = result['low'].rolling(window=n).min()
        result['highest_high'] = result['high'].rolling(window=n).max()
        result['stoch_k'] = 100 * ((result['close'] - result['lowest_low']) / 
                                 (result['highest_high'] - result['lowest_low']))
        result['stoch_d'] = result['stoch_k'].rolling(window=3).mean()
        
        # Average Directional Index (ADX)
        # This is a simplified version; a full implementation would be more complex
        tr1 = result['high'] - result['low']
        tr2 = abs(result['high'] - result['close'].shift(1))
        tr3 = abs(result['low'] - result['close'].shift(1))
        result['true_range'] = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        result['atr_14'] = result['true_range'].rolling(window=14).mean()
        
        # On-Balance Volume (OBV)
        result['obv'] = np.where(result['close'] > result['close'].shift(1), 
                               result['volume'], 
                               np.where(result['close'] < result['close'].shift(1), 
                                       -result['volume'], 0)).cumsum()
        
        # Chaikin Money Flow (CMF)
        mf_multiplier = ((result['close'] - result['low']) - (result['high'] - result['close'])) / (result['high'] - result['low'])
        mf_volume = mf_multiplier * result['volume']
        result['cmf_20'] = mf_volume.rolling(window=20).sum() / result['volume'].rolling(window=20).sum()
        
        return result
    
    def add_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical time-based features to the dataframe.
        
        Args:
            df: DataFrame with market data and datetime index
            
        Returns:
            DataFrame with added cyclical features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure we have a datetime index
        if not isinstance(result.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Day of week (0=Monday, 6=Sunday)
        result['day_of_week'] = result.index.dayofweek
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        # Month of year
        result['month'] = result.index.month
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        # Quarter
        result['quarter'] = result.index.quarter
        result['quarter_sin'] = np.sin(2 * np.pi * result['quarter'] / 4)
        result['quarter_cos'] = np.cos(2 * np.pi * result['quarter'] / 4)
        
        # Day of month
        result['day_of_month'] = result.index.day
        result['day_of_month_sin'] = np.sin(2 * np.pi * result['day_of_month'] / 31)
        result['day_of_month_cos'] = np.cos(2 * np.pi * result['day_of_month'] / 31)
        
        # Week of year
        result['week_of_year'] = result.index.isocalendar().week
        result['week_of_year_sin'] = np.sin(2 * np.pi * result['week_of_year'] / 52)
        result['week_of_year_cos'] = np.cos(2 * np.pi * result['week_of_year'] / 52)
        
        return result
    
    def add_market_regime_features(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Add market regime identification features to the dataframe.
        
        Args:
            df: DataFrame with market data
            price_col: Name of the price column
            
        Returns:
            DataFrame with added regime features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Trend strength indicators
        result['price_200d_ratio'] = result[price_col] / result[price_col].rolling(window=200).mean() - 1
        
        # Volatility regime
        vol_20 = result['return_1d'].rolling(window=20).std() if 'return_1d' in result.columns else \
                 result[price_col].pct_change().rolling(window=20).std()
        vol_100 = result['return_1d'].rolling(window=100).std() if 'return_1d' in result.columns else \
                  result[price_col].pct_change().rolling(window=100).std()
        
        # Classify volatility regime (high, medium, low)
        vol_ratio = vol_20 / vol_100
        result['volatility_regime'] = pd.cut(
            vol_ratio, 
            bins=[-float('inf'), 0.75, 1.25, float('inf')],
            labels=['low', 'normal', 'high']
        ).astype(str)
        
        # One-hot encode the volatility regime
        for regime in ['low', 'normal', 'high']:
            result[f'vol_regime_{regime}'] = (result['volatility_regime'] == regime).astype(float)
        
        # Trend regime based on moving averages
        ma_50 = result[price_col].rolling(window=50).mean()
        ma_200 = result[price_col].rolling(window=200).mean()
        
        # Classify trend regime (bull, bear, sideways)
        result['trend_regime'] = 'sideways'  # Default
        result.loc[ma_50 > ma_200 * 1.05, 'trend_regime'] = 'bull'
        result.loc[ma_50 < ma_200 * 0.95, 'trend_regime'] = 'bear'
        
        # One-hot encode the trend regime
        for regime in ['bull', 'bear', 'sideways']:
            result[f'trend_regime_{regime}'] = (result['trend_regime'] == regime).astype(float)
        
        return result
    
    def add_market_breadth_features(self, df: pd.DataFrame, breadth_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add market breadth indicators to the dataframe.
        
        Args:
            df: DataFrame with market data
            breadth_data: DataFrame with market breadth data (e.g., advance-decline, new highs-lows)
            
        Returns:
            DataFrame with added breadth features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Merge breadth data with the main dataframe
        if not breadth_data.empty:
            # Ensure both dataframes have compatible indices
            breadth_data = breadth_data.reindex(result.index, method='ffill')
            
            # Add breadth columns to the result
            for col in breadth_data.columns:
                result[f'breadth_{col}'] = breadth_data[col]
            
            # Calculate derived breadth indicators if the necessary columns exist
            if 'breadth_advances' in result.columns and 'breadth_declines' in result.columns:
                # Advance-Decline Line
                result['ad_line'] = (result['breadth_advances'] - result['breadth_declines']).cumsum()
                
                # Advance-Decline Ratio
                result['ad_ratio'] = result['breadth_advances'] / result['breadth_declines']
                
                # McClellan Oscillator (simplified version)
                ad_diff = result['breadth_advances'] - result['breadth_declines']
                ema_19 = ad_diff.ewm(span=19, adjust=False).mean()
                ema_39 = ad_diff.ewm(span=39, adjust=False).mean()
                result['mcclellan_oscillator'] = ema_19 - ema_39
            
            if 'breadth_new_highs' in result.columns and 'breadth_new_lows' in result.columns:
                # High-Low Index
                result['high_low_index'] = result['breadth_new_highs'] / (result['breadth_new_highs'] + result['breadth_new_lows'])
                
                # High-Low Difference
                result['high_low_diff'] = result['breadth_new_highs'] - result['breadth_new_lows']
        
        return result
    
    def generate_all_features(self, market_data: pd.DataFrame, breadth_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate all market features for the given data.
        
        Args:
            market_data: DataFrame with market OHLCV data
            breadth_data: Optional DataFrame with market breadth data
            
        Returns:
            DataFrame with all market features
        """
        # Start with basic price features
        result = self.add_price_features(market_data)
        
        # Add volatility features
        result = self.add_volatility_features(result)
        
        # Add technical indicators
        result = self.add_technical_indicators(result)
        
        # Add cyclical time features
        result = self.add_cycle_features(result)
        
        # Add market regime features
        result = self.add_market_regime_features(result)
        
        # Add market breadth features if data is provided
        if breadth_data is not None:
            result = self.add_market_breadth_features(result, breadth_data)
        
        return result