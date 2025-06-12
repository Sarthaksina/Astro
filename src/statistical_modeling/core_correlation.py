"""
Core Correlation Module for the Cosmic Market Oracle.

This module implements basic correlation analysis techniques for time series data,
including cross-correlation with lag optimization and visualization tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from scipy import signal, stats
import matplotlib.pyplot as plt
# import logging # Removed
from ..utils.logger import get_logger # Changed path for src
from statsmodels.tsa.stattools import ccf

# Configure logging
# logging.basicConfig( # Removed
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = get_logger(__name__) # Changed


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cross-correlation between two time series.
    
    Args:
        x: First time series
        y: Second time series
        max_lag: Maximum lag to consider (default: None, uses min(len(x), len(y)))
        
    Returns:
        Tuple of (lags, correlation values)
    """
    if max_lag is None:
        max_lag = min(len(x), len(y)) - 1
    
    # Ensure arrays are 1D
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Calculate cross-correlation
    correlation = ccf(x, y, adjusted=False)
    
    # Create lag array
    lags = np.arange(-max_lag, max_lag + 1)
    
    # Trim correlation array to match lags
    if len(correlation) > len(lags):
        mid = len(correlation) // 2
        correlation = correlation[mid - max_lag:mid + max_lag + 1]
    
    return lags, correlation


def find_optimal_lag(x: np.ndarray, y: np.ndarray, max_lag: int = None) -> Tuple[int, float]:
    """
    Find the lag that maximizes the absolute correlation between two time series.
    
    Args:
        x: First time series
        y: Second time series
        max_lag: Maximum lag to consider (default: None, uses min(len(x), len(y)))
        
    Returns:
        Tuple of (optimal lag, correlation at optimal lag)
    """
    lags, correlation = cross_correlation(x, y, max_lag)
    
    # Find lag with maximum absolute correlation
    abs_corr = np.abs(correlation)
    optimal_idx = np.argmax(abs_corr)
    optimal_lag = lags[optimal_idx]
    optimal_corr = correlation[optimal_idx]
    
    return optimal_lag, optimal_corr


def rolling_correlation(x: pd.Series, y: pd.Series, window: int = 30) -> pd.Series:
    """
    Calculate rolling correlation between two time series.
    
    Args:
        x: First time series
        y: Second time series
        window: Rolling window size
        
    Returns:
        Series of rolling correlation values
    """
    # Align series
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        x, y = x.align(y, join='inner')
    
    # Calculate rolling correlation
    rolling_corr = x.rolling(window=window).corr(y)
    
    return rolling_corr


def windowed_correlation(x: np.ndarray, y: np.ndarray, window_size: int, 
                        step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate correlation in sliding windows across two time series.
    
    Args:
        x: First time series
        y: Second time series
        window_size: Size of the sliding window
        step_size: Step size for sliding the window
        
    Returns:
        Tuple of (window centers, correlation values)
    """
    # Ensure arrays are 1D
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Check if arrays have the same length
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")
    
    # Calculate number of windows
    n_windows = (len(x) - window_size) // step_size + 1
    
    # Initialize arrays for results
    window_centers = np.zeros(n_windows)
    correlations = np.zeros(n_windows)
    
    # Calculate correlation in each window
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        window_centers[i] = start + window_size // 2
        correlations[i] = np.corrcoef(x[start:end], y[start:end])[0, 1]
    
    return window_centers, correlations


def detrend_series(series: Union[np.ndarray, pd.Series], method: str = 'linear') -> Union[np.ndarray, pd.Series]:
    """
    Detrend a time series.
    
    Args:
        series: Time series to detrend
        method: Detrending method ('linear', 'constant', or 'polynomial')
        
    Returns:
        Detrended time series
    """
    # Convert to numpy array for processing
    is_pandas = isinstance(series, pd.Series)
    index = series.index if is_pandas else None
    values = series.values if is_pandas else series
    
    if method == 'linear':
        detrended = signal.detrend(values, type='linear')
    elif method == 'constant':
        detrended = signal.detrend(values, type='constant')
    elif method == 'polynomial':
        # Fit a 2nd degree polynomial and subtract
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 2)
        trend = np.polyval(coeffs, x)
        detrended = values - trend
    else:
        raise ValueError(f"Unknown detrending method: {method}")
    
    # Convert back to pandas Series if input was a Series
    if is_pandas:
        detrended = pd.Series(detrended, index=index)
    
    return detrended


def standardize_series(series: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """
    Standardize a time series (zero mean, unit variance).
    
    Args:
        series: Time series to standardize
        
    Returns:
        Standardized time series
    """
    # Convert to numpy array for processing
    is_pandas = isinstance(series, pd.Series)
    index = series.index if is_pandas else None
    values = series.values if is_pandas else series
    
    # Standardize
    mean = np.mean(values)
    std = np.std(values)
    
    if std > 0:
        standardized = (values - mean) / std
    else:
        standardized = values - mean
    
    # Convert back to pandas Series if input was a Series
    if is_pandas:
        standardized = pd.Series(standardized, index=index)
    
    return standardized


def plot_cross_correlation(lags: np.ndarray, correlation: np.ndarray, 
                          title: str = "Cross-Correlation", 
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot cross-correlation results.
    
    Args:
        lags: Lag values
        correlation: Correlation values
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot correlation
    ax.stem(lags, correlation, markerfmt='ro', basefmt=' ')
    
    # Find and highlight maximum correlation
    abs_corr = np.abs(correlation)
    max_idx = np.argmax(abs_corr)
    max_lag = lags[max_idx]
    max_corr = correlation[max_idx]
    
    ax.plot(max_lag, max_corr, 'bo', markersize=10, label=f'Max at lag {max_lag} (r={max_corr:.2f})')
    
    # Add confidence intervals (95%)
    n = len(correlation)
    conf_level = 1.96 / np.sqrt(n)
    ax.axhline(y=conf_level, color='gray', linestyle='--', alpha=0.7, label='95% Confidence')
    ax.axhline(y=-conf_level, color='gray', linestyle='--', alpha=0.7)
    
    # Add labels and legend
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    return fig


def plot_rolling_correlation(rolling_corr: pd.Series, 
                            title: str = "Rolling Correlation", 
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot rolling correlation results.
    
    Args:
        rolling_corr: Rolling correlation series
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot rolling correlation
    rolling_corr.plot(ax=ax)
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
    
    # Add confidence intervals (95%)
    window_size = rolling_corr.rolling(1).count().iloc[0] if not np.isnan(rolling_corr.iloc[0]) else None
    if window_size:
        conf_level = 1.96 / np.sqrt(window_size)
        ax.axhline(y=conf_level, color='gray', linestyle='--', alpha=0.7, label='95% Confidence')
        ax.axhline(y=-conf_level, color='gray', linestyle='--', alpha=0.7)
    
    # Add labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Correlation')
    ax.set_title(title)
    ax.grid(True)
    
    return fig


def plot_windowed_correlation(window_centers: np.ndarray, correlations: np.ndarray,
                             title: str = "Windowed Correlation",
                             figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot windowed correlation results.
    
    Args:
        window_centers: Window center positions
        correlations: Correlation values for each window
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot windowed correlation
    ax.plot(window_centers, correlations, 'b-')
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
    
    # Add labels
    ax.set_xlabel('Window Center')
    ax.set_ylabel('Correlation')
    ax.set_title(title)
    ax.grid(True)
    
    return fig


def correlation_significance_test(r: float, n: int, alpha: float = 0.05) -> Tuple[float, bool]:
    """
    Test the significance of a correlation coefficient.
    
    Args:
        r: Correlation coefficient
        n: Sample size
        alpha: Significance level
        
    Returns:
        Tuple of (p-value, is_significant)
    """
    # Calculate t-statistic
    t = r * np.sqrt((n - 2) / (1 - r**2))
    
    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - stats.t.cdf(abs(t), df=n-2))
    
    # Check significance
    is_significant = p_value < alpha
    
    return p_value, is_significant


def correlation_matrix(data: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix for a DataFrame.
    
    Args:
        data: DataFrame with variables
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
        Correlation matrix
    """
    return data.corr(method=method)


def plot_correlation_matrix(corr_matrix: pd.DataFrame, 
                           title: str = "Correlation Matrix",
                           figsize: Tuple[int, int] = (10, 8),
                           cmap: str = 'coolwarm',
                           annot: bool = True) -> plt.Figure:
    """
    Plot correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        annot: Whether to annotate cells
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot correlation matrix
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns)
    ax.set_yticklabels(corr_matrix.index)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate cells
    if annot:
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                if not mask[i, j]:  # Skip upper triangle
                    ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                           ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white")
    
    # Add title
    ax.set_title(title)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


# Example usage
if __name__ == "__main__":
    # Generate sample data with correlation
    np.random.seed(42)
    
    # Create time index
    days = 365
    index = pd.date_range(start="2020-01-01", periods=days, freq="D")
    
    # Create time series with lag relationship
    t = np.arange(days)
    x = np.sin(2 * np.pi * t / 30) + 0.1 * np.random.randn(days)
    y = np.roll(x, 5) + 0.2 * np.random.randn(days)  # y is x with 5-day lag and noise
    
    # Create pandas Series
    x_series = pd.Series(x, index=index, name="X")
    y_series = pd.Series(y, index=index, name="Y")
    
    # Calculate cross-correlation
    lags, correlation = cross_correlation(x, y, max_lag=30)
    
    # Find optimal lag
    optimal_lag, optimal_corr = find_optimal_lag(x, y, max_lag=30)
    print(f"Optimal lag: {optimal_lag}, Correlation: {optimal_corr:.4f}")
    
    # Calculate rolling correlation
    rolling_corr = rolling_correlation(x_series, y_series, window=30)
    
    # Calculate windowed correlation
    window_centers, windowed_corr = windowed_correlation(x, y, window_size=60, step_size=10)
    
    # Test correlation significance
    p_value, is_significant = correlation_significance_test(optimal_corr, len(x))
    print(f"p-value: {p_value:.4f}, Significant: {is_significant}")
    
    # Plot results
    plot_cross_correlation(lags, correlation, title="Cross-Correlation between X and Y")
    plot_rolling_correlation(rolling_corr, title="30-Day Rolling Correlation")
    plot_windowed_correlation(window_centers, windowed_corr, title="Windowed Correlation (60-day windows)")
    
    # Create correlation matrix
    data = pd.DataFrame({"X": x_series, "Y": y_series})
    corr_matrix = correlation_matrix(data)
    plot_correlation_matrix(corr_matrix, title="Correlation Matrix")
    
    plt.show()
