"""
Time Series Decomposition Module for the Cosmic Market Oracle.

This module implements specialized time series decomposition techniques for
analyzing market-astrological data, including seasonal-trend decomposition,
wavelet analysis, and empirical mode decomposition.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from statsmodels.tsa.seasonal import STL, seasonal_decompose
import pywt
from PyEMD import EMD, EEMD, CEEMDAN
import matplotlib.pyplot as plt
from scipy import signal
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesDecomposer:
    """Base class for time series decomposition techniques."""
    
    def __init__(self, name: str = "base_decomposer"):
        """
        Initialize the time series decomposer.
        
        Args:
            name: Name of the decomposer
        """
        self.name = name
    
    def decompose(self, time_series: pd.Series, **kwargs) -> Dict[str, pd.Series]:
        """
        Decompose a time series into components.
        
        Args:
            time_series: Time series to decompose
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of decomposed components
        """
        raise NotImplementedError("Subclasses must implement decompose")
    
    def plot_decomposition(self, components: Dict[str, pd.Series], 
                          figsize: Tuple[int, int] = (12, 8),
                          title: str = None) -> plt.Figure:
        """
        Plot the decomposition components.
        
        Args:
            components: Dictionary of decomposed components
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        n_components = len(components)
        fig, axes = plt.subplots(n_components, 1, figsize=figsize, sharex=True)
        
        if n_components == 1:
            axes = [axes]
        
        for i, (name, series) in enumerate(components.items()):
            axes[i].plot(series)
            axes[i].set_title(name)
            axes[i].grid(True)
        
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f"{self.name} Decomposition")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig


class ClassicalDecomposer(TimeSeriesDecomposer):
    """Classical time series decomposition using additive or multiplicative models."""
    
    def __init__(self, model: str = "additive", period: Optional[int] = None):
        """
        Initialize the classical decomposer.
        
        Args:
            model: Decomposition model ('additive' or 'multiplicative')
            period: Period of the seasonality (optional)
        """
        super().__init__(name=f"classical_{model}")
        self.model = model
        self.period = period
    
    def decompose(self, time_series: pd.Series, **kwargs) -> Dict[str, pd.Series]:
        """
        Decompose a time series using classical decomposition.
        
        Args:
            time_series: Time series to decompose
            **kwargs: Additional parameters including:
                - period: Period of the seasonality (overrides constructor)
                
        Returns:
            Dictionary with 'trend', 'seasonal', and 'residual' components
        """
        # Get period from kwargs or use the one from constructor
        period = kwargs.get("period", self.period)
        
        # If period is not provided, try to infer it
        if period is None:
            period = self._infer_period(time_series)
            logger.info(f"Inferred period: {period}")
        
        # Perform decomposition
        result = seasonal_decompose(
            time_series, 
            model=self.model, 
            period=period,
            extrapolate_trend='freq'
        )
        
        # Create components dictionary
        components = {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid
        }
        
        return components
    
    def _infer_period(self, time_series: pd.Series) -> int:
        """
        Infer the period of the time series using spectral analysis.
        
        Args:
            time_series: Time series to analyze
            
        Returns:
            Inferred period
        """
        # Fill missing values
        ts_filled = time_series.fillna(method='ffill').fillna(method='bfill')
        
        # Get frequencies and spectrum
        freqs, spectrum = signal.periodogram(ts_filled.values)
        
        # Find the frequency with the highest power
        if len(spectrum) > 1:
            dominant_idx = np.argmax(spectrum[1:]) + 1  # Skip the DC component
            dominant_freq = freqs[dominant_idx]
            
            if dominant_freq > 0:
                period = int(1.0 / dominant_freq)
                return max(2, min(period, len(time_series) // 2))
        
        # Default to weekly seasonality if inference fails
        return 7


class STLDecomposer(TimeSeriesDecomposer):
    """Seasonal-Trend decomposition using LOESS (STL)."""
    
    def __init__(self, period: Optional[int] = None, seasonal: int = 7, 
                trend: Optional[int] = None, low_pass: Optional[int] = None,
                seasonal_deg: int = 1, trend_deg: int = 1, low_pass_deg: int = 1,
                robust: bool = False):
        """
        Initialize the STL decomposer.
        
        Args:
            period: Period of the seasonality (optional)
            seasonal: Length of the seasonal smoother
            trend: Length of the trend smoother
            low_pass: Length of the low-pass filter
            seasonal_deg: Degree of seasonal LOESS
            trend_deg: Degree of trend LOESS
            low_pass_deg: Degree of low-pass LOESS
            robust: Flag indicating whether to use robust fitting
        """
        super().__init__(name="stl")
        self.period = period
        self.seasonal = seasonal
        self.trend = trend
        self.low_pass = low_pass
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = robust
    
    def decompose(self, time_series: pd.Series, **kwargs) -> Dict[str, pd.Series]:
        """
        Decompose a time series using STL.
        
        Args:
            time_series: Time series to decompose
            **kwargs: Additional parameters including:
                - period: Period of the seasonality (overrides constructor)
                
        Returns:
            Dictionary with 'trend', 'seasonal', and 'residual' components
        """
        # Get period from kwargs or use the one from constructor
        period = kwargs.get("period", self.period)
        
        # If period is not provided, try to infer it
        if period is None:
            period = self._infer_period(time_series)
            logger.info(f"Inferred period: {period}")
        
        # Perform decomposition
        stl = STL(
            time_series,
            period=period,
            seasonal=self.seasonal,
            trend=self.trend,
            low_pass=self.low_pass,
            seasonal_deg=self.seasonal_deg,
            trend_deg=self.trend_deg,
            low_pass_deg=self.low_pass_deg,
            robust=self.robust
        )
        
        result = stl.fit()
        
        # Create components dictionary
        components = {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid
        }
        
        return components
    
    def _infer_period(self, time_series: pd.Series) -> int:
        """
        Infer the period of the time series using spectral analysis.
        
        Args:
            time_series: Time series to analyze
            
        Returns:
            Inferred period
        """
        # Fill missing values
        ts_filled = time_series.fillna(method='ffill').fillna(method='bfill')
        
        # Get frequencies and spectrum
        freqs, spectrum = signal.periodogram(ts_filled.values)
        
        # Find the frequency with the highest power
        if len(spectrum) > 1:
            dominant_idx = np.argmax(spectrum[1:]) + 1  # Skip the DC component
            dominant_freq = freqs[dominant_idx]
            
            if dominant_freq > 0:
                period = int(1.0 / dominant_freq)
                return max(2, min(period, len(time_series) // 2))
        
        # Default to weekly seasonality if inference fails
        return 7


class WaveletDecomposer(TimeSeriesDecomposer):
    """Wavelet-based time series decomposition."""
    
    def __init__(self, wavelet: str = 'db8', level: Optional[int] = None, 
                mode: str = 'symmetric'):
        """
        Initialize the wavelet decomposer.
        
        Args:
            wavelet: Wavelet to use (e.g., 'db8', 'sym4', 'coif3')
            level: Decomposition level (optional)
            mode: Signal extension mode
        """
        super().__init__(name=f"wavelet_{wavelet}")
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
    
    def decompose(self, time_series: pd.Series, **kwargs) -> Dict[str, pd.Series]:
        """
        Decompose a time series using wavelet decomposition.
        
        Args:
            time_series: Time series to decompose
            **kwargs: Additional parameters including:
                - level: Decomposition level (overrides constructor)
                
        Returns:
            Dictionary with approximation and detail components
        """
        # Get level from kwargs or use the one from constructor
        level = kwargs.get("level", self.level)
        
        # If level is not provided, calculate the maximum possible level
        if level is None:
            level = pywt.dwt_max_level(len(time_series), self.wavelet)
            level = min(level, 5)  # Limit to 5 levels for interpretability
            logger.info(f"Using wavelet decomposition level: {level}")
        
        # Fill missing values
        ts_filled = time_series.fillna(method='ffill').fillna(method='bfill')
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(ts_filled.values, self.wavelet, mode=self.mode, level=level)
        
        # Create components dictionary
        components = {}
        
        # Add approximation component
        components["approximation"] = pd.Series(
            pywt.upcoef('a', coeffs[0], self.wavelet, level=level, take=len(ts_filled)),
            index=time_series.index
        )
        
        # Add detail components
        for i, detail_coeff in enumerate(coeffs[1:], 1):
            components[f"detail_{i}"] = pd.Series(
                pywt.upcoef('d', detail_coeff, self.wavelet, level=level-(i-1), take=len(ts_filled)),
                index=time_series.index
            )
        
        return components


class EMDDecomposer(TimeSeriesDecomposer):
    """Empirical Mode Decomposition (EMD) for time series."""
    
    def __init__(self, method: str = "emd", num_imfs: Optional[int] = None,
                ensemble_size: int = 100, noise_width: float = 0.2):
        """
        Initialize the EMD decomposer.
        
        Args:
            method: EMD method ('emd', 'eemd', or 'ceemdan')
            num_imfs: Number of IMFs to extract (optional)
            ensemble_size: Ensemble size for EEMD/CEEMDAN
            noise_width: Noise width for EEMD/CEEMDAN
        """
        super().__init__(name=method.upper())
        self.method = method.lower()
        self.num_imfs = num_imfs
        self.ensemble_size = ensemble_size
        self.noise_width = noise_width
    
    def decompose(self, time_series: pd.Series, **kwargs) -> Dict[str, pd.Series]:
        """
        Decompose a time series using EMD.
        
        Args:
            time_series: Time series to decompose
            **kwargs: Additional parameters including:
                - num_imfs: Number of IMFs to extract (overrides constructor)
                
        Returns:
            Dictionary with IMF components
        """
        # Get num_imfs from kwargs or use the one from constructor
        num_imfs = kwargs.get("num_imfs", self.num_imfs)
        
        # Fill missing values
        ts_filled = time_series.fillna(method='ffill').fillna(method='bfill')
        
        # Select EMD method
        if self.method == "eemd":
            emd = EEMD(trials=self.ensemble_size, noise_width=self.noise_width)
        elif self.method == "ceemdan":
            emd = CEEMDAN(trials=self.ensemble_size, noise_width=self.noise_width)
        else:  # Default to standard EMD
            emd = EMD()
        
        # Perform decomposition
        if num_imfs is not None:
            imfs = emd.emd(ts_filled.values, max_imf=num_imfs)
        else:
            imfs = emd.emd(ts_filled.values)
        
        # Create components dictionary
        components = {}
        
        # Add IMF components
        for i, imf in enumerate(imfs):
            components[f"IMF_{i+1}"] = pd.Series(imf, index=time_series.index)
        
        # Add residue
        residue = ts_filled.values - np.sum(imfs, axis=0)
        components["residue"] = pd.Series(residue, index=time_series.index)
        
        return components


class MultiResolutionDecomposer(TimeSeriesDecomposer):
    """Multi-resolution decomposition combining multiple techniques."""
    
    def __init__(self, methods: List[str] = None):
        """
        Initialize the multi-resolution decomposer.
        
        Args:
            methods: List of decomposition methods to use
        """
        super().__init__(name="multi_resolution")
        
        if methods is None:
            methods = ["stl", "wavelet", "emd"]
        
        self.methods = methods
        
        # Initialize decomposers
        self.decomposers = {}
        
        for method in methods:
            if method.lower() == "stl":
                self.decomposers["stl"] = STLDecomposer()
            elif method.lower() == "wavelet":
                self.decomposers["wavelet"] = WaveletDecomposer()
            elif method.lower() in ["emd", "eemd", "ceemdan"]:
                self.decomposers[method.lower()] = EMDDecomposer(method=method.lower())
            elif method.lower() == "classical":
                self.decomposers["classical"] = ClassicalDecomposer()
    
    def decompose(self, time_series: pd.Series, **kwargs) -> Dict[str, Dict[str, pd.Series]]:
        """
        Decompose a time series using multiple techniques.
        
        Args:
            time_series: Time series to decompose
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with decomposition results for each method
        """
        results = {}
        
        for method, decomposer in self.decomposers.items():
            try:
                results[method] = decomposer.decompose(time_series, **kwargs)
                logger.info(f"Successfully decomposed using {method}")
            except Exception as e:
                logger.error(f"Error decomposing with {method}: {e}")
        
        return results
    
    def plot_all_decompositions(self, results: Dict[str, Dict[str, pd.Series]], 
                               figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot all decomposition results.
        
        Args:
            results: Dictionary with decomposition results for each method
            figsize: Base figure size
        """
        for method, components in results.items():
            decomposer = self.decomposers[method]
            fig = decomposer.plot_decomposition(
                components, 
                figsize=figsize,
                title=f"{method.upper()} Decomposition"
            )
            plt.show()


class AstrologicalCycleDecomposer(TimeSeriesDecomposer):
    """Specialized decomposer for astrological cycles in financial data."""
    
    def __init__(self, cycles: Dict[str, float] = None):
        """
        Initialize the astrological cycle decomposer.
        
        Args:
            cycles: Dictionary of cycle names and their periods in days
        """
        super().__init__(name="astrological_cycle")
        
        # Default astrological cycles if none provided
        if cycles is None:
            cycles = {
                "lunar": 29.53059,         # Lunar cycle (synodic month)
                "mercury": 115.88,         # Mercury synodic cycle
                "venus": 583.92,           # Venus synodic cycle
                "mars": 779.94,            # Mars synodic cycle
                "jupiter": 398.88,         # Jupiter synodic cycle
                "saturn": 378.09,          # Saturn synodic cycle
                "jupiter_saturn": 7253.45, # Jupiter-Saturn conjunction cycle (~20 years)
                "nodal": 6798.38           # Lunar nodal cycle (~18.6 years)
            }
        
        self.cycles = cycles
    
    def decompose(self, time_series: pd.Series, **kwargs) -> Dict[str, pd.Series]:
        """
        Decompose a time series into astrological cycle components.
        
        Args:
            time_series: Time series to decompose
            **kwargs: Additional parameters including:
                - cycles: Dictionary of cycle names and periods (overrides constructor)
                
        Returns:
            Dictionary with cycle components and residual
        """
        # Get cycles from kwargs or use the ones from constructor
        cycles = kwargs.get("cycles", self.cycles)
        
        # Fill missing values
        ts_filled = time_series.fillna(method='ffill').fillna(method='bfill')
        
        # Create components dictionary
        components = {}
        
        # Extract each cycle component
        residual = ts_filled.copy()
        
        for cycle_name, cycle_period in cycles.items():
            # Skip cycles longer than half the time series
            if cycle_period > len(ts_filled) / 2:
                logger.info(f"Skipping {cycle_name} cycle (period {cycle_period} days) - too long for the time series")
                continue
            
            try:
                # Create STL decomposer for this cycle
                cycle_decomposer = STLDecomposer(period=int(cycle_period))
                
                # Decompose
                cycle_components = cycle_decomposer.decompose(residual)
                
                # Add seasonal component to results
                components[f"{cycle_name}_cycle"] = cycle_components["seasonal"]
                
                # Update residual
                residual = residual - cycle_components["seasonal"]
            except Exception as e:
                logger.warning(f"Could not extract {cycle_name} cycle: {e}")
        
        # Add trend and residual
        components["trend"] = residual.rolling(window=min(30, len(residual)//10 or 1), center=True).mean()
        components["residual"] = residual - components["trend"]
        
        return components


# Example usage
if __name__ == "__main__":
    # Generate sample data with multiple cycles
    np.random.seed(42)
    
    # Create time index
    days = 365 * 2
    index = pd.date_range(start="2020-01-01", periods=days, freq="D")
    
    # Create time series with multiple cycles
    t = np.arange(days)
    lunar_cycle = 5 * np.sin(2 * np.pi * t / 29.53059)
    annual_cycle = 10 * np.sin(2 * np.pi * t / 365.25)
    trend = 0.01 * t
    noise = np.random.normal(0, 1, days)
    
    # Combine components
    y = trend + lunar_cycle + annual_cycle + noise
    
    # Create pandas Series
    time_series = pd.Series(y, index=index)
    
    # Create decomposers
    classical = ClassicalDecomposer(period=30)
    stl = STLDecomposer(period=30)
    wavelet = WaveletDecomposer()
    emd = EMDDecomposer()
    astro = AstrologicalCycleDecomposer()
    
    # Decompose
    classical_components = classical.decompose(time_series)
    stl_components = stl.decompose(time_series)
    wavelet_components = wavelet.decompose(time_series)
    emd_components = emd.decompose(time_series)
    astro_components = astro.decompose(time_series)
    
    # Plot decompositions
    classical.plot_decomposition(classical_components)
    stl.plot_decomposition(stl_components)
    wavelet.plot_decomposition(wavelet_components)
    emd.plot_decomposition(emd_components)
    astro.plot_decomposition(astro_components)
    
    plt.show()
