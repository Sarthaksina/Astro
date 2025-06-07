"""
Cyclical Correlation Module for the Cosmic Market Oracle.

This module integrates various correlation analysis techniques specifically designed 
for cyclical data, particularly useful for analyzing relationships between 
astrological and financial time series.
"""

# Standard imports
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import logging

# Import specialized modules
from src.statistical_modeling.core_correlation import (
    cross_correlation, find_optimal_lag, rolling_correlation,
    windowed_correlation, plot_cross_correlation
)
from src.statistical_modeling.phase_analysis import (
    circular_correlation, phase_coupling, phase_locking_value,
    phase_synchronization_index, plot_phase_coupling
)
from src.statistical_modeling.spectral_analysis import (
    compute_coherence, compute_periodogram, compute_welch,
    compute_lombscargle, find_peaks, plot_coherence
)
from src.statistical_modeling.statistical_testing import (
    test_cyclical_correlation, test_phase_synchronization,
    test_circular_correlation, test_astrological_correlation
)

# Configure logging
from src.utils.logging_config import setup_logging
logger = setup_logging(__name__)

class CyclicalCorrelationAnalyzer:
    """
    Main class for analyzing cyclical correlations between time series.
    
    This class integrates various correlation techniques specifically designed
    for cyclical data, including cross-correlation, phase analysis, spectral
    analysis, and statistical testing.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the CyclicalCorrelationAnalyzer.
        
        Args:
            significance_level: Significance level for statistical tests
        """
        self.significance_level = significance_level
        self.results = {}
    
    def analyze_time_domain(self, x: np.ndarray, y: np.ndarray, 
                          max_lag: Optional[int] = None, 
                          window_size: int = 30) -> Dict:
        """
        Perform time domain correlation analysis.
        
        Args:
            x: First time series
            y: Second time series
            max_lag: Maximum lag to consider
            window_size: Window size for rolling correlation
        
        Returns:
            Dictionary with analysis results
        """
        # Ensure arrays are numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Calculate cross-correlation
        lags, correlation = cross_correlation(x, y, max_lag)
        
        # Find optimal lag
        optimal_lag, optimal_corr = find_optimal_lag(x, y, max_lag)
        
        # Calculate rolling correlation if inputs are pandas Series
        if isinstance(x, pd.Series) and isinstance(y, pd.Series):
            rolling_corr = rolling_correlation(x, y, window=window_size)
        else:
            # Convert to pandas Series for rolling correlation
            x_series = pd.Series(x)
            y_series = pd.Series(y)
            rolling_corr = rolling_correlation(x_series, y_series, window=window_size)
        
        # Calculate windowed correlation
        window_centers, windowed_corr = windowed_correlation(
            x, y, window_size=window_size, step_size=max(1, window_size // 10)
        )
        
        # Store results
        time_domain_results = {
            'cross_correlation': {
                'lags': lags,
                'correlation': correlation,
                'optimal_lag': optimal_lag,
                'optimal_correlation': optimal_corr
            },
            'rolling_correlation': rolling_corr,
            'windowed_correlation': {
                'window_centers': window_centers,
                'correlation': windowed_corr
            }
        }
        
        self.results['time_domain'] = time_domain_results
        return time_domain_results
    
    def analyze_phase_domain(self, x: np.ndarray, y: np.ndarray, 
                           method: str = 'circular') -> Dict:
        """
        Perform phase domain correlation analysis.
        
        Args:
            x: First time series
            y: Second time series
            method: Method for phase analysis ('circular', 'hilbert', or 'wavelet')
        
        Returns:
            Dictionary with analysis results
        """
        # Ensure arrays are numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        
        phase_domain_results = {}
        
        if method == 'circular' or method == 'all':
            # Convert to phase angles if not already
            if np.max(np.abs(x)) > 2*np.pi or np.max(np.abs(y)) > 2*np.pi:
                # Normalize to [-1, 1] and then convert to [-pi, pi]
                x_norm = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
                y_norm = 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1
                x_phase = np.arccos(x_norm)
                y_phase = np.arccos(y_norm)
            else:
                # Assume already in radians
                x_phase = x
                y_phase = y
            
            # Calculate circular correlation
            circ_corr = circular_correlation(x_phase, y_phase)
            
            # Calculate phase coupling metrics
            plv = phase_locking_value(x_phase, y_phase)
            psi = phase_synchronization_index(x_phase, y_phase)
            
            phase_domain_results['circular'] = {
                'circular_correlation': circ_corr,
                'phase_locking_value': plv,
                'phase_synchronization_index': psi
            }
        
        if method == 'hilbert' or method == 'all':
            # Import here to avoid circular imports
            from src.statistical_modeling.spectral_analysis import compute_hilbert_transform
            
            # Compute Hilbert transform
            _, _, x_phase = compute_hilbert_transform(x)
            _, _, y_phase = compute_hilbert_transform(y)
            
            # Calculate circular correlation
            circ_corr = circular_correlation(x_phase, y_phase)
            
            # Calculate phase coupling metrics
            plv = phase_locking_value(x_phase, y_phase)
            psi = phase_synchronization_index(x_phase, y_phase)
            
            phase_domain_results['hilbert'] = {
                'circular_correlation': circ_corr,
                'phase_locking_value': plv,
                'phase_synchronization_index': psi
            }
        
        if method == 'wavelet' or method == 'all':
            # Wavelet-based phase analysis requires more complex implementation
            # This is a placeholder for future implementation
            phase_domain_results['wavelet'] = {
                'message': 'Wavelet-based phase analysis not implemented yet'
            }
        
        self.results['phase_domain'] = phase_domain_results
        return phase_domain_results
    
    def analyze_frequency_domain(self, x: np.ndarray, y: np.ndarray, 
                               fs: float = 1.0, 
                               method: str = 'welch') -> Dict:
        """
        Perform frequency domain correlation analysis.
        
        Args:
            x: First time series
            y: Second time series
            fs: Sampling frequency
            method: Method for spectral analysis ('welch', 'periodogram', 'lombscargle')
        
        Returns:
            Dictionary with analysis results
        """
        # Ensure arrays are numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        
        frequency_domain_results = {}
        
        # Compute coherence
        frequencies, coherence = compute_coherence(x, y, fs=fs)
        frequency_domain_results['coherence'] = {
            'frequencies': frequencies,
            'coherence': coherence
        }
        
        # Compute spectral analysis based on method
        if method == 'welch' or method == 'all':
            freq_x, power_x = compute_welch(x, fs=fs)
            freq_y, power_y = compute_welch(y, fs=fs)
            
            # Find peaks
            peaks_x, peak_powers_x, _ = find_peaks(freq_x, power_x, prominence=0.1)
            peaks_y, peak_powers_y, _ = find_peaks(freq_y, power_y, prominence=0.1)
            
            frequency_domain_results['welch'] = {
                'x': {
                    'frequencies': freq_x,
                    'power': power_x,
                    'peak_frequencies': peaks_x,
                    'peak_powers': peak_powers_x
                },
                'y': {
                    'frequencies': freq_y,
                    'power': power_y,
                    'peak_frequencies': peaks_y,
                    'peak_powers': peak_powers_y
                }
            }
        
        if method == 'periodogram' or method == 'all':
            freq_x, power_x = compute_periodogram(x, fs=fs)
            freq_y, power_y = compute_periodogram(y, fs=fs)
            
            # Find peaks
            peaks_x, peak_powers_x, _ = find_peaks(freq_x, power_x, prominence=0.1)
            peaks_y, peak_powers_y, _ = find_peaks(freq_y, power_y, prominence=0.1)
            
            frequency_domain_results['periodogram'] = {
                'x': {
                    'frequencies': freq_x,
                    'power': power_x,
                    'peak_frequencies': peaks_x,
                    'peak_powers': peak_powers_x
                },
                'y': {
                    'frequencies': freq_y,
                    'power': power_y,
                    'peak_frequencies': peaks_y,
                    'peak_powers': peak_powers_y
                }
            }
        
        if method == 'lombscargle' or method == 'all':
            # For Lomb-Scargle, we need time points
            t = np.arange(len(x)) / fs
            
            # Compute Lomb-Scargle periodogram
            freq_x, power_x = compute_lombscargle(t, x)
            freq_y, power_y = compute_lombscargle(t, y)
            
            # Find peaks
            peaks_x, peak_powers_x, _ = find_peaks(freq_x, power_x, prominence=0.1)
            peaks_y, peak_powers_y, _ = find_peaks(freq_y, power_y, prominence=0.1)
            
            frequency_domain_results['lombscargle'] = {
                'x': {
                    'frequencies': freq_x,
                    'power': power_x,
                    'peak_frequencies': peaks_x,
                    'peak_powers': peak_powers_x
                },
                'y': {
                    'frequencies': freq_y,
                    'power': power_y,
                    'peak_frequencies': peaks_y,
                    'peak_powers': peak_powers_y
                }
            }
        
        self.results['frequency_domain'] = frequency_domain_results
        return frequency_domain_results
    
    def perform_statistical_tests(self, x: np.ndarray, y: np.ndarray, 
                                frequencies: Optional[np.ndarray] = None) -> Dict:
        """
        Perform statistical tests for cyclical correlation.
        
        Args:
            x: First time series
            y: Second time series
            frequencies: Frequencies to test (if None, uses default range)
        
        Returns:
            Dictionary with test results
        """
        # Ensure arrays are numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        
        statistical_results = {}
        
        # Test for standard correlation
        corr = np.corrcoef(x, y)[0, 1]
        n = len(x)
        
        from src.statistical_modeling.statistical_testing import test_correlation_significance
        corr_test = test_correlation_significance(
            corr, n, method='pearson', alpha=self.significance_level
        )
        
        statistical_results['standard_correlation'] = corr_test
        
        # Test for cyclical correlation if frequencies are provided
        if frequencies is not None:
            from src.statistical_modeling.statistical_testing import test_cyclical_correlation
            
            cyclical_test = test_cyclical_correlation(
                x, y, frequencies, alpha=self.significance_level
            )
            
            statistical_results['cyclical_correlation'] = cyclical_test
        
        # Test for phase synchronization
        try:
            # Import here to avoid circular imports
            from src.statistical_modeling.spectral_analysis import compute_hilbert_transform
            
            # Compute Hilbert transform to get phase
            _, _, x_phase = compute_hilbert_transform(x)
            _, _, y_phase = compute_hilbert_transform(y)
            
            from src.statistical_modeling.statistical_testing import test_phase_synchronization
            phase_test = test_phase_synchronization(
                x_phase, y_phase, n_surrogates=100, alpha=self.significance_level
            )
            
            statistical_results['phase_synchronization'] = phase_test
        except Exception as e:
            logger.warning(f"Could not perform phase synchronization test: {e}")
        
        # Test for Granger causality
        try:
            from src.statistical_modeling.statistical_testing import test_granger_causality
            
            # Test x causing y
            granger_xy = test_granger_causality(
                x, y, max_lag=min(10, n//10), alpha=self.significance_level
            )
            
            # Test y causing x
            granger_yx = test_granger_causality(
                y, x, max_lag=min(10, n//10), alpha=self.significance_level
            )
            
            statistical_results['granger_causality'] = {
                'x_causes_y': granger_xy,
                'y_causes_x': granger_yx
            }
        except Exception as e:
            logger.warning(f"Could not perform Granger causality test: {e}")
        
        self.results['statistical_tests'] = statistical_results
        return statistical_results
    
    def analyze_astrological_correlation(self, market_data: np.ndarray, 
                                       astro_data: Dict[str, np.ndarray], 
                                       frequencies: Optional[np.ndarray] = None, 
                                       methods: Optional[List[str]] = None) -> Dict:
        """
        Analyze correlation between market data and astrological factors.
        
        Args:
            market_data: Market time series
            astro_data: Dictionary of astrological time series
            frequencies: Frequencies to test
            methods: Analysis methods to use
        
        Returns:
            Dictionary with analysis results
        """
        # Set default methods if not provided
        if methods is None:
            methods = ['time_domain', 'frequency_domain', 'statistical_tests']
        
        # Ensure market_data is a numpy array
        market_data = np.asarray(market_data)
        
        # Initialize results dictionary
        astro_results = {}
        
        # Analyze each astrological factor
        for factor_name, factor_data in astro_data.items():
            factor_results = {}
            
            # Ensure factor_data is a numpy array
            factor_data = np.asarray(factor_data)
            
            # Perform requested analyses
            if 'time_domain' in methods:
                factor_results['time_domain'] = self.analyze_time_domain(
                    market_data, factor_data
                )
            
            if 'phase_domain' in methods:
                factor_results['phase_domain'] = self.analyze_phase_domain(
                    market_data, factor_data
                )
            
            if 'frequency_domain' in methods:
                factor_results['frequency_domain'] = self.analyze_frequency_domain(
                    market_data, factor_data
                )
            
            if 'statistical_tests' in methods:
                factor_results['statistical_tests'] = self.perform_statistical_tests(
                    market_data, factor_data, frequencies
                )
            
            astro_results[factor_name] = factor_results
        
        # Use the statistical testing module for a comprehensive test
        from src.statistical_modeling.statistical_testing import test_astrological_correlation
        
        astro_test = test_astrological_correlation(
            market_data, astro_data, frequencies, alpha=self.significance_level
        )
        
        astro_results['overall_test'] = astro_test
        
        self.results['astrological_correlation'] = astro_results
        return astro_results
    
    def plot_results(self, result_type: str = 'time_domain', 
                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot analysis results.
        
        Args:
            result_type: Type of results to plot
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if result_type not in self.results:
            raise ValueError(f"No results available for {result_type}")
        
        if result_type == 'time_domain':
            # Plot cross-correlation
            lags = self.results['time_domain']['cross_correlation']['lags']
            correlation = self.results['time_domain']['cross_correlation']['correlation']
            
            return plot_cross_correlation(
                lags, correlation, 
                title="Cross-Correlation Analysis", 
                figsize=figsize
            )
        
        elif result_type == 'phase_domain':
            # This is a placeholder - actual implementation would depend on the
            # specific phase analysis results available
            logger.warning("Phase domain plotting not fully implemented")
            return None
        
        elif result_type == 'frequency_domain':
            # Plot coherence
            frequencies = self.results['frequency_domain']['coherence']['frequencies']
            coherence = self.results['frequency_domain']['coherence']['coherence']
            
            return plot_coherence(
                frequencies, coherence,
                title="Spectral Coherence Analysis",
                figsize=figsize
            )
        
        elif result_type == 'statistical_tests':
            # This is a placeholder - actual implementation would depend on the
            # specific statistical test results available
            logger.warning("Statistical test plotting not fully implemented")
            return None
        
        elif result_type == 'astrological_correlation':
            # Plot astrological correlation results
            from src.statistical_modeling.statistical_testing import plot_correlation_test_results
            
            return plot_correlation_test_results(
                self.results['astrological_correlation']['overall_test'],
                title="Market-Astrological Correlations",
                figsize=figsize
            )
        
        else:
            logger.warning(f"Plotting not implemented for {result_type}")
            return None


# Convenience functions for direct use without instantiating the class

def analyze_cyclical_correlation(x: np.ndarray, y: np.ndarray, 
                               methods: Optional[List[str]] = None, 
                               **kwargs) -> Dict:
    """
    Analyze cyclical correlation between two time series.
    
    Args:
        x: First time series
        y: Second time series
        methods: List of analysis methods to use
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with analysis results
    """
    # Set default methods if not provided
    if methods is None:
        methods = ['time_domain', 'frequency_domain', 'statistical_tests']
    
    # Create analyzer instance
    analyzer = CyclicalCorrelationAnalyzer(
        significance_level=kwargs.get('significance_level', 0.05)
    )
    
    # Perform requested analyses
    results = {}
    
    if 'time_domain' in methods:
        results['time_domain'] = analyzer.analyze_time_domain(
            x, y,
            max_lag=kwargs.get('max_lag', None),
            window_size=kwargs.get('window_size', 30)
        )
    
    if 'phase_domain' in methods:
        results['phase_domain'] = analyzer.analyze_phase_domain(
            x, y,
            method=kwargs.get('phase_method', 'circular')
        )
    
    if 'frequency_domain' in methods:
        results['frequency_domain'] = analyzer.analyze_frequency_domain(
            x, y,
            fs=kwargs.get('fs', 1.0),
            method=kwargs.get('spectral_method', 'welch')
        )
    
    if 'statistical_tests' in methods:
        results['statistical_tests'] = analyzer.perform_statistical_tests(
            x, y,
            frequencies=kwargs.get('frequencies', None)
        )
    
    return results


def analyze_market_astro_correlation(market_data: np.ndarray, 
                                   astro_data: Dict[str, np.ndarray], 
                                   frequencies: Optional[np.ndarray] = None, 
                                   methods: Optional[List[str]] = None, 
                                   **kwargs) -> Dict:
    """
    Analyze correlation between market data and astrological factors.
    
    Args:
        market_data: Market time series
        astro_data: Dictionary of astrological time series
        frequencies: Frequencies to test
        methods: Analysis methods to use
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with analysis results
    """
    # Create analyzer instance
    analyzer = CyclicalCorrelationAnalyzer(
        significance_level=kwargs.get('significance_level', 0.05)
    )
    
    # Perform analysis
    results = analyzer.analyze_astrological_correlation(
        market_data, astro_data, frequencies, methods
    )
    
    return results


def detect_significant_cycles(time_series: np.ndarray, 
                            min_period: int = 2, 
                            max_period: Optional[int] = None, 
                            method: str = 'lombscargle', 
                            **kwargs) -> Dict:
    """
    Detect significant cycles in a time series.
    
    Args:
        time_series: Time series data
        min_period: Minimum period to consider
        max_period: Maximum period to consider
        method: Method for cycle detection
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with detected cycles
    """
    # Ensure time_series is a numpy array
    time_series = np.asarray(time_series)
    
    # Set default max_period if not provided
    if max_period is None:
        max_period = len(time_series) // 2
    
    # Create period array
    periods = np.arange(min_period, max_period + 1)
    frequencies = 1.0 / periods
    
    # Set sampling frequency
    fs = kwargs.get('fs', 1.0)
    
    # Detect cycles based on method
    if method == 'lombscargle':
        # For Lomb-Scargle, we need time points
        t = np.arange(len(time_series)) / fs
        
        from src.statistical_modeling.spectral_analysis import compute_lombscargle, find_peaks
        
        # Compute Lomb-Scargle periodogram
        freq, power = compute_lombscargle(t, time_series, frequencies=frequencies)
        
        # Find peaks
        peak_freqs, peak_powers, _ = find_peaks(
            freq, power, 
            prominence=kwargs.get('prominence', 0.1),
            height=kwargs.get('height', None)
        )
        
        # Convert to periods
        peak_periods = 1.0 / peak_freqs
        
        # Test significance
        from src.statistical_modeling.statistical_testing import test_periodogram_significance
        
        sig_freqs, sig_powers = test_periodogram_significance(
            freq, power, alpha=kwargs.get('significance_level', 0.05)
        )
        
        # Filter peaks by significance
        significant_peaks = np.isin(peak_freqs, sig_freqs)
        sig_peak_freqs = peak_freqs[significant_peaks]
        sig_peak_powers = peak_powers[significant_peaks]
        sig_peak_periods = 1.0 / sig_peak_freqs
        
        return {
            'method': 'lombscargle',
            'frequencies': freq,
            'power': power,
            'peak_frequencies': peak_freqs,
            'peak_powers': peak_powers,
            'peak_periods': peak_periods,
            'significant_frequencies': sig_peak_freqs,
            'significant_powers': sig_peak_powers,
            'significant_periods': sig_peak_periods
        }
    
    elif method == 'fft':
        from src.statistical_modeling.spectral_analysis import compute_periodogram, find_peaks
        
        # Compute periodogram
        freq, power = compute_periodogram(time_series, fs=fs)
        
        # Find peaks
        peak_freqs, peak_powers, _ = find_peaks(
            freq, power, 
            prominence=kwargs.get('prominence', 0.1),
            height=kwargs.get('height', None)
        )
        
        # Convert to periods
        peak_periods = 1.0 / peak_freqs
        
        # Test significance
        from src.statistical_modeling.statistical_testing import test_periodogram_significance
        
        sig_freqs, sig_powers = test_periodogram_significance(
            freq, power, alpha=kwargs.get('significance_level', 0.05)
        )
        
        # Filter peaks by significance
        significant_peaks = np.isin(peak_freqs, sig_freqs)
        sig_peak_freqs = peak_freqs[significant_peaks]
        sig_peak_powers = peak_powers[significant_peaks]
        sig_peak_periods = 1.0 / sig_peak_freqs
        
        return {
            'method': 'fft',
            'frequencies': freq,
            'power': power,
            'peak_frequencies': peak_freqs,
            'peak_powers': peak_powers,
            'peak_periods': peak_periods,
            'significant_frequencies': sig_peak_freqs,
            'significant_powers': sig_peak_powers,
            'significant_periods': sig_peak_periods
        }
    
    elif method == 'wavelet':
        # This is a placeholder for wavelet-based cycle detection
        logger.warning("Wavelet-based cycle detection not fully implemented")
        return {'method': 'wavelet', 'message': 'Not implemented yet'}
    
    else:
        raise ValueError(f"Unknown cycle detection method: {method}")


def find_common_cycles(time_series_dict: Dict[str, np.ndarray], 
                     min_period: int = 2, 
                     max_period: Optional[int] = None, 
                     significance_level: float = 0.05, 
                     **kwargs) -> Dict:
    """
    Find common cycles across multiple time series.
    
    Args:
        time_series_dict: Dictionary of time series
        min_period: Minimum period to consider
        max_period: Maximum period to consider
        significance_level: Significance level for detection
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with common cycles
    """
    # Detect cycles in each time series
    cycles_by_series = {}
    
    for name, series in time_series_dict.items():
        cycles = detect_significant_cycles(
            series, 
            min_period=min_period, 
            max_period=max_period, 
            method=kwargs.get('method', 'lombscargle'),
            significance_level=significance_level,
            **kwargs
        )
        
        cycles_by_series[name] = cycles
    
    # Find common significant periods
    all_sig_periods = []
    for name, cycles in cycles_by_series.items():
        if 'significant_periods' in cycles:
            all_sig_periods.append(set(np.round(cycles['significant_periods'], 1)))
    
    # Find intersection of significant periods
    if all_sig_periods:
        common_periods = set.intersection(*all_sig_periods)
    else:
        common_periods = set()
    
    return {
        'cycles_by_series': cycles_by_series,
        'common_periods': sorted(list(common_periods))
    }


# Example usage
if __name__ == "__main__":
    # Generate sample data with cyclical components
    np.random.seed(42)
    
    # Create time array
    days = 365 * 5  # 5 years
    t = np.arange(days)
    
    # Create market data with cyclical components
    market_data = (
        0.5 * np.sin(2 * np.pi * t / 30) +  # 30-day cycle
        1.0 * np.sin(2 * np.pi * t / 365) +  # Annual cycle
        0.3 * np.random.randn(days)  # Noise
    )
    
    # Create astrological factors
    astro_data = {
        'Mercury': 0.7 * np.sin(2 * np.pi * t / 88) + 0.2 * np.random.randn(days),  # Mercury cycle
        'Venus': 0.6 * np.sin(2 * np.pi * t / 225) + 0.2 * np.random.randn(days),  # Venus cycle
        'Mars': 0.5 * np.sin(2 * np.pi * t / 687) + 0.2 * np.random.randn(days),  # Mars cycle
        'Jupiter': 0.4 * np.sin(2 * np.pi * t / (365 * 11.86)) + 0.2 * np.random.randn(days),  # Jupiter cycle
        'Saturn': 0.3 * np.sin(2 * np.pi * t / (365 * 29.46)) + 0.2 * np.random.randn(days),  # Saturn cycle
        'Moon Phase': 0.8 * np.sin(2 * np.pi * t / 29.5) + 0.2 * np.random.randn(days),  # Lunar cycle
        'Solar Activity': 0.4 * np.sin(2 * np.pi * t / (365 * 11)) + 0.2 * np.random.randn(days),  # Solar cycle
        'Random Factor': 0.3 * np.random.randn(days)  # Pure noise
    }
    
    # Example 1: Analyze correlation between market data and Moon Phase
    print("\nExample 1: Analyzing correlation between market data and Moon Phase")
    moon_results = analyze_cyclical_correlation(market_data, astro_data['Moon Phase'])
    
    # Print key results
    optimal_lag = moon_results['time_domain']['cross_correlation']['optimal_lag']
    optimal_corr = moon_results['time_domain']['cross_correlation']['optimal_correlation']
    print(f"Optimal lag: {optimal_lag} days, Correlation: {optimal_corr:.4f}")
    
    # Example 2: Detect significant cycles in market data
    print("\nExample 2: Detecting significant cycles in market data")
    cycles = detect_significant_cycles(market_data, min_period=7, max_period=400)
    
    # Print detected cycles
    if 'significant_periods' in cycles:
        print("Significant cycles detected (periods in days):")
        for period in sorted(cycles['significant_periods']):
            print(f"  {period:.1f} days")
    
    # Example 3: Analyze correlation between market data and all astrological factors
    print("\nExample 3: Analyzing market-astrological correlations")
    
    # Define frequencies to test (corresponding to common astrological cycles)
    frequencies = 1 / np.array([7, 14, 29.5, 88, 225, 365, 687, 365*11, 365*11.86, 365*29.46])
    
    astro_results = analyze_market_astro_correlation(
        market_data, astro_data, frequencies=frequencies
    )
    
    # Print key results
    print("Significant astrological correlations:")
    for factor, results in astro_results.items():
        if factor != 'overall_test' and 'statistical_tests' in results:
            if results['statistical_tests']['standard_correlation']['is_significant']:
                corr = results['statistical_tests']['standard_correlation']['correlation']
                print(f"  {factor}: r = {corr:.4f}")
    
    # Example 4: Find common cycles across multiple time series
    print("\nExample 4: Finding common cycles across market data and astrological factors")
    
    # Create dictionary with market data and selected astrological factors
    time_series_dict = {
        'Market': market_data,
        'Moon Phase': astro_data['Moon Phase'],
        'Mercury': astro_data['Mercury'],
        'Venus': astro_data['Venus']
    }
    
    common_cycles = find_common_cycles(
        time_series_dict, min_period=7, max_period=400
    )
    
    # Print common cycles
    print("Common cycles across time series (periods in days):")
    for period in common_cycles['common_periods']:
        print(f"  {period} days")
    
    print("\nCyclical Correlation Analysis complete!")
    
    # Create plots (uncomment to display)
    # plt.show()
