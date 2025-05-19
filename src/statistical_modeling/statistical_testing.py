"""
Statistical Testing Module for the Cosmic Market Oracle.

This module implements hypothesis tests for detecting significant cyclical
relationships between time series, with a focus on astrological and financial data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
from scipy import stats, signal
import logging
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
import statsmodels.api as sm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_stationarity(x: np.ndarray, test_type: str = 'adf',
                    regression: str = 'c') -> Dict[str, Union[float, bool]]:
    """
    Test for stationarity of a time series.
    
    Args:
        x: Time series data
        test_type: Type of test ('adf' for Augmented Dickey-Fuller or 'kpss' for KPSS)
        regression: Type of regression to include ('c' for constant, 'ct' for constant and trend)
        
    Returns:
        Dictionary with test results
    """
    if test_type == 'adf':
        # Augmented Dickey-Fuller test (null hypothesis: unit root exists)
        result = adfuller(x, regression=regression)
        
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05  # Reject null hypothesis if p < 0.05
        }
    
    elif test_type == 'kpss':
        # KPSS test (null hypothesis: series is stationary)
        regression_type = 'c' if regression == 'c' else 'ct'
        result = kpss(x, regression=regression_type)
        
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[3],
            'is_stationary': result[1] > 0.05  # Fail to reject null hypothesis if p > 0.05
        }
    
    else:
        raise ValueError(f"Unknown test type: {test_type}")


def test_correlation_significance(r: float, n: int, method: str = 'pearson',
                                alpha: float = 0.05) -> Dict[str, Union[float, bool]]:
    """
    Test the significance of a correlation coefficient.
    
    Args:
        r: Correlation coefficient
        n: Sample size
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    if method == 'pearson':
        # For Pearson correlation, use t-test
        t = r * np.sqrt((n - 2) / (1 - r**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t), df=n-2))
        
        # Calculate confidence interval
        r_z = np.arctanh(r)  # Fisher z-transformation
        se = 1 / np.sqrt(n - 3)
        z = stats.norm.ppf(1 - alpha/2)
        lo_z, hi_z = r_z - z*se, r_z + z*se
        lo, hi = np.tanh(lo_z), np.tanh(hi_z)
        
    elif method == 'spearman':
        # For Spearman correlation, use t-test approximation
        t = r * np.sqrt((n - 2) / (1 - r**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t), df=n-2))
        
        # Calculate confidence interval (approximate)
        r_z = np.arctanh(r)
        se = 1.06 / np.sqrt(n - 3)  # Adjusted SE for Spearman
        z = stats.norm.ppf(1 - alpha/2)
        lo_z, hi_z = r_z - z*se, r_z + z*se
        lo, hi = np.tanh(lo_z), np.tanh(hi_z)
        
    elif method == 'kendall':
        # For Kendall's tau, use normal approximation
        se = np.sqrt((2 * (2*n + 5)) / (9 * n * (n - 1)))
        z = r / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Calculate confidence interval
        z_crit = stats.norm.ppf(1 - alpha/2)
        lo, hi = r - z_crit*se, r + z_crit*se
        
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return {
        'correlation': r,
        'p_value': p_value,
        'confidence_interval': (lo, hi),
        'is_significant': p_value < alpha
    }


def test_cross_correlation_significance(lags: np.ndarray, ccf: np.ndarray, n: int,
                                      alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Test the significance of cross-correlation values.
    
    Args:
        lags: Lag values
        ccf: Cross-correlation values
        n: Sample size
        alpha: Significance level
        
    Returns:
        Tuple of (significant_lags, significant_ccf, confidence_bounds)
    """
    # Calculate confidence bounds
    se = 1.0 / np.sqrt(n)
    z = stats.norm.ppf(1 - alpha/2)
    conf_bound = z * se
    
    # Find significant correlations
    significant = np.abs(ccf) > conf_bound
    significant_lags = lags[significant]
    significant_ccf = ccf[significant]
    
    return significant_lags, significant_ccf, conf_bound


def test_periodogram_significance(frequencies: np.ndarray, power: np.ndarray,
                               alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test the significance of periodogram peaks.
    
    Args:
        frequencies: Frequency values
        power: Power spectrum values
        alpha: Significance level
        
    Returns:
        Tuple of (significant_frequencies, significant_power)
    """
    # For white noise, periodogram values follow an exponential distribution
    # Calculate threshold based on exponential distribution
    threshold = -np.log(alpha) * np.mean(power)
    
    # Find significant peaks
    significant = power > threshold
    significant_frequencies = frequencies[significant]
    significant_power = power[significant]
    
    return significant_frequencies, significant_power


def test_coherence_significance(coherence: np.ndarray, n_segments: int,
                             alpha: float = 0.05) -> float:
    """
    Calculate the significance threshold for coherence.
    
    Args:
        coherence: Coherence values
        n_segments: Number of segments used in coherence calculation
        alpha: Significance level
        
    Returns:
        Significance threshold
    """
    # For magnitude squared coherence, the threshold is:
    # 1 - alpha^(1/(n_segments-1))
    threshold = 1 - alpha**(1/(n_segments-1))
    
    return threshold


def test_granger_causality(x: np.ndarray, y: np.ndarray, max_lag: int,
                         alpha: float = 0.05) -> Dict[str, Union[float, bool, int]]:
    """
    Test for Granger causality between two time series.
    
    Args:
        x: First time series (potential cause)
        y: Second time series (potential effect)
        max_lag: Maximum lag to test
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Create pandas DataFrame
    data = pd.DataFrame({'x': x, 'y': y})
    
    # Run Granger causality test
    result = sm.tsa.stattools.grangercausalitytests(data[['y', 'x']], maxlag=max_lag, verbose=False)
    
    # Extract results for the optimal lag
    best_lag = None
    best_p_value = 1.0
    
    for lag in range(1, max_lag + 1):
        p_value = result[lag][0]['ssr_ftest'][1]
        if p_value < best_p_value:
            best_p_value = p_value
            best_lag = lag
    
    # Get F-statistic for the best lag
    f_stat = result[best_lag][0]['ssr_ftest'][0]
    
    return {
        'best_lag': best_lag,
        'f_statistic': f_stat,
        'p_value': best_p_value,
        'is_causal': best_p_value < alpha
    }


def test_phase_synchronization(phase1: np.ndarray, phase2: np.ndarray,
                             n_surrogates: int = 1000,
                             alpha: float = 0.05) -> Dict[str, Union[float, bool]]:
    """
    Test for phase synchronization between two phase time series using surrogate data.
    
    Args:
        phase1: First phase time series (in radians)
        phase2: Second phase time series (in radians)
        n_surrogates: Number of surrogate datasets to generate
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Calculate phase synchronization index
    phase_diff = (phase1 - phase2) % (2 * np.pi)
    psi_observed = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    # Generate surrogate data by shuffling one of the phase series
    psi_surrogates = np.zeros(n_surrogates)
    
    for i in range(n_surrogates):
        # Create surrogate by shuffling phase2
        phase2_surrogate = np.random.permutation(phase2)
        phase_diff_surrogate = (phase1 - phase2_surrogate) % (2 * np.pi)
        psi_surrogates[i] = np.abs(np.mean(np.exp(1j * phase_diff_surrogate)))
    
    # Calculate p-value
    p_value = np.mean(psi_surrogates >= psi_observed)
    
    # Calculate threshold
    threshold = np.percentile(psi_surrogates, 100 * (1 - alpha))
    
    return {
        'phase_sync_index': psi_observed,
        'p_value': p_value,
        'threshold': threshold,
        'is_significant': psi_observed > threshold
    }


def test_circular_correlation(alpha: np.ndarray, beta: np.ndarray,
                           n_surrogates: int = 1000,
                           alpha_level: float = 0.05) -> Dict[str, Union[float, bool]]:
    """
    Test for circular correlation between two circular variables using surrogate data.
    
    Args:
        alpha: First circular variable (in radians)
        beta: Second circular variable (in radians)
        n_surrogates: Number of surrogate datasets to generate
        alpha_level: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Calculate circular correlation
    sin_alpha = np.sin(alpha - np.mean(alpha))
    sin_beta = np.sin(beta - np.mean(beta))
    
    rho_observed = np.sum(sin_alpha * sin_beta) / np.sqrt(np.sum(sin_alpha**2) * np.sum(sin_beta**2))
    
    # Generate surrogate data by shuffling one of the variables
    rho_surrogates = np.zeros(n_surrogates)
    
    for i in range(n_surrogates):
        # Create surrogate by shuffling beta
        beta_surrogate = np.random.permutation(beta)
        sin_beta_surrogate = np.sin(beta_surrogate - np.mean(beta_surrogate))
        
        rho_surrogates[i] = np.sum(sin_alpha * sin_beta_surrogate) / np.sqrt(np.sum(sin_alpha**2) * np.sum(sin_beta_surrogate**2))
    
    # Calculate p-value (two-tailed test)
    p_value = np.mean(np.abs(rho_surrogates) >= np.abs(rho_observed))
    
    # Calculate threshold
    threshold_pos = np.percentile(rho_surrogates, 100 * (1 - alpha_level/2))
    threshold_neg = np.percentile(rho_surrogates, 100 * (alpha_level/2))
    
    return {
        'circular_correlation': rho_observed,
        'p_value': p_value,
        'threshold_positive': threshold_pos,
        'threshold_negative': threshold_neg,
        'is_significant': (rho_observed > threshold_pos) or (rho_observed < threshold_neg)
    }


def test_multiple_frequencies(time_series: np.ndarray, frequencies: np.ndarray,
                           alpha: float = 0.05,
                           correction_method: str = 'fdr_bh') -> Dict[str, List]:
    """
    Test for the presence of multiple frequencies in a time series with multiple testing correction.
    
    Args:
        time_series: Time series data
        frequencies: Frequencies to test (in cycles per sample)
        alpha: Significance level
        correction_method: Multiple testing correction method
        
    Returns:
        Dictionary with test results
    """
    n = len(time_series)
    p_values = []
    amplitudes = []
    phases = []
    
    # Test each frequency
    for freq in frequencies:
        # Create sine and cosine components
        t = np.arange(n)
        sin_comp = np.sin(2 * np.pi * freq * t)
        cos_comp = np.cos(2 * np.pi * freq * t)
        
        # Fit linear regression
        X = np.column_stack((sin_comp, cos_comp))
        model = sm.OLS(time_series, X)
        results = model.fit()
        
        # Extract p-value (joint test of both coefficients)
        p_value = results.f_pvalue
        p_values.append(p_value)
        
        # Calculate amplitude and phase
        a, b = results.params
        amplitude = np.sqrt(a**2 + b**2)
        phase = np.arctan2(b, a)
        
        amplitudes.append(amplitude)
        phases.append(phase)
    
    # Apply multiple testing correction
    reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method=correction_method)
    
    # Collect significant frequencies
    significant_frequencies = []
    significant_amplitudes = []
    significant_phases = []
    significant_p_values = []
    
    for i, (freq, is_significant) in enumerate(zip(frequencies, reject)):
        if is_significant:
            significant_frequencies.append(freq)
            significant_amplitudes.append(amplitudes[i])
            significant_phases.append(phases[i])
            significant_p_values.append(p_adjusted[i])
    
    return {
        'frequencies': significant_frequencies,
        'amplitudes': significant_amplitudes,
        'phases': significant_phases,
        'p_values': significant_p_values
    }


def test_cyclical_correlation(x: np.ndarray, y: np.ndarray, 
                           frequencies: np.ndarray,
                           alpha: float = 0.05,
                           correction_method: str = 'fdr_bh') -> Dict[str, List]:
    """
    Test for cyclical correlation between two time series at multiple frequencies.
    
    Args:
        x: First time series
        y: Second time series
        frequencies: Frequencies to test (in cycles per sample)
        alpha: Significance level
        correction_method: Multiple testing correction method
        
    Returns:
        Dictionary with test results
    """
    n = len(x)
    p_values = []
    correlations = []
    phases = []
    
    # Test each frequency
    for freq in frequencies:
        # Create complex exponential
        t = np.arange(n)
        exp_comp = np.exp(2j * np.pi * freq * t)
        
        # Calculate complex correlation
        x_complex = np.sum(x * exp_comp) / n
        y_complex = np.sum(y * exp_comp) / n
        
        # Calculate amplitude and phase for each series
        x_amp = np.abs(x_complex)
        y_amp = np.abs(y_complex)
        x_phase = np.angle(x_complex)
        y_phase = np.angle(y_complex)
        
        # Calculate correlation and phase difference
        correlation = x_amp * y_amp
        phase_diff = (x_phase - y_phase) % (2 * np.pi)
        
        # Calculate p-value using F-test approximation
        # (simplified approach, assumes white noise)
        f_stat = correlation**2 * (n-2) / (1 - correlation**2)
        p_value = 1 - stats.f.cdf(f_stat, 1, n-2)
        
        p_values.append(p_value)
        correlations.append(correlation)
        phases.append(phase_diff)
    
    # Apply multiple testing correction
    reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method=correction_method)
    
    # Collect significant frequencies
    significant_frequencies = []
    significant_correlations = []
    significant_phases = []
    significant_p_values = []
    
    for i, (freq, is_significant) in enumerate(zip(frequencies, reject)):
        if is_significant:
            significant_frequencies.append(freq)
            significant_correlations.append(correlations[i])
            significant_phases.append(phases[i])
            significant_p_values.append(p_adjusted[i])
    
    return {
        'frequencies': significant_frequencies,
        'correlations': significant_correlations,
        'phase_differences': significant_phases,
        'p_values': significant_p_values
    }


def test_seasonal_patterns(time_series: np.ndarray, periods: List[int],
                        alpha: float = 0.05) -> Dict[str, Dict[int, Dict]]:
    """
    Test for seasonal patterns in a time series.
    
    Args:
        time_series: Time series data
        periods: List of periods to test
        alpha: Significance level
        
    Returns:
        Dictionary with test results for each period
    """
    results = {}
    
    for period in periods:
        # Create seasonal dummies
        n = len(time_series)
        dummies = np.zeros((n, period))
        for i in range(period):
            dummies[i::period, i] = 1
        
        # Remove one dummy to avoid perfect multicollinearity
        dummies = dummies[:, 1:]
        
        # Fit regression model
        model = sm.OLS(time_series, dummies)
        fit_results = model.fit()
        
        # Extract results
        f_stat = fit_results.fvalue
        p_value = fit_results.f_pvalue
        r_squared = fit_results.rsquared
        
        results[period] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'r_squared': r_squared,
            'is_significant': p_value < alpha
        }
    
    return {'seasonal_patterns': results}


def test_astrological_correlation(market_data: np.ndarray, 
                               astro_data: Dict[str, np.ndarray],
                               frequencies: np.ndarray = None,
                               alpha: float = 0.05,
                               correction_method: str = 'fdr_bh') -> Dict[str, Dict]:
    """
    Test for correlation between market data and multiple astrological factors.
    
    Args:
        market_data: Market time series data
        astro_data: Dictionary of astrological factor time series
        frequencies: Frequencies to test (if None, uses standard correlation)
        alpha: Significance level
        correction_method: Multiple testing correction method
        
    Returns:
        Dictionary with test results for each astrological factor
    """
    results = {}
    
    # If frequencies are provided, use cyclical correlation test
    if frequencies is not None:
        for factor_name, factor_data in astro_data.items():
            factor_results = test_cyclical_correlation(
                market_data, factor_data, frequencies, alpha, correction_method
            )
            results[factor_name] = factor_results
    
    # Otherwise, use standard correlation test
    else:
        factor_names = list(astro_data.keys())
        p_values = []
        correlations = []
        
        for factor_name, factor_data in astro_data.items():
            # Calculate correlation
            correlation = np.corrcoef(market_data, factor_data)[0, 1]
            
            # Calculate p-value
            n = len(market_data)
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
            
            p_values.append(p_value)
            correlations.append(correlation)
        
        # Apply multiple testing correction
        reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method=correction_method)
        
        # Collect results
        for i, factor_name in enumerate(factor_names):
            results[factor_name] = {
                'correlation': correlations[i],
                'p_value': p_adjusted[i],
                'is_significant': reject[i]
            }
    
    return {'astrological_correlations': results}


def plot_correlation_test_results(results: Dict[str, Dict],
                               title: str = "Correlation Test Results",
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot correlation test results.
    
    Args:
        results: Dictionary with correlation test results
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract factor names and correlations
    factor_names = list(results['astrological_correlations'].keys())
    correlations = [results['astrological_correlations'][factor]['correlation'] for factor in factor_names]
    is_significant = [results['astrological_correlations'][factor]['is_significant'] for factor in factor_names]
    
    # Sort by absolute correlation
    sorted_indices = np.argsort(np.abs(correlations))[::-1]
    factor_names = [factor_names[i] for i in sorted_indices]
    correlations = [correlations[i] for i in sorted_indices]
    is_significant = [is_significant[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colors based on significance
    colors = ['green' if sig else 'gray' for sig in is_significant]
    
    # Plot horizontal bar chart
    bars = ax.barh(factor_names, correlations, color=colors)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title(title)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Significant'),
        Patch(facecolor='gray', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def plot_cyclical_correlation_results(results: Dict[str, Dict],
                                   factor_name: str,
                                   title: str = None,
                                   figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot cyclical correlation test results for a specific factor.
    
    Args:
        results: Dictionary with cyclical correlation test results
        factor_name: Name of the factor to plot
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract results for the specified factor
    factor_results = results['astrological_correlations'][factor_name]
    
    # Extract data
    frequencies = factor_results['frequencies']
    correlations = factor_results['correlations']
    phase_diffs = factor_results['phase_differences']
    
    # Convert frequencies to periods
    periods = [1/f if f > 0 else float('inf') for f in frequencies]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot correlations
    ax1.stem(periods, correlations, markerfmt='ro', basefmt=' ')
    
    # Add labels
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Correlation')
    ax1.set_title(f"Significant Cyclical Correlations for {factor_name}" if title is None else title)
    ax1.grid(True)
    
    # Set x-axis to log scale
    ax1.set_xscale('log')
    
    # Plot phase differences
    ax2.stem(periods, phase_diffs, markerfmt='bo', basefmt=' ')
    
    # Add labels
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Phase Difference (radians)')
    ax2.set_title(f"Phase Differences for {factor_name}")
    ax2.grid(True)
    
    # Set x-axis to log scale
    ax2.set_xscale('log')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


# Example usage
if __name__ == "__main__":
    # Generate sample data
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
    
    # Test for standard correlation
    corr_results = test_astrological_correlation(market_data, astro_data)
    
    # Test for cyclical correlation
    frequencies = 1 / np.array([7, 14, 29.5, 88, 225, 365, 687, 365*11, 365*11.86, 365*29.46])
    cyclical_results = test_astrological_correlation(market_data, astro_data, frequencies)
    
    # Plot results
    plot_correlation_test_results(corr_results, title="Market-Astrological Correlations")
    
    # Plot cyclical correlation for Moon Phase
    plot_cyclical_correlation_results(cyclical_results, 'Moon Phase')
    
    # Test for seasonal patterns
    seasonal_results = test_seasonal_patterns(market_data, [7, 30, 90, 365])
    print("Seasonal Pattern Results:")
    for period, result in seasonal_results['seasonal_patterns'].items():
        print(f"Period {period}: p-value = {result['p_value']:.4f}, significant = {result['is_significant']}")
    
    plt.show()
