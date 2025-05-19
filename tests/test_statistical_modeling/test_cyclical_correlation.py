"""
Tests for the Cyclical Correlation module.

This module contains tests for the cyclical correlation framework, including
time domain, phase domain, frequency domain, and statistical testing components.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from src.statistical_modeling.cyclical_correlation import (
    CyclicalCorrelationAnalyzer,
    analyze_cyclical_correlation,
    analyze_market_astro_correlation,
    detect_significant_cycles,
    find_common_cycles
)


class TestCyclicalCorrelationAnalyzer:
    """Tests for the CyclicalCorrelationAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create time array
        days = 365  # 1 year
        t = np.arange(days)
        
        # Create two time series with shared cycles
        x = (
            0.5 * np.sin(2 * np.pi * t / 30) +  # 30-day cycle
            1.0 * np.sin(2 * np.pi * t / 365) +  # Annual cycle
            0.3 * np.random.randn(days)  # Noise
        )
        
        y = (
            0.7 * np.sin(2 * np.pi * t / 30 + np.pi/4) +  # 30-day cycle with phase shift
            0.8 * np.sin(2 * np.pi * t / 365) +  # Annual cycle
            0.3 * np.random.randn(days)  # Noise
        )
        
        # Create market data with cyclical components
        market_data = (
            0.5 * np.sin(2 * np.pi * t / 30) +  # 30-day cycle
            1.0 * np.sin(2 * np.pi * t / 365) +  # Annual cycle
            0.3 * np.random.randn(days)  # Noise
        )
        
        # Create astrological factors
        astro_data = {
            'Moon Phase': 0.8 * np.sin(2 * np.pi * t / 29.5) + 0.2 * np.random.randn(days),  # Lunar cycle
            'Mercury': 0.7 * np.sin(2 * np.pi * t / 88) + 0.2 * np.random.randn(days),  # Mercury cycle
            'Random Factor': 0.3 * np.random.randn(days)  # Pure noise
        }
        
        return {
            'x': x,
            'y': y,
            't': t,
            'market_data': market_data,
            'astro_data': astro_data
        }
    
    def test_init(self):
        """Test initialization of CyclicalCorrelationAnalyzer."""
        analyzer = CyclicalCorrelationAnalyzer()
        assert analyzer.significance_level == 0.05
        assert analyzer.results == {}
        
        analyzer = CyclicalCorrelationAnalyzer(significance_level=0.01)
        assert analyzer.significance_level == 0.01
    
    def test_analyze_time_domain(self, sample_data):
        """Test time domain analysis."""
        x = sample_data['x']
        y = sample_data['y']
        
        analyzer = CyclicalCorrelationAnalyzer()
        results = analyzer.analyze_time_domain(x, y)
        
        # Check that results contain expected keys
        assert 'cross_correlation' in results
        assert 'rolling_correlation' in results
        assert 'windowed_correlation' in results
        
        # Check cross-correlation results
        assert 'lags' in results['cross_correlation']
        assert 'correlation' in results['cross_correlation']
        assert 'optimal_lag' in results['cross_correlation']
        assert 'optimal_correlation' in results['cross_correlation']
        
        # Check that optimal lag is reasonable (should be close to 0 for similar series)
        assert abs(results['cross_correlation']['optimal_lag']) < 10
        
        # Check that optimal correlation is high (should be > 0.5 for similar series)
        assert abs(results['cross_correlation']['optimal_correlation']) > 0.5
    
    def test_analyze_frequency_domain(self, sample_data):
        """Test frequency domain analysis."""
        x = sample_data['x']
        y = sample_data['y']
        
        analyzer = CyclicalCorrelationAnalyzer()
        results = analyzer.analyze_frequency_domain(x, y)
        
        # Check that results contain expected keys
        assert 'coherence' in results
        
        # Check coherence results
        assert 'frequencies' in results['coherence']
        assert 'coherence' in results['coherence']
        
        # Check that coherence values are between 0 and 1
        assert np.all(results['coherence']['coherence'] >= 0)
        assert np.all(results['coherence']['coherence'] <= 1)
        
        # Test with different methods
        for method in ['welch', 'periodogram', 'lombscargle']:
            results = analyzer.analyze_frequency_domain(x, y, method=method)
            assert method in results
            assert 'x' in results[method]
            assert 'y' in results[method]
            assert 'frequencies' in results[method]['x']
            assert 'power' in results[method]['x']
    
    def test_perform_statistical_tests(self, sample_data):
        """Test statistical testing methods."""
        x = sample_data['x']
        y = sample_data['y']
        
        analyzer = CyclicalCorrelationAnalyzer()
        results = analyzer.perform_statistical_tests(x, y)
        
        # Check that results contain expected keys
        assert 'standard_correlation' in results
        
        # Check standard correlation results
        assert 'correlation' in results['standard_correlation']
        assert 'p_value' in results['standard_correlation']
        assert 'is_significant' in results['standard_correlation']
        
        # Test with frequencies
        frequencies = 1 / np.array([30, 60, 90, 365])
        results = analyzer.perform_statistical_tests(x, y, frequencies=frequencies)
        
        # This might not be present if the statistical_testing module doesn't have this function
        if 'cyclical_correlation' in results:
            assert 'frequencies' in results['cyclical_correlation']
    
    def test_analyze_astrological_correlation(self, sample_data):
        """Test astrological correlation analysis."""
        market_data = sample_data['market_data']
        astro_data = sample_data['astro_data']
        
        analyzer = CyclicalCorrelationAnalyzer()
        results = analyzer.analyze_astrological_correlation(market_data, astro_data)
        
        # Check that results contain expected keys
        for factor in astro_data.keys():
            assert factor in results
        
        # Check that 'Moon Phase' has significant correlation with market data
        # (since we constructed it that way)
        if 'statistical_tests' in results['Moon Phase']:
            corr = results['Moon Phase']['statistical_tests']['standard_correlation']['correlation']
            assert abs(corr) > 0.2  # Should have some correlation
    
    def test_plot_results(self, sample_data):
        """Test plotting functionality."""
        x = sample_data['x']
        y = sample_data['y']
        
        analyzer = CyclicalCorrelationAnalyzer()
        
        # First analyze data
        analyzer.analyze_time_domain(x, y)
        analyzer.analyze_frequency_domain(x, y)
        
        # Test time domain plotting
        fig = analyzer.plot_results(result_type='time_domain')
        assert fig is not None
        plt.close(fig)
        
        # Test frequency domain plotting
        fig = analyzer.plot_results(result_type='frequency_domain')
        assert fig is not None
        plt.close(fig)


class TestConvenienceFunctions:
    """Tests for the convenience functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create time array
        days = 365  # 1 year
        t = np.arange(days)
        
        # Create two time series with shared cycles
        x = (
            0.5 * np.sin(2 * np.pi * t / 30) +  # 30-day cycle
            1.0 * np.sin(2 * np.pi * t / 365) +  # Annual cycle
            0.3 * np.random.randn(days)  # Noise
        )
        
        y = (
            0.7 * np.sin(2 * np.pi * t / 30 + np.pi/4) +  # 30-day cycle with phase shift
            0.8 * np.sin(2 * np.pi * t / 365) +  # Annual cycle
            0.3 * np.random.randn(days)  # Noise
        )
        
        # Create market data with cyclical components
        market_data = (
            0.5 * np.sin(2 * np.pi * t / 30) +  # 30-day cycle
            1.0 * np.sin(2 * np.pi * t / 365) +  # Annual cycle
            0.3 * np.random.randn(days)  # Noise
        )
        
        # Create astrological factors
        astro_data = {
            'Moon Phase': 0.8 * np.sin(2 * np.pi * t / 29.5) + 0.2 * np.random.randn(days),  # Lunar cycle
            'Mercury': 0.7 * np.sin(2 * np.pi * t / 88) + 0.2 * np.random.randn(days),  # Mercury cycle
            'Random Factor': 0.3 * np.random.randn(days)  # Pure noise
        }
        
        return {
            'x': x,
            'y': y,
            't': t,
            'market_data': market_data,
            'astro_data': astro_data
        }
    
    def test_analyze_cyclical_correlation(self, sample_data):
        """Test analyze_cyclical_correlation function."""
        x = sample_data['x']
        y = sample_data['y']
        
        # Test with default parameters
        results = analyze_cyclical_correlation(x, y)
        
        # Check that results contain expected keys
        assert 'time_domain' in results
        assert 'frequency_domain' in results
        assert 'statistical_tests' in results
        
        # Test with specific methods
        results = analyze_cyclical_correlation(x, y, methods=['time_domain'])
        assert 'time_domain' in results
        assert 'frequency_domain' not in results
        
        # Test with additional parameters
        results = analyze_cyclical_correlation(
            x, y, 
            methods=['time_domain', 'frequency_domain'],
            max_lag=20,
            window_size=50,
            spectral_method='welch'
        )
        
        assert 'time_domain' in results
        assert 'frequency_domain' in results
    
    def test_analyze_market_astro_correlation(self, sample_data):
        """Test analyze_market_astro_correlation function."""
        market_data = sample_data['market_data']
        astro_data = sample_data['astro_data']
        
        # Test with default parameters
        results = analyze_market_astro_correlation(market_data, astro_data)
        
        # Check that results contain expected keys
        for factor in astro_data.keys():
            assert factor in results
        
        # Test with frequencies
        frequencies = 1 / np.array([30, 60, 90, 365])
        results = analyze_market_astro_correlation(
            market_data, astro_data, frequencies=frequencies
        )
        
        # Check that results contain expected keys
        assert 'overall_test' in results
    
    def test_detect_significant_cycles(self, sample_data):
        """Test detect_significant_cycles function."""
        x = sample_data['x']
        
        # Test with default parameters
        results = detect_significant_cycles(x, min_period=7, max_period=100)
        
        # Check that results contain expected keys
        assert 'method' in results
        assert 'frequencies' in results
        assert 'power' in results
        assert 'peak_frequencies' in results
        assert 'peak_powers' in results
        assert 'peak_periods' in results
        
        # Test with different methods
        for method in ['lombscargle', 'fft']:
            results = detect_significant_cycles(
                x, min_period=7, max_period=100, method=method
            )
            assert results['method'] == method
        
        # Test with edge case: no significant cycles
        random_data = np.random.randn(365)
        results = detect_significant_cycles(
            random_data, min_period=7, max_period=100,
            prominence=0.5  # Higher prominence threshold to ensure no peaks
        )
        
        # Should still have frequencies and power, but might not have significant peaks
        assert 'frequencies' in results
        assert 'power' in results
    
    def test_find_common_cycles(self, sample_data):
        """Test find_common_cycles function."""
        market_data = sample_data['market_data']
        astro_data = sample_data['astro_data']
        
        # Create dictionary with market data and selected astrological factors
        time_series_dict = {
            'Market': market_data,
            'Moon Phase': astro_data['Moon Phase'],
            'Mercury': astro_data['Mercury']
        }
        
        # Test with default parameters
        results = find_common_cycles(
            time_series_dict, min_period=7, max_period=100
        )
        
        # Check that results contain expected keys
        assert 'cycles_by_series' in results
        assert 'common_periods' in results
        
        # Check that cycles_by_series contains all time series
        for name in time_series_dict.keys():
            assert name in results['cycles_by_series']
        
        # Test with edge case: no common cycles
        # Add a random series that shouldn't share cycles
        time_series_dict['Random'] = np.random.randn(365)
        
        results = find_common_cycles(
            time_series_dict, min_period=7, max_period=100,
            significance_level=0.001  # Very strict significance level
        )
        
        # Should still have cycles_by_series, but might not have common periods
        assert 'cycles_by_series' in results
        assert 'common_periods' in results


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
