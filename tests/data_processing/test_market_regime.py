"""
Tests for the Market Regime Labeling System.

This module tests the functionality of the market regime classification and validation.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import tempfile
import json

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data_processing.market_regime import MarketRegimeClassifier, RegimeValidator


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    # Create date range
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')
    
    # Create price series with different regimes
    n_days = len(dates)
    
    # Base series with trend
    t = np.arange(n_days) / 252  # Time in years
    
    # Bull market (first third)
    bull_end = n_days // 3
    bull_prices = 100 * np.exp(0.2 * t[:bull_end])  # 20% annual growth
    
    # Bear market (second third)
    bear_end = 2 * n_days // 3
    bear_prices = bull_prices[-1] * np.exp(-0.3 * t[bull_end:bear_end])  # 30% annual decline
    
    # Sideways market (last third)
    sideways_prices = bear_prices[-1] * np.ones(n_days - bear_end)  # Flat
    
    # Combine price series
    prices = np.concatenate([bull_prices, bear_prices, sideways_prices])
    
    # Add some noise
    noise = np.random.normal(0, 0.01, n_days)  # 1% daily noise
    prices = prices * (1 + noise)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.01, n_days)),
        'low': prices * (1 - np.random.uniform(0, 0.01, n_days)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_days)
    }, index=dates)
    
    return data


@pytest.fixture
def hmm_classifier():
    """Create a HMM classifier for testing."""
    return MarketRegimeClassifier(method="hmm", n_regimes=3, data_dir=tempfile.mkdtemp())


@pytest.fixture
def kmeans_classifier():
    """Create a K-means classifier for testing."""
    return MarketRegimeClassifier(method="kmeans", n_regimes=3, data_dir=tempfile.mkdtemp())


@pytest.fixture
def rule_classifier():
    """Create a rule-based classifier for testing."""
    return MarketRegimeClassifier(method="rule_based", data_dir=tempfile.mkdtemp())


@pytest.fixture
def regime_validator():
    """Create a regime validator for testing."""
    return RegimeValidator(data_dir=tempfile.mkdtemp())


def test_hmm_classifier_fit_predict(hmm_classifier, sample_market_data):
    """Test HMM classifier fitting and prediction."""
    # Fit the classifier
    hmm_classifier.fit(sample_market_data)
    
    # Predict regimes
    regimes = hmm_classifier.predict(sample_market_data)
    
    # Check that the regimes are assigned
    assert 'regime' in regimes.columns
    assert 'regime_name' in regimes.columns
    
    # Check that all regimes are assigned
    assert not regimes['regime'].isna().any()
    assert not regimes['regime_name'].isna().any()
    
    # Check that the number of unique regimes is less than or equal to n_regimes
    assert len(regimes['regime'].unique()) <= hmm_classifier.n_regimes


def test_kmeans_classifier_fit_predict(kmeans_classifier, sample_market_data):
    """Test K-means classifier fitting and prediction."""
    # Fit the classifier
    kmeans_classifier.fit(sample_market_data)
    
    # Predict regimes
    regimes = kmeans_classifier.predict(sample_market_data)
    
    # Check that the regimes are assigned
    assert 'regime' in regimes.columns
    assert 'regime_name' in regimes.columns
    
    # Check that all regimes are assigned
    assert not regimes['regime'].isna().any()
    assert not regimes['regime_name'].isna().any()
    
    # Check that the number of unique regimes is less than or equal to n_regimes
    assert len(regimes['regime'].unique()) <= kmeans_classifier.n_regimes


def test_rule_classifier_predict(rule_classifier, sample_market_data):
    """Test rule-based classifier prediction."""
    # Predict regimes (no fitting needed)
    regimes = rule_classifier.predict(sample_market_data)
    
    # Check that the regimes are assigned
    assert 'regime' in regimes.columns
    assert 'regime_name' in regimes.columns
    
    # Check that all regimes are assigned
    assert not regimes['regime'].isna().any()
    assert not regimes['regime_name'].isna().any()
    
    # Check that the regime names are as expected
    expected_names = {
        "Bull Market", "Bear Market", "High Volatility", 
        "Low Volatility", "Sideways Market"
    }
    assert set(regimes['regime_name'].unique()).issubset(expected_names)


def test_model_save_load(hmm_classifier, sample_market_data, tmpdir):
    """Test saving and loading a model."""
    # Fit the classifier
    hmm_classifier.fit(sample_market_data)
    
    # Predict regimes
    original_regimes = hmm_classifier.predict(sample_market_data)
    
    # Save the model
    model_file = os.path.join(tmpdir, "hmm_model.json")
    hmm_classifier.save_model(model_file)
    
    # Create a new classifier
    new_classifier = MarketRegimeClassifier(method="hmm", n_regimes=3, data_dir=tmpdir)
    
    # Load the model
    new_classifier.load_model(model_file)
    
    # Predict regimes with the loaded model
    loaded_regimes = new_classifier.predict(sample_market_data)
    
    # Check that the regimes are the same
    assert (original_regimes['regime'] == loaded_regimes['regime']).all()


def test_regime_validator(regime_validator, rule_classifier, sample_market_data):
    """Test the regime validator."""
    # Predict regimes
    regimes = rule_classifier.predict(sample_market_data)
    
    # Create expert validation
    start_date = sample_market_data.index[0]
    mid_date = sample_market_data.index[len(sample_market_data) // 2]
    end_date = sample_market_data.index[-1]
    
    expert_labels = {
        (start_date, mid_date): "Bull Market",
        (mid_date, end_date): "Bear Market"
    }
    
    # Add expert validation
    regime_validator.add_expert_validation(
        expert_name="test_expert",
        market_data=sample_market_data,
        regime_labels=expert_labels
    )
    
    # Check that the validation was saved
    assert "test_expert" in regime_validator.expert_validations
    
    # Load expert validations
    regime_validator.load_expert_validations()
    
    # Compare with model
    comparison = regime_validator.compare_with_model(
        classifier=rule_classifier,
        market_data=sample_market_data
    )
    
    # Check that the comparison has the expected keys
    assert "expert_comparisons" in comparison
    assert "average_agreement" in comparison
    assert "test_expert" in comparison["expert_comparisons"]
    
    # Check that the agreement metrics are present
    expert_comparison = comparison["expert_comparisons"]["test_expert"]
    assert "agreement_count" in expert_comparison
    assert "total_dates" in expert_comparison
    assert "agreement_pct" in expert_comparison
    assert "confusion_matrix" in expert_comparison


def test_plot_regimes(rule_classifier, sample_market_data, tmpdir):
    """Test plotting regimes."""
    # Predict regimes
    regimes = rule_classifier.predict(sample_market_data)
    
    # Plot regimes
    plot_file = os.path.join(tmpdir, "regimes.png")
    rule_classifier.plot_regimes(regimes, save_path=plot_file)
    
    # Check that the plot was saved
    assert os.path.exists(plot_file)


def test_find_contiguous_segments(rule_classifier):
    """Test finding contiguous segments in a DatetimeIndex."""
    # Create a DatetimeIndex with gaps
    dates = [
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
        datetime(2020, 1, 3),
        datetime(2020, 1, 6),  # Gap
        datetime(2020, 1, 7),
        datetime(2020, 1, 8)
    ]
    index = pd.DatetimeIndex(dates)
    
    # Find contiguous segments
    segments = rule_classifier._find_contiguous_segments(index)
    
    # Check that the segments are correct
    assert len(segments) == 2
    assert segments[0] == (datetime(2020, 1, 1), datetime(2020, 1, 3))
    assert segments[1] == (datetime(2020, 1, 6), datetime(2020, 1, 8))


def test_edge_case_empty_data(rule_classifier):
    """Test edge case with empty data."""
    # Create empty DataFrame
    empty_data = pd.DataFrame(columns=['close'])
    
    # Check that prediction raises an error
    with pytest.raises(ValueError):
        rule_classifier.predict(empty_data)


def test_edge_case_missing_close_column(rule_classifier):
    """Test edge case with missing 'close' column."""
    # Create DataFrame without 'close' column
    data = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [102, 103, 104],
        'low': [99, 100, 101]
    })
    
    # Check that prediction raises an error
    with pytest.raises(ValueError):
        rule_classifier.predict(data)


def test_failure_case_invalid_method():
    """Test failure case with invalid method."""
    # Check that initialization raises an error
    with pytest.raises(ValueError):
        MarketRegimeClassifier(method="invalid_method")
        
    # Create classifier with valid method
    classifier = MarketRegimeClassifier(method="hmm")
    
    # Change method to invalid
    classifier.method = "invalid_method"
    
    # Check that prediction raises an error
    with pytest.raises(ValueError):
        classifier.predict(pd.DataFrame({'close': [100, 101, 102]}))
