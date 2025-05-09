# Tests for the feature engineering module

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.feature_engineering.astrological_features import AstrologicalFeatureGenerator
from src.feature_engineering.market_features import MarketFeatureGenerator
from src.feature_engineering.feature_selection import FeatureSelector
from src.feature_engineering.feature_importance import FeatureImportanceAnalyzer
from src.astro_engine.planetary_positions import PlanetaryCalculator


@pytest.fixture
def sample_date():
    """Provide a sample date for testing"""
    return datetime(2023, 1, 1, 12, 0, 0)


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing"""
    # Create date range
    dates = pd.date_range(start='2022-01-01', end='2022-01-31', freq='D')
    
    # Create sample data
    np.random.seed(42)  # For reproducibility
    data = {
        'open': np.random.normal(100, 5, len(dates)),
        'high': np.random.normal(105, 5, len(dates)),
        'low': np.random.normal(95, 5, len(dates)),
        'close': np.random.normal(102, 5, len(dates)),
        'volume': np.random.normal(1000000, 200000, len(dates))
    }
    
    # Ensure high > open, close, low and low < open, close, high
    for i in range(len(dates)):
        values = [data['open'][i], data['close'][i]]
        data['high'][i] = max(values) + abs(np.random.normal(2, 1))
        data['low'][i] = min(values) - abs(np.random.normal(2, 1))
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    return df


# Tests for AstrologicalFeatureGenerator
def test_astrological_feature_generator_initialization():
    """Test that the AstrologicalFeatureGenerator can be initialized"""
    generator = AstrologicalFeatureGenerator()
    assert generator is not None


def test_encode_cyclical_feature():
    """Test encoding of cyclical features"""
    generator = AstrologicalFeatureGenerator()
    values = np.array([0, 90, 180, 270, 360])
    encoded = generator.encode_cyclical_feature(values, 360)
    
    # Check shape
    assert encoded.shape == (5, 2)
    
    # Check values for 0 and 360 degrees (should be the same)
    np.testing.assert_almost_equal(encoded[0], encoded[4])
    
    # Check values for 90 degrees
    np.testing.assert_almost_equal(encoded[1][0], 0, decimal=5)  # sin(90°) = 1
    np.testing.assert_almost_equal(encoded[1][1], 0, decimal=5)  # cos(90°) = 0


def test_generate_planet_features(sample_date):
    """Test generation of planetary features"""
    generator = AstrologicalFeatureGenerator()
    features = generator.generate_planet_features(sample_date)
    
    # Check that we have features for all planets
    assert 'sun_longitude' in features
    assert 'moon_longitude' in features
    assert 'jupiter_longitude' in features
    
    # Check that we have sine and cosine encodings
    assert 'sun_longitude_sin' in features
    assert 'sun_longitude_cos' in features
    
    # Check that we have retrograde indicators
    assert 'mercury_is_retrograde' in features


# Tests for MarketFeatureGenerator
def test_market_feature_generator_initialization():
    """Test that the MarketFeatureGenerator can be initialized"""
    generator = MarketFeatureGenerator()
    assert generator is not None


def test_add_price_features(sample_market_data):
    """Test adding price-derived features"""
    generator = MarketFeatureGenerator()
    result = generator.add_price_features(sample_market_data)
    
    # Check that we have the expected features
    assert 'return_1d' in result.columns
    assert 'return_5d' in result.columns
    assert 'ma_10' in result.columns
    assert 'price_to_ma_10' in result.columns
    
    # Check that the first return is NaN (no previous data)
    assert pd.isna(result['return_1d'].iloc[0])


def test_add_technical_indicators(sample_market_data):
    """Test adding technical indicators"""
    generator = MarketFeatureGenerator()
    result = generator.add_technical_indicators(sample_market_data)
    
    # Check that we have the expected indicators
    assert 'rsi_14' in result.columns
    assert 'macd' in result.columns
    assert 'bb_upper_20' in result.columns
    assert 'stoch_k' in result.columns


# Tests for FeatureSelector
def test_feature_selector_initialization():
    """Test that the FeatureSelector can be initialized"""
    selector = FeatureSelector()
    assert selector is not None


def test_select_by_correlation():
    """Test feature selection by correlation"""
    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100)
    })
    
    # Create target with strong correlation to feature1
    y = X['feature1'] * 2 + np.random.normal(0, 0.1, 100)
    
    # Select features
    selector = FeatureSelector()
    selected = selector.select_by_correlation(X, y, k=2)
    
    # Check that we selected 2 features
    assert selected.shape[1] == 2
    
    # Check that feature1 is selected (it has the strongest correlation)
    assert 'feature1' in selected.columns


# Tests for FeatureImportanceAnalyzer
def test_feature_importance_analyzer_initialization():
    """Test that the FeatureImportanceAnalyzer can be initialized"""
    analyzer = FeatureImportanceAnalyzer()
    assert analyzer is not None


def test_get_model_feature_importance():
    """Test extracting feature importance from a model"""
    # Skip this test if sklearn is not available
    pytest.importorskip("sklearn")
    
    from sklearn.ensemble import RandomForestRegressor
    
    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100)
    })
    
    # Create target
    y = X['feature1'] * 2 + X['feature2'] + np.random.normal(0, 0.1, 100)
    
    # Train a model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Get feature importance
    analyzer = FeatureImportanceAnalyzer()
    importance = analyzer.get_model_feature_importance(model, X)
    
    # Check that we have importance for all features
    assert set(importance['feature']) == set(X.columns)
    
    # Check that importance values sum to approximately 1
    assert abs(importance['importance'].sum() - 1.0) < 1e-6