# Cosmic Market Oracle - Tests for Feature Generator Module

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.feature_engineering.feature_generator import (
    FeatureGenerator,
    generate_features,
    create_interaction_features,
    normalize_features,
    select_features
)


@pytest.fixture
def sample_integrated_data():
    """Create sample integrated market and astrological data for testing."""
    # Create a DataFrame with 30 days of mock integrated data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Market data features
    market_data = {
        'Date': dates,
        'Open': np.linspace(100, 110, len(dates)) + np.random.normal(0, 1, len(dates)),
        'High': np.linspace(102, 112, len(dates)) + np.random.normal(0, 1, len(dates)),
        'Low': np.linspace(98, 108, len(dates)) + np.random.normal(0, 1, len(dates)),
        'Close': np.linspace(101, 111, len(dates)) + np.random.normal(0, 1, len(dates)),
        'Volume': np.random.randint(1000000, 5000000, len(dates)),
        'Returns': np.random.normal(0.001, 0.01, len(dates)),
        'Volatility': np.random.uniform(0.1, 0.2, len(dates)),
        'RSI': np.random.uniform(30, 70, len(dates)),
        'MACD': np.random.normal(0, 1, len(dates)),
    }
    
    # Astrological features
    astro_data = {
        'Sun_longitude_sin': np.sin(np.linspace(0, 2*np.pi, len(dates))),
        'Sun_longitude_cos': np.cos(np.linspace(0, 2*np.pi, len(dates))),
        'Moon_longitude_sin': np.sin(np.linspace(0, 6*np.pi, len(dates))),
        'Moon_longitude_cos': np.cos(np.linspace(0, 6*np.pi, len(dates))),
        'Mercury_longitude_sin': np.sin(np.linspace(0, 3*np.pi, len(dates))),
        'Mercury_longitude_cos': np.cos(np.linspace(0, 3*np.pi, len(dates))),
        'Venus_longitude_sin': np.sin(np.linspace(0, 2.5*np.pi, len(dates))),
        'Venus_longitude_cos': np.cos(np.linspace(0, 2.5*np.pi, len(dates))),
        'Mars_longitude_sin': np.sin(np.linspace(0, 1.5*np.pi, len(dates))),
        'Mars_longitude_cos': np.cos(np.linspace(0, 1.5*np.pi, len(dates))),
        'Jupiter_longitude_sin': np.sin(np.linspace(0, 0.2*np.pi, len(dates))),
        'Jupiter_longitude_cos': np.cos(np.linspace(0, 0.2*np.pi, len(dates))),
        'Saturn_longitude_sin': np.sin(np.linspace(0, 0.1*np.pi, len(dates))),
        'Saturn_longitude_cos': np.cos(np.linspace(0, 0.1*np.pi, len(dates))),
        'Sun_Moon_aspect': np.random.uniform(0, 1, len(dates)),
        'Sun_Mercury_aspect': np.random.uniform(0, 1, len(dates)),
        'Moon_Jupiter_aspect': np.random.uniform(0, 1, len(dates)),
        'Venus_Mars_aspect': np.random.uniform(0, 1, len(dates)),
        'Jupiter_Saturn_aspect': np.random.uniform(0, 1, len(dates)),
    }
    
    # Combine market and astrological data
    data = {**market_data, **astro_data}
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df


class TestFeatureGenerator:
    """Tests for the feature generator module."""
    
    def test_generate_features(self, sample_integrated_data):
        """Test the main feature generation function."""
        df = sample_integrated_data.copy()
        
        # Call the function
        features_df = generate_features(df)
        
        # Verify the features DataFrame has the expected shape
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == len(df)
        
        # Verify that the function adds new features
        assert len(features_df.columns) > len(df.columns)
        
        # Verify that interaction features are included
        interaction_cols = [col for col in features_df.columns if 'interaction' in col.lower()]
        assert len(interaction_cols) > 0
    
    def test_create_interaction_features(self, sample_integrated_data):
        """Test creation of interaction features between market and astrological data."""
        df = sample_integrated_data.copy()
        
        # Select some market and astro features for testing
        market_features = ['Returns', 'Volatility', 'RSI']
        astro_features = ['Sun_Moon_aspect', 'Jupiter_Saturn_aspect']
        
        # Call the function
        interaction_df = create_interaction_features(df, market_features, astro_features)
        
        # Verify the interaction DataFrame has the expected columns
        expected_columns = [
            'Returns_x_Sun_Moon_aspect',
            'Returns_x_Jupiter_Saturn_aspect',
            'Volatility_x_Sun_Moon_aspect',
            'Volatility_x_Jupiter_Saturn_aspect',
            'RSI_x_Sun_Moon_aspect',
            'RSI_x_Jupiter_Saturn_aspect'
        ]
        
        for col in expected_columns:
            assert col in interaction_df.columns, f"Column {col} missing from interaction features"
        
        # Verify the calculations are correct
        for m_feat in market_features:
            for a_feat in astro_features:
                interaction_col = f"{m_feat}_x_{a_feat}"
                expected_values = df[m_feat] * df[a_feat]
                pd.testing.assert_series_equal(
                    interaction_df[interaction_col],
                    expected_values,
                    check_names=False
                )
    
    def test_normalize_features(self, sample_integrated_data):
        """Test normalization of features."""
        df = sample_integrated_data.copy()
        
        # Select columns to normalize
        columns_to_normalize = ['Returns', 'Volatility', 'RSI', 'Sun_Moon_aspect']
        
        # Call the function
        normalized_df = normalize_features(df[columns_to_normalize])
        
        # Verify the normalized DataFrame has the expected shape
        assert isinstance(normalized_df, pd.DataFrame)
        assert normalized_df.shape == df[columns_to_normalize].shape
        
        # Verify that all values are between 0 and 1 (or -1 and 1 for z-score)
        for col in normalized_df.columns:
            assert normalized_df[col].min() >= -3, f"Column {col} has values below -3"
            assert normalized_df[col].max() <= 3, f"Column {col} has values above 3"
            
            # Check that mean is close to 0 and std close to 1 for z-score normalization
            assert abs(normalized_df[col].mean()) < 0.1, f"Column {col} mean is not close to 0"
            assert abs(normalized_df[col].std() - 1.0) < 0.1, f"Column {col} std is not close to 1"
    
    def test_select_features(self, sample_integrated_data):
        """Test feature selection based on correlation with target."""
        df = sample_integrated_data.copy()
        
        # Add a target column for testing
        df['Target'] = df['Returns'].shift(-1)  # Next day's returns
        
        # Call the function
        selected_features = select_features(df.dropna(), 'Target', top_n=5)
        
        # Verify the selected features list has the expected length
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 5  # Could be less if there are ties
        
        # Verify that all selected features are columns in the original DataFrame
        for feature in selected_features:
            assert feature in df.columns, f"Selected feature {feature} not in original DataFrame"
        
        # Verify that 'Target' is not in the selected features
        assert 'Target' not in selected_features
    
    def test_feature_generator_class(self, sample_integrated_data):
        """Test the FeatureGenerator class."""
        df = sample_integrated_data.copy()
        
        # Initialize the generator
        generator = FeatureGenerator()
        
        # Generate features
        features_df = generator.generate_features(df)
        
        # Verify the features DataFrame has the expected shape
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == len(df)
        
        # Verify that the class method adds new features
        assert len(features_df.columns) > len(df.columns)
        
        # Test with feature selection
        df['Target'] = df['Returns'].shift(-1)  # Next day's returns
        generator_with_selection = FeatureGenerator(select_top_n=5)
        selected_features_df = generator_with_selection.generate_features(df.dropna(), target_column='Target')
        
        # Verify that the number of columns is reduced after selection
        assert len(selected_features_df.columns) <= 6  # 5 features + Target
