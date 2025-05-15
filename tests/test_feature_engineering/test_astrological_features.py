# Cosmic Market Oracle - Tests for Astrological Features Module

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.feature_engineering.astrological_features import (
    AstrologicalFeatureGenerator,
    encode_cyclical_features,
    calculate_aspect_strength,
    get_planet_phase,
    calculate_planet_strength
)


@pytest.fixture
def sample_planetary_data():
    """Create sample planetary data for testing."""
    # Create a DataFrame with 30 days of mock planetary positions
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create sample data for Sun, Moon, and Jupiter
    data = {
        'Date': dates,
        'Sun_longitude': np.linspace(0, 30, len(dates)),  # Moving through one sign
        'Sun_latitude': np.zeros(len(dates)),
        'Sun_distance': np.ones(len(dates)) * 1.0,
        'Sun_speed': np.ones(len(dates)) * 1.0,
        'Sun_is_retrograde': np.zeros(len(dates), dtype=bool),
        
        'Moon_longitude': np.linspace(0, 360, len(dates)),  # Full cycle
        'Moon_latitude': np.sin(np.linspace(0, np.pi, len(dates))) * 5,
        'Moon_distance': np.ones(len(dates)) * 0.3,
        'Moon_speed': np.ones(len(dates)) * 13.0,
        'Moon_is_retrograde': np.zeros(len(dates), dtype=bool),
        
        'Jupiter_longitude': np.linspace(90, 92, len(dates)),  # Slow movement
        'Jupiter_latitude': np.zeros(len(dates)),
        'Jupiter_distance': np.ones(len(dates)) * 5.0,
        'Jupiter_speed': np.ones(len(dates)) * 0.1,
        'Jupiter_is_retrograde': np.zeros(len(dates), dtype=bool),
    }
    
    # Add Saturn with retrograde motion
    retrograde_indices = np.arange(len(dates) // 2, len(dates))
    data['Saturn_longitude'] = np.linspace(180, 178, len(dates))  # Moving backward
    data['Saturn_latitude'] = np.zeros(len(dates))
    data['Saturn_distance'] = np.ones(len(dates)) * 9.0
    data['Saturn_speed'] = np.ones(len(dates)) * 0.05
    data['Saturn_is_retrograde'] = np.zeros(len(dates), dtype=bool)
    data['Saturn_is_retrograde'][retrograde_indices] = True
    data['Saturn_speed'][retrograde_indices] *= -1  # Negative speed for retrograde
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df


class TestAstrologicalFeatures:
    """Tests for astrological feature generation functions."""
    
    def test_encode_cyclical_features(self):
        """Test encoding of cyclical features (e.g., longitude)."""
        # Test with a simple array of longitudes
        longitudes = np.array([0, 90, 180, 270, 360])
        
        # Call the function
        sin_long, cos_long = encode_cyclical_features(longitudes)
        
        # Verify the results have the expected shape
        assert len(sin_long) == len(longitudes)
        assert len(cos_long) == len(longitudes)
        
        # Verify specific values
        expected_sin = np.sin(np.radians(longitudes))
        expected_cos = np.cos(np.radians(longitudes))
        
        np.testing.assert_allclose(sin_long, expected_sin, rtol=1e-10)
        np.testing.assert_allclose(cos_long, expected_cos, rtol=1e-10)
        
        # Test with a pandas Series
        longitude_series = pd.Series(longitudes)
        sin_series, cos_series = encode_cyclical_features(longitude_series)
        
        assert isinstance(sin_series, pd.Series)
        assert isinstance(cos_series, pd.Series)
        np.testing.assert_allclose(sin_series.values, expected_sin, rtol=1e-10)
        np.testing.assert_allclose(cos_series.values, expected_cos, rtol=1e-10)
    
    def test_calculate_aspect_strength(self):
        """Test calculation of aspect strength based on orb."""
        # Test exact aspect (0 orb)
        exact_strength = calculate_aspect_strength(0, max_orb=10)
        assert exact_strength == 1.0
        
        # Test aspect at half the maximum orb
        half_orb_strength = calculate_aspect_strength(5, max_orb=10)
        assert 0 < half_orb_strength < 1.0
        
        # Test aspect at maximum orb
        max_orb_strength = calculate_aspect_strength(10, max_orb=10)
        assert max_orb_strength > 0 and max_orb_strength < 0.1  # Should be very weak
        
        # Test aspect beyond maximum orb
        beyond_max_strength = calculate_aspect_strength(11, max_orb=10)
        assert beyond_max_strength == 0.0
        
        # Test with different weighting function
        linear_strength = calculate_aspect_strength(5, max_orb=10, weighting='linear')
        assert linear_strength == 0.5  # Linear should be exactly 0.5 at half orb
    
    def test_get_planet_phase(self):
        """Test determination of planetary phase."""
        # Test conjunction (same longitude)
        conjunction_phase = get_planet_phase(0, 0)
        assert conjunction_phase == 'conjunction'
        
        # Test opposition (180 degrees apart)
        opposition_phase = get_planet_phase(0, 180)
        assert opposition_phase == 'opposition'
        
        # Test first quarter (90 degrees apart)
        first_quarter = get_planet_phase(0, 90)
        assert first_quarter == 'first_quarter'
        
        # Test third quarter (270 degrees apart)
        third_quarter = get_planet_phase(0, 270)
        assert third_quarter == 'third_quarter'
        
        # Test with orb
        near_conjunction = get_planet_phase(0, 5, orb=10)
        assert near_conjunction == 'conjunction'
        
        # Test with values outside of 0-360 range
        wrapped_phase = get_planet_phase(370, 10)
        assert wrapped_phase == 'conjunction'
    
    def test_calculate_planet_strength(self):
        """Test calculation of planetary strength based on various factors."""
        # Test with a planet at its domicile
        domicile_strength = calculate_planet_strength(
            planet='Sun',
            longitude=120,  # Leo
            latitude=0,
            speed=1.0,
            distance=1.0,
            is_retrograde=False
        )
        
        # Test with a retrograde planet
        retrograde_strength = calculate_planet_strength(
            planet='Mercury',
            longitude=60,
            latitude=0,
            speed=-0.5,
            distance=0.8,
            is_retrograde=True
        )
        
        # Retrograde should weaken the planet
        assert retrograde_strength < domicile_strength
        
        # Test with a planet at its fall
        fall_strength = calculate_planet_strength(
            planet='Saturn',
            longitude=100,  # Cancer
            latitude=0,
            speed=0.1,
            distance=9.0,
            is_retrograde=False
        )
        
        # A planet in its fall should be weaker
        assert fall_strength < domicile_strength
    
    def test_astrological_feature_generator(self, sample_planetary_data):
        """Test the AstrologicalFeatureGenerator class."""
        df = sample_planetary_data.copy()
        
        # Initialize the generator
        generator = AstrologicalFeatureGenerator()
        
        # Generate features
        features_df = generator.generate_features(df)
        
        # Verify the features DataFrame has the expected columns
        expected_columns = [
            # Cyclical encodings
            'Sun_longitude_sin', 'Sun_longitude_cos',
            'Moon_longitude_sin', 'Moon_longitude_cos',
            'Jupiter_longitude_sin', 'Jupiter_longitude_cos',
            'Saturn_longitude_sin', 'Saturn_longitude_cos',
            
            # Planet strengths
            'Sun_strength', 'Moon_strength', 'Jupiter_strength', 'Saturn_strength',
            
            # Aspect features
            'Sun_Moon_aspect', 'Sun_Jupiter_aspect', 'Sun_Saturn_aspect',
            'Moon_Jupiter_aspect', 'Moon_Saturn_aspect', 'Jupiter_Saturn_aspect',
        ]
        
        for col in expected_columns:
            assert col in features_df.columns, f"Column {col} missing from features DataFrame"
        
        # Verify the DataFrame has the same length as the input
        assert len(features_df) == len(df)
        
        # Verify that cyclical encoding preserves the information
        for planet in ['Sun', 'Moon', 'Jupiter', 'Saturn']:
            # Reconstruct longitude from sin and cos (approximately)
            sin_col = f"{planet}_longitude_sin"
            cos_col = f"{planet}_longitude_cos"
            
            reconstructed_long = np.degrees(np.arctan2(
                features_df[sin_col],
                features_df[cos_col]
            )) % 360
            
            original_long = df[f"{planet}_longitude"]
            
            # Allow for some numerical precision issues
            np.testing.assert_allclose(
                reconstructed_long,
                original_long,
                rtol=1e-10,
                atol=1e-10
            )
