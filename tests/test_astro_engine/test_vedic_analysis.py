"""
Unit tests for the vedic_analysis module.
"""
import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from astro_engine.vedic_analysis import VedicAnalyzer
from astro_engine.planetary_positions import PlanetaryCalculator
from astro_engine.constants import SUN, MOON, MARS, MERCURY, JUPITER, VENUS, SATURN, RAHU, KETU


class TestVedicAnalyzer:
    """Test cases for the VedicAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a VedicAnalyzer instance for testing."""
        return VedicAnalyzer()
    
    @pytest.fixture
    def sample_positions(self):
        """Create sample planetary positions for testing."""
        return {
            'date': datetime(2023, 1, 1),
            SUN: {'longitude': 280.5, 'is_retrograde': False},
            MOON: {'longitude': 120.3, 'is_retrograde': False},
            MARS: {'longitude': 315.7, 'is_retrograde': False},
            MERCURY: {'longitude': 275.2, 'is_retrograde': True},
            JUPITER: {'longitude': 340.1, 'is_retrograde': False},
            VENUS: {'longitude': 300.8, 'is_retrograde': False},
            SATURN: {'longitude': 310.5, 'is_retrograde': True},
            RAHU: {'longitude': 15.3, 'is_retrograde': True},
            KETU: {'longitude': 195.3, 'is_retrograde': True}
        }
    
    @pytest.fixture
    def sample_dignities(self):
        """Create sample planetary dignities for testing."""
        return {
            SUN: {
                'house': 10,
                'sign': 10,
                'dignity': 'neutral',
                'combustion': {'is_combust': False}
            },
            MOON: {
                'house': 5,
                'sign': 5,
                'dignity': 'neutral',
                'combustion': {'is_combust': False}
            },
            MARS: {
                'house': 11,
                'sign': 11,
                'dignity': 'friend',
                'combustion': {'is_combust': False}
            },
            MERCURY: {
                'house': 10,
                'sign': 10,
                'dignity': 'neutral',
                'combustion': {'is_combust': True}
            },
            JUPITER: {
                'house': 12,
                'sign': 12,
                'dignity': 'exalted',
                'combustion': {'is_combust': False}
            },
            VENUS: {
                'house': 10,
                'sign': 10,
                'dignity': 'neutral',
                'combustion': {'is_combust': False}
            },
            SATURN: {
                'house': 11,
                'sign': 11,
                'dignity': 'neutral',
                'combustion': {'is_combust': False}
            },
            RAHU: {
                'house': 1,
                'sign': 1,
                'dignity': 'neutral',
                'combustion': {'is_combust': False}
            },
            KETU: {
                'house': 7,
                'sign': 7,
                'dignity': 'neutral',
                'combustion': {'is_combust': False}
            }
        }
    
    @pytest.fixture
    def sample_divisional_strengths(self):
        """Create sample divisional chart strengths for testing."""
        return {
            SUN: {'overall_strength': 0.7, 'volatility': 0.3},
            MOON: {'overall_strength': 0.6, 'volatility': 0.2},
            MARS: {'overall_strength': 0.5, 'volatility': 0.6},
            MERCURY: {'overall_strength': 0.4, 'volatility': 0.4},
            JUPITER: {'overall_strength': 0.8, 'volatility': 0.2},
            VENUS: {'overall_strength': 0.7, 'volatility': 0.3},
            SATURN: {'overall_strength': 0.5, 'volatility': 0.5},
            RAHU: {'overall_strength': 0.4, 'volatility': 0.7},
            KETU: {'overall_strength': 0.4, 'volatility': 0.7}
        }
    
    def test_analyze_date(self, analyzer, mocker):
        """Test the analyze_date method."""
        # Mock the PlanetaryCalculator to avoid actual calculations
        mock_calculator = mocker.patch('astro_engine.vedic_analysis.PlanetaryCalculator')
        mock_calculator_instance = mock_calculator.return_value
        
        # Mock the calculate_positions method to return sample positions
        sample_positions = {
            'date': datetime(2023, 1, 1),
            SUN: {'longitude': 280.5, 'is_retrograde': False},
            MOON: {'longitude': 120.3, 'is_retrograde': False}
        }
        mock_calculator_instance.calculate_positions.return_value = sample_positions
        
        # Mock the internal methods to avoid complex calculations
        mocker.patch.object(analyzer, '_calculate_dignities', return_value={})
        mocker.patch.object(analyzer, '_calculate_divisional_strengths', return_value={})
        mocker.patch.object(analyzer, '_calculate_yogas', return_value={})
        mocker.patch.object(analyzer, '_calculate_dasha_influences', return_value={})
        mocker.patch.object(analyzer, '_integrate_forecasts', return_value={
            'trend': 'bullish',
            'volatility': 'moderate',
            'confidence': 0.75,
            'description': 'Test forecast'
        })
        mocker.patch.object(analyzer, '_generate_sector_forecasts', return_value={})
        mocker.patch.object(analyzer, '_generate_timing_signals', return_value={})
        
        # Call the analyze_date method
        result = analyzer.analyze_date("2023-01-01")
        
        # Check that the result has the expected structure
        assert isinstance(result, dict)
        assert 'date' in result
        assert 'positions' in result
        assert 'integrated_forecast' in result
        assert result['integrated_forecast']['trend'] == 'bullish'
        assert result['integrated_forecast']['volatility'] == 'moderate'
    
    def test_integrate_forecasts(self, analyzer, sample_positions, sample_dignities, sample_divisional_strengths):
        """Test the _integrate_forecasts method."""
        # Create sample yoga and dasha forecasts
        yoga_forecast = {
            'trend_score': 0.3,
            'volatility_score': 0.5,
            'confidence': 0.7
        }
        
        dasha_forecast = {
            'trend_score': 0.2,
            'volatility_score': 0.4,
            'confidence': 0.6
        }
        
        # Call the _integrate_forecasts method
        result = analyzer._integrate_forecasts(
            yoga_forecast, 
            dasha_forecast, 
            sample_dignities, 
            sample_divisional_strengths, 
            sample_positions
        )
        
        # Check that the result has the expected structure
        assert isinstance(result, dict)
        assert 'trend' in result
        assert 'trend_score' in result
        assert 'volatility' in result
        assert 'volatility_score' in result
        assert 'confidence' in result
        assert 'description' in result
        
        # Check that the trend is one of the expected values
        assert result['trend'] in ['bullish', 'bearish', 'neutral']
        
        # Check that the volatility is one of the expected values
        assert result['volatility'] in ['high', 'moderate', 'low']
    
    def test_analyze_date_range(self, analyzer, mocker):
        """Test the analyze_date_range method."""
        # Mock the analyze_date method to return a sample result
        sample_result = {
            'date': datetime(2023, 1, 1),
            'integrated_forecast': {
                'trend': 'bullish',
                'trend_score': 0.5,
                'volatility': 'moderate',
                'volatility_score': 0.4,
                'confidence': 0.7,
                'description': 'Test forecast'
            }
        }
        mocker.patch.object(analyzer, 'analyze_date', return_value=sample_result)
        
        # Call the analyze_date_range method
        start_date = "2023-01-01"
        end_date = "2023-01-03"
        result = analyzer.analyze_date_range(start_date, end_date)
        
        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        expected_columns = ['date', 'trend', 'trend_score', 'volatility', 'volatility_score', 'confidence']
        for col in expected_columns:
            assert col in result.columns
        
        # Check that the DataFrame has the expected number of rows (3 days)
        assert len(result) == 3
    
    def test_generate_sector_forecasts(self, analyzer, sample_positions, sample_dignities):
        """Test the _generate_sector_forecasts method."""
        # Create sample yogas
        yogas = {
            'raja_yoga': [{'planets': [SUN, JUPITER], 'strength': 0.8}],
            'dhana_yoga': [{'planets': [VENUS, JUPITER], 'strength': 0.7}]
        }
        
        # Call the _generate_sector_forecasts method
        result = analyzer._generate_sector_forecasts(sample_positions, sample_dignities, yogas)
        
        # Check that the result has the expected structure
        assert isinstance(result, dict)
        
        # Check that the result contains sector forecasts
        assert len(result) > 0
        
        # Check the structure of each sector forecast
        for sector, forecast in result.items():
            assert isinstance(sector, str)
            assert isinstance(forecast, dict)
            assert 'outlook' in forecast
            assert forecast['outlook'] in ['bullish', 'bearish', 'neutral']
            assert 'score' in forecast
            assert isinstance(forecast['score'], float)
    
    def test_generate_timing_signals(self, analyzer, sample_positions):
        """Test the _generate_timing_signals method."""
        # Call the _generate_timing_signals method
        result = analyzer._generate_timing_signals(sample_positions)
        
        # Check that the result has the expected structure
        assert isinstance(result, dict)
        
        # Check that the result contains timing signals
        assert 'entry' in result
        assert 'exit' in result
        
        # Check the structure of each timing signal
        for signal_type in ['entry', 'exit']:
            assert isinstance(result[signal_type], dict)
            assert 'signals' in result[signal_type]
            assert isinstance(result[signal_type]['signals'], list)
            assert 'strength' in result[signal_type]
            assert isinstance(result[signal_type]['strength'], float)
