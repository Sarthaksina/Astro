"""
Pytest fixtures for astro_engine tests.
"""
import pytest
from datetime import datetime
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from astro_engine.constants import SUN, MOON, MARS, MERCURY, JUPITER, VENUS, SATURN, RAHU, KETU
from astro_engine.planetary_positions import PlanetaryCalculator


@pytest.fixture
def mock_planetary_calculator():
    """Create a mock PlanetaryCalculator for testing."""
    calculator = MagicMock(spec=PlanetaryCalculator)
    
    # Mock the calculate_positions method to return sample positions
    sample_positions = {
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
    calculator.calculate_positions.return_value = sample_positions
    
    return calculator


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    prices = np.linspace(100, 110, len(dates)) + np.random.normal(0, 1, len(dates))
    
    return pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
