"""
Tests for the Bayesian time series models in the Cosmic Market Oracle.

This module contains unit tests for the time series Bayesian models,
ensuring that they function correctly for analyzing financial and astrological time series data.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pymc as pm

from src.statistical_modeling.time_series_bayesian import TimeSeriesModel
from src.statistical_modeling.ar_models import ARModel, SeasonalARModel
from src.statistical_modeling.arima_models import MAModel, ARMAModel
from src.statistical_modeling.arima_extension import ARIMAModel, ARIMAXModel
from src.statistical_modeling.seasonal_models import SeasonalDecompositionModel
from src.statistical_modeling.fourier_seasonal import FourierSeasonalModel


@pytest.fixture
def sample_time_series():
    """
    Generate a sample time series for testing.
    
    Returns:
        Pandas Series with sample time series data
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 100
    time_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate AR(1) process
    ar_coef = 0.7
    sigma = 0.5
    ts = np.zeros(n_samples)
    ts[0] = np.random.normal(0, 1)
    
    for t in range(1, n_samples):
        ts[t] = ar_coef * ts[t-1] + np.random.normal(0, sigma)
    
    # Create pandas Series
    time_series = pd.Series(ts, index=time_index)
    
    return time_series


@pytest.fixture
def sample_seasonal_time_series():
    """
    Generate a sample seasonal time series for testing.
    
    Returns:
        Pandas Series with sample seasonal time series data
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 100
    time_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate trend
    trend = 0.01 * np.arange(n_samples) + 10
    
    # Generate weekly seasonality
    weekly_period = 7
    weekly_seasonality = 3 * np.sin(2 * np.pi * np.arange(n_samples) / weekly_period)
    
    # Generate noise
    noise = np.random.normal(0, 1, n_samples)
    
    # Combine components
    ts = trend + weekly_seasonality + noise
    
    # Create pandas Series
    time_series = pd.Series(ts, index=time_index)
    
    return time_series


@pytest.fixture
def sample_exog_variables():
    """
    Generate sample exogenous variables for testing.
    
    Returns:
        Pandas DataFrame with sample exogenous variables
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 100
    time_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate exogenous variables
    exog1 = np.sin(2 * np.pi * np.arange(n_samples) / 30)  # Monthly cycle
    exog2 = np.random.normal(0, 1, n_samples)  # Random noise
    
    # Create pandas DataFrame
    exog = pd.DataFrame({
        'Monthly': exog1,
        'Random': exog2
    }, index=time_index)
    
    return exog


class TestTimeSeriesModel:
    """
    Tests for the base TimeSeriesModel class.
    """
    
    def test_initialization(self):
        """
        Test initialization of TimeSeriesModel.
        """
        model = TimeSeriesModel(name="test_model")
        
        assert model.name == "test_model"
        assert model.model is None
        assert model.trace is None
        assert model.time_series is None
        assert model.time_index is None
        assert model.n_timesteps is None
        assert model.forecast_horizon is None
        assert model.forecast_results is None
    
    def test_preprocess_data(self, sample_time_series):
        """
        Test data preprocessing.
        """
        model = TimeSeriesModel()
        
        # Test with pandas Series
        data, time_index = model.preprocess_data(sample_time_series)
        
        assert isinstance(data, np.ndarray)
        assert isinstance(time_index, pd.DatetimeIndex)
        assert len(data) == len(sample_time_series)
        assert np.array_equal(time_index, sample_time_series.index)
        
        # Test with numpy array
        data_array = sample_time_series.values
        data, time_index = model.preprocess_data(data_array)
        
        assert isinstance(data, np.ndarray)
        assert time_index is None
        assert len(data) == len(data_array)
        assert np.array_equal(data, data_array)
    
    def test_build_model_not_implemented(self):
        """
        Test that build_model raises NotImplementedError.
        """
        model = TimeSeriesModel()
        
        with pytest.raises(NotImplementedError):
            model.build_model()


class TestARModel:
    """
    Tests for the ARModel class.
    """
    
    def test_initialization(self):
        """
        Test initialization of ARModel.
        """
        model = ARModel(name="test_ar_model")
        
        assert model.name == "test_ar_model"
        assert model.model is None
        assert model.trace is None
        assert model.time_series is None
        assert model.time_index is None
        assert model.n_timesteps is None
        assert model.forecast_horizon is None
        assert model.forecast_results is None
        assert model.order is None
        assert model.ar_coefficients is None
    
    def test_build_model(self, sample_time_series):
        """
        Test building an AR model.
        """
        model = ARModel()
        
        # Build model
        built_model = model.build_model(sample_time_series, order=2)
        
        assert isinstance(built_model, pm.Model)
        assert model.order == 2
        assert model.time_series is not None
        assert model.time_index is not None
        assert model.n_timesteps == len(sample_time_series)
    
    def test_forecast_without_sampling(self, sample_time_series):
        """
        Test that forecast raises ValueError if model has not been sampled.
        """
        model = ARModel()
        model.build_model(sample_time_series, order=2)
        
        with pytest.raises(ValueError):
            model.forecast(horizon=10)


class TestSeasonalARModel:
    """
    Tests for the SeasonalARModel class.
    """
    
    def test_initialization(self):
        """
        Test initialization of SeasonalARModel.
        """
        model = SeasonalARModel(name="test_seasonal_ar_model")
        
        assert model.name == "test_seasonal_ar_model"
        assert model.model is None
        assert model.trace is None
        assert model.time_series is None
        assert model.time_index is None
        assert model.n_timesteps is None
        assert model.forecast_horizon is None
        assert model.forecast_results is None
        assert model.order is None
        assert model.ar_coefficients is None
        assert model.seasonal_period is None
        assert model.seasonal_order is None
    
    def test_build_model(self, sample_seasonal_time_series):
        """
        Test building a Seasonal AR model.
        """
        model = SeasonalARModel()
        
        # Build model
        built_model = model.build_model(
            sample_seasonal_time_series, 
            order=1, 
            seasonal_order=1, 
            seasonal_period=7
        )
        
        assert isinstance(built_model, pm.Model)
        assert model.order == 1
        assert model.seasonal_order == 1
        assert model.seasonal_period == 7
        assert model.time_series is not None
        assert model.time_index is not None
        assert model.n_timesteps == len(sample_seasonal_time_series)


class TestMAModel:
    """
    Tests for the MAModel class.
    """
    
    def test_initialization(self):
        """
        Test initialization of MAModel.
        """
        model = MAModel(name="test_ma_model")
        
        assert model.name == "test_ma_model"
        assert model.model is None
        assert model.trace is None
        assert model.time_series is None
        assert model.time_index is None
        assert model.n_timesteps is None
        assert model.forecast_horizon is None
        assert model.forecast_results is None
        assert model.order is None
        assert model.ma_coefficients is None
    
    def test_build_model(self, sample_time_series):
        """
        Test building an MA model.
        """
        model = MAModel()
        
        # Build model
        built_model = model.build_model(sample_time_series, order=2)
        
        assert isinstance(built_model, pm.Model)
        assert model.order == 2
        assert model.time_series is not None
        assert model.time_index is not None
        assert model.n_timesteps == len(sample_time_series)


class TestARMAModel:
    """
    Tests for the ARMAModel class.
    """
    
    def test_initialization(self):
        """
        Test initialization of ARMAModel.
        """
        model = ARMAModel(name="test_arma_model")
        
        assert model.name == "test_arma_model"
        assert model.model is None
        assert model.trace is None
        assert model.time_series is None
        assert model.time_index is None
        assert model.n_timesteps is None
        assert model.forecast_horizon is None
        assert model.forecast_results is None
        assert model.ar_order is None
        assert model.ma_order is None
        assert model.ar_coefficients is None
        assert model.ma_coefficients is None
    
    def test_build_model(self, sample_time_series):
        """
        Test building an ARMA model.
        """
        model = ARMAModel()
        
        # Build model
        built_model = model.build_model(
            sample_time_series, 
            ar_order=1, 
            ma_order=1
        )
        
        assert isinstance(built_model, pm.Model)
        assert model.ar_order == 1
        assert model.ma_order == 1
        assert model.time_series is not None
        assert model.time_index is not None
        assert model.n_timesteps == len(sample_time_series)


class TestARIMAModel:
    """
    Tests for the ARIMAModel class.
    """
    
    def test_initialization(self):
        """
        Test initialization of ARIMAModel.
        """
        model = ARIMAModel(name="test_arima_model")
        
        assert model.name == "test_arima_model"
        assert model.model is None
        assert model.trace is None
        assert model.time_series is None
        assert model.time_index is None
        assert model.n_timesteps is None
        assert model.forecast_horizon is None
        assert model.forecast_results is None
        assert model.ar_order is None
        assert model.ma_order is None
        assert model.ar_coefficients is None
        assert model.ma_coefficients is None
        assert model.d is None
        assert model.original_data is None
    
    def test_build_model(self, sample_time_series):
        """
        Test building an ARIMA model.
        """
        model = ARIMAModel()
        
        # Build model
        built_model = model.build_model(
            sample_time_series, 
            ar_order=1, 
            differencing=1, 
            ma_order=1
        )
        
        assert isinstance(built_model, pm.Model)
        assert model.ar_order == 1
        assert model.ma_order == 1
        assert model.d == 1
        assert model.time_series is not None
        assert model.time_index is not None
        assert model.n_timesteps == len(sample_time_series) - 1  # Due to differencing
        assert model.original_data is not None


class TestARIMAXModel:
    """
    Tests for the ARIMAXModel class.
    """
    
    def test_initialization(self):
        """
        Test initialization of ARIMAXModel.
        """
        model = ARIMAXModel(name="test_arimax_model")
        
        assert model.name == "test_arimax_model"
        assert model.model is None
        assert model.trace is None
        assert model.time_series is None
        assert model.time_index is None
        assert model.n_timesteps is None
        assert model.forecast_horizon is None
        assert model.forecast_results is None
        assert model.ar_order is None
        assert model.ma_order is None
        assert model.ar_coefficients is None
        assert model.ma_coefficients is None
        assert model.d is None
        assert model.original_data is None
        assert model.exog_variables is None
        assert model.exog_coefficients is None
    
    def test_build_model(self, sample_time_series, sample_exog_variables):
        """
        Test building an ARIMAX model.
        """
        model = ARIMAXModel()
        
        # Build model
        built_model = model.build_model(
            sample_time_series, 
            exog_variables=sample_exog_variables,
            ar_order=1, 
            differencing=1, 
            ma_order=1
        )
        
        assert isinstance(built_model, pm.Model)
        assert model.ar_order == 1
        assert model.ma_order == 1
        assert model.d == 1
        assert model.time_series is not None
        assert model.time_index is not None
        assert model.n_timesteps == len(sample_time_series) - 1  # Due to differencing
        assert model.original_data is not None
        assert model.exog_variables is not None
        assert model.exog_names == ['Monthly', 'Random']


class TestSeasonalDecompositionModel:
    """
    Tests for the SeasonalDecompositionModel class.
    """
    
    def test_initialization(self):
        """
        Test initialization of SeasonalDecompositionModel.
        """
        model = SeasonalDecompositionModel(name="test_seasonal_decomposition_model")
        
        assert model.name == "test_seasonal_decomposition_model"
        assert model.model is None
        assert model.trace is None
        assert model.time_series is None
        assert model.time_index is None
        assert model.n_timesteps is None
        assert model.forecast_horizon is None
        assert model.forecast_results is None
        assert model.seasonal_periods is None
        assert model.seasonal_components is None
        assert model.trend_order is None
        assert model.seasonal_harmonics is None
    
    def test_build_model(self, sample_seasonal_time_series):
        """
        Test building a Seasonal Decomposition model.
        """
        model = SeasonalDecompositionModel()
        
        # Build model
        built_model = model.build_model(
            sample_seasonal_time_series,
            trend_order=1,
            seasonal_periods=[7]
        )
        
        assert isinstance(built_model, pm.Model)
        assert model.trend_order == 1
        assert model.seasonal_periods == [7]
        assert model.seasonal_harmonics == {7: 3}  # Default is period // 2
        assert model.time_series is not None
        assert model.time_index is not None
        assert model.n_timesteps == len(sample_seasonal_time_series)


class TestFourierSeasonalModel:
    """
    Tests for the FourierSeasonalModel class.
    """
    
    def test_initialization(self):
        """
        Test initialization of FourierSeasonalModel.
        """
        model = FourierSeasonalModel(name="test_fourier_seasonal_model")
        
        assert model.name == "test_fourier_seasonal_model"
        assert model.model is None
        assert model.trace is None
        assert model.time_series is None
        assert model.time_index is None
        assert model.n_timesteps is None
        assert model.forecast_horizon is None
        assert model.forecast_results is None
        assert model.seasonal_periods is None
        assert model.seasonal_components is None
        assert model.trend_order is None
        assert model.fourier_orders is None
        assert model.regression_terms is None
    
    def test_build_model(self, sample_seasonal_time_series):
        """
        Test building a Fourier Seasonal model.
        """
        model = FourierSeasonalModel()
        
        # Build model
        built_model = model.build_model(
            sample_seasonal_time_series,
            trend_order=1,
            seasonal_periods=[7],
            fourier_orders={7: 3}
        )
        
        assert isinstance(built_model, pm.Model)
        assert model.trend_order == 1
        assert model.seasonal_periods == [7]
        assert model.fourier_orders == {7: 3}
        assert model.regression_terms is None
        assert model.time_series is not None
        assert model.time_index is not None
        assert model.n_timesteps == len(sample_seasonal_time_series)
    
    def test_build_model_with_regression(self, sample_seasonal_time_series, sample_exog_variables):
        """
        Test building a Fourier Seasonal model with regression terms.
        """
        model = FourierSeasonalModel()
        
        # Convert exog variables to dict
        regression_terms = {
            'Monthly': sample_exog_variables['Monthly'].values,
            'Random': sample_exog_variables['Random'].values
        }
        
        # Build model
        built_model = model.build_model(
            sample_seasonal_time_series,
            trend_order=1,
            seasonal_periods=[7],
            fourier_orders={7: 3},
            regression_terms=regression_terms
        )
        
        assert isinstance(built_model, pm.Model)
        assert model.trend_order == 1
        assert model.seasonal_periods == [7]
        assert model.fourier_orders == {7: 3}
        assert model.regression_terms is not None
        assert set(model.regression_terms.keys()) == {'Monthly', 'Random'}
        assert model.time_series is not None
        assert model.time_index is not None
        assert model.n_timesteps == len(sample_seasonal_time_series)


if __name__ == "__main__":
    # Run tests
    pytest.main(["-v", __file__])
