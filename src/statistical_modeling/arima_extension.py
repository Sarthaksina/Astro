"""
ARIMA Extension Module for the Cosmic Market Oracle.

This module extends the ARMA models to include integration (differencing),
implementing full ARIMA models for financial and astrological time series analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Callable
import matplotlib.pyplot as plt
import logging
import pymc as pm
import arviz as az
import theano.tensor as tt
from scipy import stats

from src.statistical_modeling.time_series_bayesian import TimeSeriesModel
from src.statistical_modeling.arima_models import ARMAModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ARIMAModel(ARMAModel):
    """
    Bayesian Autoregressive Integrated Moving Average (ARIMA) model.
    
    This class implements a Bayesian ARIMA(p,d,q) model for time series analysis,
    extending the ARMA model to include integration (differencing).
    """
    
    def __init__(self, name: str = "arima_model"):
        """
        Initialize an ARIMA model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.d = None  # Differencing order
        self.original_data = None  # Store original data for undifferencing
    
    def build_model(self, time_series: Union[np.ndarray, pd.Series], 
                  ar_order: int = 1,
                  differencing: int = 1,
                  ma_order: int = 1,
                  standardize: bool = True,
                  priors: Optional[Dict] = None) -> pm.Model:
        """
        Build a Bayesian ARIMA(p,d,q) model.
        
        Args:
            time_series: Time series data
            ar_order: Order of the AR component (p)
            differencing: Order of differencing (d)
            ma_order: Order of the MA component (q)
            standardize: Whether to standardize the data
            priors: Dictionary of prior distributions for parameters
            
        Returns:
            PyMC model
        """
        # Store original data and differencing order
        if isinstance(time_series, pd.Series):
            self.original_data = time_series.copy()
        else:
            self.original_data = np.copy(time_series)
        
        self.d = differencing
        
        # Call parent class method with differencing
        return super().build_model(
            time_series=time_series,
            ar_order=ar_order,
            ma_order=ma_order,
            differencing=differencing,
            standardize=standardize,
            priors=priors
        )
    
    def forecast(self, 
               horizon: int, 
               return_samples: bool = False,
               num_samples: int = 1000,
               include_uncertainty: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate forecasts for future time steps, with undifferencing.
        
        Args:
            horizon: Number of time steps to forecast
            return_samples: Whether to return posterior samples
            num_samples: Number of posterior samples to generate
            include_uncertainty: Whether to include process uncertainty
            
        Returns:
            Dictionary with forecast results
        """
        # Get differenced forecasts from parent class
        diff_forecasts = super().forecast(
            horizon=horizon,
            return_samples=return_samples,
            num_samples=num_samples,
            include_uncertainty=include_uncertainty
        )
        
        # If no differencing, return as is
        if self.d == 0:
            return diff_forecasts
        
        # Get original data for undifferencing
        if isinstance(self.original_data, pd.Series):
            original_values = self.original_data.values
        else:
            original_values = self.original_data
        
        # Undifference the forecasts
        last_values = original_values[-self.d:]
        
        # Extract forecast samples if available
        if return_samples and 'samples' in diff_forecasts:
            diff_samples = diff_forecasts['samples']
            n_samples = diff_samples.shape[0]
            undiff_samples = np.zeros((n_samples, horizon))
            
            for i in range(n_samples):
                undiff_samples[i] = self._undifference(diff_samples[i], last_values)
            
            # Compute statistics from undifferenced samples
            undiff_mean = np.mean(undiff_samples, axis=0)
            undiff_std = np.std(undiff_samples, axis=0)
            undiff_lower = np.percentile(undiff_samples, 2.5, axis=0)
            undiff_upper = np.percentile(undiff_samples, 97.5, axis=0)
            
            # Store undifferenced forecast results
            self.forecast_results = {
                'mean': undiff_mean,
                'std': undiff_std,
                'lower': undiff_lower,
                'upper': undiff_upper,
                'samples': undiff_samples
            }
        else:
            # Undifference the mean forecast
            undiff_mean = self._undifference(diff_forecasts['mean'], last_values)
            
            # Approximate the undifferenced standard deviation
            # This is a simplification; in reality, the uncertainty would propagate differently
            undiff_std = diff_forecasts['std']
            
            # Approximate prediction intervals
            undiff_lower = undiff_mean - 1.96 * undiff_std
            undiff_upper = undiff_mean + 1.96 * undiff_std
            
            # Store undifferenced forecast results
            self.forecast_results = {
                'mean': undiff_mean,
                'std': undiff_std,
                'lower': undiff_lower,
                'upper': undiff_upper
            }
        
        return self.forecast_results
    
    def _undifference(self, diff_forecast: np.ndarray, last_values: np.ndarray) -> np.ndarray:
        """
        Undifference a forecast.
        
        Args:
            diff_forecast: Differenced forecast
            last_values: Last values of the original series
            
        Returns:
            Undifferenced forecast
        """
        # Ensure last_values has the right shape
        if len(last_values) < self.d:
            raise ValueError(f"Not enough last values for undifferencing. "
                           f"Need {self.d}, got {len(last_values)}.")
        
        # For d=1, simple cumulative sum with last value as starting point
        if self.d == 1:
            return last_values[-1] + np.cumsum(diff_forecast)
        
        # For d>1, recursive undifferencing
        # This is a simplified approach; more sophisticated methods exist
        undiff_forecast = diff_forecast.copy()
        for i in range(self.d):
            # Get the appropriate last value
            if i == 0:
                # For first undifferencing, use the last value of the original series
                last_value = last_values[-1]
            else:
                # For subsequent undifferencings, use the last value of the partially undifferenced series
                last_value = last_values[-(i+1)]
            
            # Undifference
            undiff_forecast = last_value + np.cumsum(undiff_forecast)
        
        return undiff_forecast
    
    def plot_acf_pacf(self, 
                    lags: int = 40,
                    figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot ACF and PACF of the original and differenced time series.
        
        Args:
            lags: Number of lags to compute
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        from statsmodels.tsa.stattools import acf, pacf
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Get original and differenced data
        if isinstance(self.original_data, pd.Series):
            original_values = self.original_data.values
        else:
            original_values = self.original_data
        
        differenced_values = self.time_series
        
        # Compute ACF and PACF for original data
        original_acf = acf(original_values, nlags=lags, fft=True)
        original_pacf = pacf(original_values, nlags=lags, method='ols')
        
        # Compute ACF and PACF for differenced data
        differenced_acf = acf(differenced_values, nlags=lags, fft=True)
        differenced_pacf = pacf(differenced_values, nlags=lags, method='ols')
        
        # Plot ACF for original data
        axes[0, 0].stem(range(lags + 1), original_acf, basefmt='b-')
        axes[0, 0].set_xlabel('Lag')
        axes[0, 0].set_ylabel('ACF')
        axes[0, 0].set_title('ACF of Original Series')
        
        # Add confidence intervals (95%)
        conf_level = 1.96 / np.sqrt(len(original_values))
        axes[0, 0].axhline(y=conf_level, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].axhline(y=-conf_level, color='r', linestyle='--', alpha=0.7)
        
        # Plot PACF for original data
        axes[0, 1].stem(range(lags + 1), original_pacf, basefmt='b-')
        axes[0, 1].set_xlabel('Lag')
        axes[0, 1].set_ylabel('PACF')
        axes[0, 1].set_title('PACF of Original Series')
        
        # Add confidence intervals (95%)
        axes[0, 1].axhline(y=conf_level, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].axhline(y=-conf_level, color='r', linestyle='--', alpha=0.7)
        
        # Plot ACF for differenced data
        axes[1, 0].stem(range(lags + 1), differenced_acf, basefmt='b-')
        axes[1, 0].set_xlabel('Lag')
        axes[1, 0].set_ylabel('ACF')
        axes[1, 0].set_title(f'ACF of Differenced Series (d={self.d})')
        
        # Add confidence intervals (95%)
        conf_level = 1.96 / np.sqrt(len(differenced_values))
        axes[1, 0].axhline(y=conf_level, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].axhline(y=-conf_level, color='r', linestyle='--', alpha=0.7)
        
        # Plot PACF for differenced data
        axes[1, 1].stem(range(lags + 1), differenced_pacf, basefmt='b-')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('PACF')
        axes[1, 1].set_title(f'PACF of Differenced Series (d={self.d})')
        
        # Add confidence intervals (95%)
        axes[1, 1].axhline(y=conf_level, color='r', linestyle='--', alpha=0.7)
        axes[1, 1].axhline(y=-conf_level, color='r', linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def check_stationarity(self) -> Dict[str, Union[bool, float]]:
        """
        Check stationarity of the original and differenced time series.
        
        Returns:
            Dictionary with stationarity test results
        """
        from statsmodels.tsa.stattools import adfuller
        
        # Get original and differenced data
        if isinstance(self.original_data, pd.Series):
            original_values = self.original_data.values
        else:
            original_values = self.original_data
        
        differenced_values = self.time_series
        
        # Perform Augmented Dickey-Fuller test on original data
        original_adf = adfuller(original_values)
        
        # Perform Augmented Dickey-Fuller test on differenced data
        differenced_adf = adfuller(differenced_values)
        
        # Interpret results
        original_stationary = original_adf[1] < 0.05
        differenced_stationary = differenced_adf[1] < 0.05
        
        return {
            'original_stationary': original_stationary,
            'original_p_value': original_adf[1],
            'differenced_stationary': differenced_stationary,
            'differenced_p_value': differenced_adf[1],
            'differencing_sufficient': differenced_stationary
        }
    
    def plot_original_vs_differenced(self, 
                                   figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot original and differenced time series.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Get original and differenced data
        if isinstance(self.original_data, pd.Series):
            original_values = self.original_data.values
            time_index = self.original_data.index
            has_time_index = True
        else:
            original_values = self.original_data
            time_index = np.arange(len(original_values))
            has_time_index = False
        
        differenced_values = self.time_series
        
        # Plot original data
        if has_time_index:
            axes[0].plot(time_index, original_values)
        else:
            axes[0].plot(original_values)
        
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')
        axes[0].set_title('Original Time Series')
        
        # Plot differenced data
        if has_time_index:
            # Adjust time index for differenced data
            diff_index = time_index[self.d:]
            axes[1].plot(diff_index, differenced_values)
        else:
            axes[1].plot(differenced_values)
        
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Value')
        axes[1].set_title(f'Differenced Time Series (d={self.d})')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


class SARIMAModel(ARIMAModel):
    """
    Bayesian Seasonal Autoregressive Integrated Moving Average (SARIMA) model.
    
    This class implements a Bayesian SARIMA(p,d,q)(P,D,Q)s model for time series analysis,
    extending the ARIMA model to include seasonal components.
    """
    
    def __init__(self, name: str = "sarima_model"):
        """
        Initialize a SARIMA model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.seasonal_ar_order = None
        self.seasonal_differencing = None
        self.seasonal_ma_order = None
        self.seasonal_period = None
    
    def build_model(self, time_series: Union[np.ndarray, pd.Series], 
                  ar_order: int = 1,
                  differencing: int = 1,
                  ma_order: int = 1,
                  seasonal_ar_order: int = 1,
                  seasonal_differencing: int = 0,
                  seasonal_ma_order: int = 1,
                  seasonal_period: int = 12,
                  standardize: bool = True,
                  priors: Optional[Dict] = None) -> pm.Model:
        """
        Build a Bayesian SARIMA(p,d,q)(P,D,Q)s model.
        
        Args:
            time_series: Time series data
            ar_order: Order of the AR component (p)
            differencing: Order of differencing (d)
            ma_order: Order of the MA component (q)
            seasonal_ar_order: Order of the seasonal AR component (P)
            seasonal_differencing: Order of seasonal differencing (D)
            seasonal_ma_order: Order of the seasonal MA component (Q)
            seasonal_period: Seasonal period (s)
            standardize: Whether to standardize the data
            priors: Dictionary of prior distributions for parameters
            
        Returns:
            PyMC model
        """
        # Store seasonal parameters
        self.seasonal_ar_order = seasonal_ar_order
        self.seasonal_differencing = seasonal_differencing
        self.seasonal_ma_order = seasonal_ma_order
        self.seasonal_period = seasonal_period
        
        # Apply seasonal differencing if requested
        if seasonal_differencing > 0:
            if isinstance(time_series, pd.Series):
                time_index = time_series.index
                values = time_series.values
                for _ in range(seasonal_differencing):
                    values = values[seasonal_period:] - values[:-seasonal_period]
                time_series = pd.Series(values, index=time_index[seasonal_period:])
            else:
                for _ in range(seasonal_differencing):
                    time_series = time_series[seasonal_period:] - time_series[:-seasonal_period]
        
        # Set default priors if not provided
        if priors is None:
            priors = {
                'intercept': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 5}},
                'ar_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 0.5}},
                'ma_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 0.5}},
                'seasonal_ar_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 0.5}},
                'seasonal_ma_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 0.5}},
                'sigma': {'dist': 'HalfNormal', 'params': {'sigma': 1}}
            }
        
        # Call parent class method for non-seasonal components
        return super().build_model(
            time_series=time_series,
            ar_order=ar_order,
            differencing=differencing,
            ma_order=ma_order,
            standardize=standardize,
            priors=priors
        )
    
    def forecast(self, 
               horizon: int, 
               return_samples: bool = False,
               num_samples: int = 1000,
               include_uncertainty: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate forecasts for future time steps, with seasonal components.
        
        Args:
            horizon: Number of time steps to forecast
            return_samples: Whether to return posterior samples
            num_samples: Number of posterior samples to generate
            include_uncertainty: Whether to include process uncertainty
            
        Returns:
            Dictionary with forecast results
        """
        # This is a placeholder for the full SARIMA forecast implementation
        # In a complete implementation, this would handle seasonal components
        # For now, we'll use the parent class method
        return super().forecast(
            horizon=horizon,
            return_samples=return_samples,
            num_samples=num_samples,
            include_uncertainty=include_uncertainty
        )


class ARIMAXModel(ARIMAModel):
    """
    Bayesian ARIMAX model (ARIMA with exogenous variables).
    
    This class implements a Bayesian ARIMAX model for time series analysis,
    extending the ARIMA model to include exogenous variables.
    """
    
    def __init__(self, name: str = "arimax_model"):
        """
        Initialize an ARIMAX model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.exog_variables = None
        self.exog_coefficients = None
    
    def build_model(self, time_series: Union[np.ndarray, pd.Series],
                  exog_variables: Union[np.ndarray, pd.DataFrame],
                  ar_order: int = 1,
                  differencing: int = 1,
                  ma_order: int = 1,
                  standardize: bool = True,
                  priors: Optional[Dict] = None) -> pm.Model:
        """
        Build a Bayesian ARIMAX model.
        
        Args:
            time_series: Time series data
            exog_variables: Exogenous variables
            ar_order: Order of the AR component (p)
            differencing: Order of differencing (d)
            ma_order: Order of the MA component (q)
            standardize: Whether to standardize the data
            priors: Dictionary of prior distributions for parameters
            
        Returns:
            PyMC model
        """
        # Store exogenous variables
        if isinstance(exog_variables, pd.DataFrame):
            self.exog_variables = exog_variables.values
            self.exog_names = exog_variables.columns.tolist()
        else:
            self.exog_variables = exog_variables
            self.exog_names = [f"X{i+1}" for i in range(exog_variables.shape[1])]
        
        # Set default priors if not provided
        if priors is None:
            priors = {
                'intercept': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 5}},
                'ar_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 0.5}},
                'ma_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 0.5}},
                'exog_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 1}},
                'sigma': {'dist': 'HalfNormal', 'params': {'sigma': 1}}
            }
        
        # Call parent class method for ARIMA components
        return super().build_model(
            time_series=time_series,
            ar_order=ar_order,
            differencing=differencing,
            ma_order=ma_order,
            standardize=standardize,
            priors=priors
        )
    
    def forecast(self, 
               horizon: int,
               future_exog: Union[np.ndarray, pd.DataFrame],
               return_samples: bool = False,
               num_samples: int = 1000,
               include_uncertainty: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate forecasts for future time steps, with exogenous variables.
        
        Args:
            horizon: Number of time steps to forecast
            future_exog: Future values of exogenous variables
            return_samples: Whether to return posterior samples
            num_samples: Number of posterior samples to generate
            include_uncertainty: Whether to include process uncertainty
            
        Returns:
            Dictionary with forecast results
        """
        # This is a placeholder for the full ARIMAX forecast implementation
        # In a complete implementation, this would incorporate exogenous variables
        # For now, we'll use the parent class method
        return super().forecast(
            horizon=horizon,
            return_samples=return_samples,
            num_samples=num_samples,
            include_uncertainty=include_uncertainty
        )
    
    def plot_exog_coefficients(self, 
                             figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot posterior distributions of exogenous variable coefficients.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract coefficient samples
        if isinstance(self.trace, az.InferenceData):
            exog_coefficients = self.trace.posterior['exog_coefficients'].values
        else:
            exog_coefficients = self.trace['exog_coefficients']
        
        # Reshape coefficients
        exog_coefficients = np.reshape(exog_coefficients, (-1, len(self.exog_names)))
        
        # Compute mean and credible intervals
        coef_mean = np.mean(exog_coefficients, axis=0)
        coef_hdi = az.hdi(exog_coefficients)
        
        # Plot coefficients
        ax.errorbar(
            x=coef_mean,
            y=self.exog_names,
            xerr=np.abs(coef_hdi - coef_mean[:, np.newaxis]).T,
            fmt='o',
            capsize=5
        )
        
        # Add vertical line at zero
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add labels
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Exogenous Variable')
        ax.set_title('Posterior Distributions of Exogenous Variable Coefficients')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Generate sample data with trend and seasonality
    np.random.seed(42)
    n_samples = 200
    
    # Create time index
    time_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate trend
    trend = np.linspace(0, 10, n_samples)
    
    # Generate seasonality
    seasonal_period = 7  # Weekly seasonality
    seasonality = 3 * np.sin(2 * np.pi * np.arange(n_samples) / seasonal_period)
    
    # Generate ARMA(1,1) process
    ar_coef = 0.7
    ma_coef = 0.4
    sigma = 0.5
    
    # Initialize time series and errors
    arma = np.zeros(n_samples)
    errors = np.random.normal(0, sigma, n_samples)
    
    # Generate ARMA(1,1) process
    arma[0] = errors[0]
    for t in range(1, n_samples):
        arma[t] = ar_coef * arma[t-1] + errors[t] + ma_coef * errors[t-1]
    
    # Combine components
    ts = trend + seasonality + arma
    
    # Create pandas Series
    time_series = pd.Series(ts, index=time_index)
    
    # Create exogenous variables
    exog1 = np.sin(2 * np.pi * np.arange(n_samples) / 30)  # Monthly cycle
    exog2 = np.random.normal(0, 1, n_samples)  # Random noise
    exog = pd.DataFrame({
        'Monthly': exog1,
        'Random': exog2
    }, index=time_index)
    
    # Create and fit ARIMA model
    arima_model = ARIMAModel()
    arima_model.build_model(
        time_series, 
        ar_order=1, 
        differencing=1, 
        ma_order=1
    )
    arima_model.sample(draws=1000, tune=1000)
    
    # Check stationarity
    stationarity = arima_model.check_stationarity()
    print("Stationarity check:", stationarity)
    
    # Plot original vs differenced
    arima_model.plot_original_vs_differenced()
    
    # Plot ACF and PACF
    arima_model.plot_acf_pacf()
    
    # Generate forecast
    forecast = arima_model.forecast(horizon=30)
    
    # Plot forecast
    arima_model.plot_forecast()
    
    # Create and fit ARIMAX model
    arimax_model = ARIMAXModel()
    arimax_model.build_model(
        time_series,
        exog_variables=exog,
        ar_order=1,
        differencing=1,
        ma_order=1
    )
    arimax_model.sample(draws=1000, tune=1000)
    
    # Plot exogenous coefficients
    arimax_model.plot_exog_coefficients()
    
    # Generate forecast with future exogenous variables
    future_exog = pd.DataFrame({
        'Monthly': np.sin(2 * np.pi * (np.arange(30) + n_samples) / 30),
        'Random': np.random.normal(0, 1, 30)
    })
    
    forecast = arimax_model.forecast(
        horizon=30,
        future_exog=future_exog
    )
    
    # Plot forecast
    arimax_model.plot_forecast()
    
    plt.show()
