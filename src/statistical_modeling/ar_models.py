"""
Autoregressive Models Module for the Cosmic Market Oracle.

This module implements Bayesian autoregressive (AR) models for time series analysis,
providing specialized models for analyzing financial and astrological time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Callable
import matplotlib.pyplot as plt
# import logging # Removed
from src.utils.logger import get_logger # Added
import pymc as pm
import arviz as az
import theano.tensor as tt
from scipy import stats

from src.statistical_modeling.time_series_bayesian import TimeSeriesModel

# Configure logging
# logging.basicConfig( # Removed
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = get_logger(__name__) # Changed


class ARModel(TimeSeriesModel):
    """
    Bayesian Autoregressive (AR) model.
    
    This class implements a Bayesian AR(p) model for time series analysis.
    """
    
    def __init__(self, name: str = "ar_model"):
        """
        Initialize an AR model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.order = None
        self.ar_coefficients = None
    
    def build_model(self, time_series: Union[np.ndarray, pd.Series], 
                  order: int = 1,
                  differencing: int = 0,
                  standardize: bool = True,
                  priors: Optional[Dict] = None) -> pm.Model:
        """
        Build a Bayesian AR(p) model.
        
        Args:
            time_series: Time series data
            order: Order of the AR model (p)
            differencing: Order of differencing
            standardize: Whether to standardize the data
            priors: Dictionary of prior distributions for parameters
            
        Returns:
            PyMC model
        """
        # Preprocess data
        data, time_index = self.preprocess_data(time_series, differencing)
        
        # Store order
        self.order = order
        
        # Standardize data if requested
        if standardize:
            data_mean = np.mean(data)
            data_std = np.std(data)
            data = (data - data_mean) / data_std
        else:
            data_mean = 0
            data_std = 1
        
        # Set default priors if not provided
        if priors is None:
            priors = {
                'intercept': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 5}},
                'ar_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 0.5}},
                'sigma': {'dist': 'HalfNormal', 'params': {'sigma': 1}}
            }
        
        # Create model
        self.model = pm.Model()
        
        with self.model:
            # Priors for intercept
            intercept_prior = getattr(pm, priors['intercept']['dist'])
            intercept = intercept_prior(
                'intercept', **priors['intercept']['params']
            )
            
            # Priors for AR coefficients
            ar_prior = getattr(pm, priors['ar_coefficients']['dist'])
            ar_coefficients = ar_prior(
                'ar_coefficients', 
                **priors['ar_coefficients']['params'],
                shape=order
            )
            
            # Prior for error term
            sigma_prior = getattr(pm, priors['sigma']['dist'])
            sigma = sigma_prior('sigma', **priors['sigma']['params'])
            
            # Set up AR model
            mu = pm.Deterministic('mu', tt.zeros(len(data)))
            
            # Initialize with intercept
            mu = tt.set_subtensor(mu[:order], intercept)
            
            # Compute AR predictions
            for t in range(order, len(data)):
                pred = intercept
                for i in range(order):
                    pred = pred + ar_coefficients[i] * data[t-i-1]
                mu = tt.set_subtensor(mu[t], pred)
            
            # Likelihood
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
            
            # Set up forecast
            steps_ahead = 10  # Default forecast horizon
            forecast = pm.Deterministic('forecast', tt.zeros(steps_ahead))
            
            # Initialize forecast with last values
            last_values = data[-order:]
            
            # Compute forecast
            for t in range(steps_ahead):
                pred = intercept
                for i in range(order):
                    if t-i-1 < 0:
                        # Use historical data
                        pred = pred + ar_coefficients[i] * last_values[order+t-i-1]
                    else:
                        # Use forecasted values
                        pred = pred + ar_coefficients[i] * forecast[t-i-1]
                forecast = tt.set_subtensor(forecast[t], pred)
        
        # Store standardization parameters for later use
        self.data_mean = data_mean
        self.data_std = data_std
        
        return self.model
    
    def forecast(self, 
               horizon: int, 
               return_samples: bool = False,
               num_samples: int = 1000,
               include_uncertainty: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate forecasts for future time steps.
        
        Args:
            horizon: Number of time steps to forecast
            return_samples: Whether to return posterior samples
            num_samples: Number of posterior samples to generate
            include_uncertainty: Whether to include process uncertainty
            
        Returns:
            Dictionary with forecast results
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been built or sampled yet.")
        
        # Store forecast horizon
        self.forecast_horizon = horizon
        
        # Extract model parameters
        if isinstance(self.trace, az.InferenceData):
            intercept = self.trace.posterior['intercept'].values.flatten()
            ar_coefficients = self.trace.posterior['ar_coefficients'].values
            sigma = self.trace.posterior['sigma'].values.flatten()
        else:
            intercept = self.trace['intercept']
            ar_coefficients = self.trace['ar_coefficients']
            sigma = self.trace['sigma']
        
        # Reshape AR coefficients
        ar_coefficients = np.reshape(ar_coefficients, (-1, self.order))
        
        # Get last values from the time series
        last_values = self.time_series[-self.order:]
        
        # Generate forecasts
        n_samples = len(intercept)
        forecasts = np.zeros((n_samples, horizon))
        
        for i in range(n_samples):
            forecast_values = np.zeros(horizon)
            
            for t in range(horizon):
                # Compute prediction
                pred = intercept[i]
                for j in range(self.order):
                    if t-j-1 < 0:
                        # Use historical data
                        pred += ar_coefficients[i, j] * last_values[self.order+t-j-1]
                    else:
                        # Use forecasted values
                        pred += ar_coefficients[i, j] * forecast_values[t-j-1]
                
                # Add process uncertainty if requested
                if include_uncertainty:
                    pred += np.random.normal(0, sigma[i])
                
                forecast_values[t] = pred
            
            forecasts[i] = forecast_values
        
        # Unstandardize forecasts if data was standardized
        if hasattr(self, 'data_mean') and hasattr(self, 'data_std'):
            forecasts = forecasts * self.data_std + self.data_mean
        
        # Compute forecast statistics
        forecast_mean = np.mean(forecasts, axis=0)
        forecast_std = np.std(forecasts, axis=0)
        forecast_lower = np.percentile(forecasts, 2.5, axis=0)
        forecast_upper = np.percentile(forecasts, 97.5, axis=0)
        
        # Store forecast results
        self.forecast_results = {
            'mean': forecast_mean,
            'std': forecast_std,
            'lower': forecast_lower,
            'upper': forecast_upper
        }
        
        if return_samples:
            self.forecast_results['samples'] = forecasts
        
        return self.forecast_results
    
    def plot_ar_coefficients(self, 
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot posterior distributions of AR coefficients.
        
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
            ar_coefficients = self.trace.posterior['ar_coefficients'].values
        else:
            ar_coefficients = self.trace['ar_coefficients']
        
        # Reshape coefficients
        ar_coefficients = np.reshape(ar_coefficients, (-1, self.order))
        
        # Compute mean and credible intervals
        coef_mean = np.mean(ar_coefficients, axis=0)
        coef_hdi = az.hdi(ar_coefficients)
        
        # Create labels
        labels = [f"AR({i+1})" for i in range(self.order)]
        
        # Plot coefficients
        ax.errorbar(
            x=coef_mean,
            y=labels,
            xerr=np.abs(coef_hdi - coef_mean[:, np.newaxis]).T,
            fmt='o',
            capsize=5
        )
        
        # Add vertical line at zero
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add labels
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Lag')
        ax.set_title('Posterior Distributions of AR Coefficients')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def plot_forecast_components(self, 
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot the contribution of each AR component to the forecast.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.forecast_results is None:
            raise ValueError("No forecast results available. Call forecast first.")
        
        # Extract model parameters
        if isinstance(self.trace, az.InferenceData):
            intercept = self.trace.posterior['intercept'].mean(dim=('chain', 'draw')).values
            ar_coefficients = self.trace.posterior['ar_coefficients'].mean(dim=('chain', 'draw')).values
        else:
            intercept = np.mean(self.trace['intercept'])
            ar_coefficients = np.mean(self.trace['ar_coefficients'], axis=0)
        
        # Get last values from the time series
        last_values = self.time_series[-self.order:]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Initialize arrays
        components = np.zeros((self.order + 1, self.forecast_horizon))
        components[0] = intercept  # Intercept component
        
        # Compute component contributions
        for t in range(self.forecast_horizon):
            for i in range(self.order):
                if t-i-1 < 0:
                    # Use historical data
                    components[i+1, t] = ar_coefficients[i] * last_values[self.order+t-i-1]
                else:
                    # Use forecasted values
                    components[i+1, t] = ar_coefficients[i] * self.forecast_results['mean'][t-i-1]
        
        # Create x-axis values
        if self.time_index is not None and isinstance(self.time_index, pd.DatetimeIndex):
            # Use datetime index if available
            last_date = self.time_index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=self.forecast_horizon,
                freq=pd.infer_freq(self.time_index)
            )
            x = forecast_dates
        else:
            # Use integer index
            last_idx = self.n_timesteps - 1
            x = np.arange(last_idx + 1, last_idx + 1 + self.forecast_horizon)
        
        # Plot stacked components
        labels = ['Intercept'] + [f"AR({i+1})" for i in range(self.order)]
        ax.stackplot(x, components, labels=labels, alpha=0.7)
        
        # Plot total forecast
        ax.plot(x, self.forecast_results['mean'], 'k-', linewidth=2, label='Total Forecast')
        
        # Add labels and legend
        ax.set_xlabel('Time')
        ax.set_ylabel('Component Contribution')
        ax.set_title('Forecast Components')
        ax.legend(loc='upper left')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


class SeasonalARModel(ARModel):
    """
    Bayesian Seasonal Autoregressive (SAR) model.
    
    This class implements a Bayesian Seasonal AR model for time series analysis.
    """
    
    def __init__(self, name: str = "seasonal_ar_model"):
        """
        Initialize a Seasonal AR model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.seasonal_period = None
        self.seasonal_order = None
    
    def build_model(self, time_series: Union[np.ndarray, pd.Series], 
                  order: int = 1,
                  seasonal_order: int = 1,
                  seasonal_period: int = 12,
                  differencing: int = 0,
                  seasonal_differencing: int = 0,
                  standardize: bool = True,
                  priors: Optional[Dict] = None) -> pm.Model:
        """
        Build a Bayesian Seasonal AR model.
        
        Args:
            time_series: Time series data
            order: Order of the AR model (p)
            seasonal_order: Order of the seasonal AR model (P)
            seasonal_period: Seasonal period (s)
            differencing: Order of differencing
            seasonal_differencing: Order of seasonal differencing
            standardize: Whether to standardize the data
            priors: Dictionary of prior distributions for parameters
            
        Returns:
            PyMC model
        """
        # Store seasonal parameters
        self.seasonal_period = seasonal_period
        self.seasonal_order = seasonal_order
        
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
        
        # Preprocess data
        data, time_index = self.preprocess_data(time_series, differencing)
        
        # Store order
        self.order = order
        
        # Standardize data if requested
        if standardize:
            data_mean = np.mean(data)
            data_std = np.std(data)
            data = (data - data_mean) / data_std
        else:
            data_mean = 0
            data_std = 1
        
        # Set default priors if not provided
        if priors is None:
            priors = {
                'intercept': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 5}},
                'ar_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 0.5}},
                'seasonal_ar_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 0.5}},
                'sigma': {'dist': 'HalfNormal', 'params': {'sigma': 1}}
            }
        
        # Create model
        self.model = pm.Model()
        
        with self.model:
            # Priors for intercept
            intercept_prior = getattr(pm, priors['intercept']['dist'])
            intercept = intercept_prior(
                'intercept', **priors['intercept']['params']
            )
            
            # Priors for AR coefficients
            ar_prior = getattr(pm, priors['ar_coefficients']['dist'])
            ar_coefficients = ar_prior(
                'ar_coefficients', 
                **priors['ar_coefficients']['params'],
                shape=order
            )
            
            # Priors for seasonal AR coefficients
            seasonal_ar_prior = getattr(pm, priors['seasonal_ar_coefficients']['dist'])
            seasonal_ar_coefficients = seasonal_ar_prior(
                'seasonal_ar_coefficients', 
                **priors['seasonal_ar_coefficients']['params'],
                shape=seasonal_order
            )
            
            # Prior for error term
            sigma_prior = getattr(pm, priors['sigma']['dist'])
            sigma = sigma_prior('sigma', **priors['sigma']['params'])
            
            # Set up AR model
            mu = pm.Deterministic('mu', tt.zeros(len(data)))
            
            # Initialize with intercept
            max_lag = max(order, seasonal_order * seasonal_period)
            mu = tt.set_subtensor(mu[:max_lag], intercept)
            
            # Compute AR predictions
            for t in range(max_lag, len(data)):
                pred = intercept
                
                # Add non-seasonal AR terms
                for i in range(order):
                    pred = pred + ar_coefficients[i] * data[t-i-1]
                
                # Add seasonal AR terms
                for i in range(seasonal_order):
                    lag = (i + 1) * seasonal_period
                    if t >= lag:
                        pred = pred + seasonal_ar_coefficients[i] * data[t-lag]
                
                mu = tt.set_subtensor(mu[t], pred)
            
            # Likelihood
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
            
            # Set up forecast
            steps_ahead = 10  # Default forecast horizon
            forecast = pm.Deterministic('forecast', tt.zeros(steps_ahead))
            
            # Compute forecast
            for t in range(steps_ahead):
                pred = intercept
                
                # Add non-seasonal AR terms
                for i in range(order):
                    if t-i-1 < 0:
                        # Use historical data
                        pred = pred + ar_coefficients[i] * data[len(data)+t-i-1]
                    else:
                        # Use forecasted values
                        pred = pred + ar_coefficients[i] * forecast[t-i-1]
                
                # Add seasonal AR terms
                for i in range(seasonal_order):
                    lag = (i + 1) * seasonal_period
                    if t >= lag:
                        # Use forecasted values
                        pred = pred + seasonal_ar_coefficients[i] * forecast[t-lag]
                    elif len(data) + t >= lag:
                        # Use historical data
                        pred = pred + seasonal_ar_coefficients[i] * data[len(data)+t-lag]
                
                forecast = tt.set_subtensor(forecast[t], pred)
        
        # Store standardization parameters for later use
        self.data_mean = data_mean
        self.data_std = data_std
        
        return self.model
    
    def plot_seasonal_coefficients(self, 
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot posterior distributions of seasonal AR coefficients.
        
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
            seasonal_ar_coefficients = self.trace.posterior['seasonal_ar_coefficients'].values
        else:
            seasonal_ar_coefficients = self.trace['seasonal_ar_coefficients']
        
        # Reshape coefficients
        seasonal_ar_coefficients = np.reshape(seasonal_ar_coefficients, (-1, self.seasonal_order))
        
        # Compute mean and credible intervals
        coef_mean = np.mean(seasonal_ar_coefficients, axis=0)
        coef_hdi = az.hdi(seasonal_ar_coefficients)
        
        # Create labels
        labels = [f"SAR({i+1}×{self.seasonal_period})" for i in range(self.seasonal_order)]
        
        # Plot coefficients
        ax.errorbar(
            x=coef_mean,
            y=labels,
            xerr=np.abs(coef_hdi - coef_mean[:, np.newaxis]).T,
            fmt='o',
            capsize=5
        )
        
        # Add vertical line at zero
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add labels
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Seasonal Lag')
        ax.set_title('Posterior Distributions of Seasonal AR Coefficients')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Generate sample data with seasonal pattern
    np.random.seed(42)
    n_samples = 200
    
    # Create time index
    time_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate AR(1) process
    ar_coef = 0.7
    sigma = 0.5
    
    # Initialize time series
    ts = np.zeros(n_samples)
    ts[0] = np.random.normal(0, 1)
    
    # Generate AR(1) process
    for t in range(1, n_samples):
        ts[t] = ar_coef * ts[t-1] + np.random.normal(0, sigma)
    
    # Add seasonal component
    seasonal_period = 7  # Weekly seasonality
    seasonal_pattern = 2 * np.sin(2 * np.pi * np.arange(n_samples) / seasonal_period)
    ts = ts + seasonal_pattern
    
    # Create pandas Series
    time_series = pd.Series(ts, index=time_index)
    
    # Create and fit AR model
    ar_model = ARModel()
    ar_model.build_model(time_series, order=2)
    ar_model.sample(draws=1000, tune=1000)
    
    # Plot diagnostics
    ar_model.plot_diagnostics()
    
    # Plot AR coefficients
    ar_model.plot_ar_coefficients()
    
    # Generate forecast
    forecast = ar_model.forecast(horizon=30)
    
    # Plot forecast
    ar_model.plot_forecast()
    
    # Create and fit Seasonal AR model
    sar_model = SeasonalARModel()
    sar_model.build_model(
        time_series, 
        order=1, 
        seasonal_order=1, 
        seasonal_period=7
    )
    sar_model.sample(draws=1000, tune=1000)
    
    # Plot seasonal coefficients
    sar_model.plot_seasonal_coefficients()
    
    # Generate forecast
    forecast = sar_model.forecast(horizon=30)
    
    # Plot forecast
    sar_model.plot_forecast()
    
    plt.show()
