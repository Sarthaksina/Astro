"""
Seasonal Models Module for the Cosmic Market Oracle.

This module implements Bayesian seasonal models for time series analysis,
providing specialized models for analyzing seasonal patterns in financial 
and astrological time series data.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SeasonalModel(TimeSeriesModel):
    """
    Base class for Bayesian seasonal models.
    
    This class extends the TimeSeriesModel class to provide functionality
    for building and analyzing seasonal time series models.
    """
    
    def __init__(self, name: str = "seasonal_model"):
        """
        Initialize a seasonal model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.seasonal_periods = None
        self.seasonal_components = None
    
    def build_model(self, *args, **kwargs) -> pm.Model:
        """
        Build the seasonal model.
        
        This is a placeholder method that should be overridden by subclasses.
        
        Returns:
            PyMC model
        """
        raise NotImplementedError("Subclasses must implement build_model method")
    
    def extract_components(self) -> Dict[str, np.ndarray]:
        """
        Extract components from the model.
        
        This is a placeholder method that should be overridden by subclasses.
        
        Returns:
            Dictionary with components
        """
        raise NotImplementedError("Subclasses must implement extract_components method")
    
    def plot_components(self, 
                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot time series components (trend, seasonal, residual).
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Extract components
        components = self.extract_components()
        
        # Create figure with subplots
        n_components = len(components)
        fig, axes = plt.subplots(n_components, 1, figsize=figsize)
        
        # If only one component, convert axes to list
        if n_components == 1:
            axes = [axes]
        
        # Plot each component
        for i, (component_name, component_values) in enumerate(components.items()):
            # Create x-axis values
            if self.time_index is not None and isinstance(self.time_index, pd.DatetimeIndex):
                x = self.time_index
            else:
                x = np.arange(self.n_timesteps)
            
            # Plot component
            axes[i].plot(x, component_values)
            
            # Add labels
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Value')
            axes[i].set_title(f'{component_name.capitalize()} Component')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def forecast_components(self, 
                          horizon: int) -> Dict[str, np.ndarray]:
        """
        Forecast components for future time steps.
        
        This is a placeholder method that should be overridden by subclasses.
        
        Args:
            horizon: Number of time steps to forecast
            
        Returns:
            Dictionary with component forecasts
        """
        raise NotImplementedError("Subclasses must implement forecast_components method")
    
    def plot_component_forecasts(self, 
                               horizon: int,
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot forecasts of time series components.
        
        Args:
            horizon: Number of time steps to forecast
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Forecast components
        component_forecasts = self.forecast_components(horizon)
        
        # Create figure with subplots
        n_components = len(component_forecasts)
        fig, axes = plt.subplots(n_components, 1, figsize=figsize)
        
        # If only one component, convert axes to list
        if n_components == 1:
            axes = [axes]
        
        # Extract historical components
        historical_components = self.extract_components()
        
        # Plot each component
        for i, (component_name, forecast_values) in enumerate(component_forecasts.items()):
            # Get historical values
            historical_values = historical_components.get(component_name, None)
            
            # Create x-axis values for historical data
            if self.time_index is not None and isinstance(self.time_index, pd.DatetimeIndex):
                x_hist = self.time_index
                
                # Create x-axis values for forecast
                last_date = self.time_index[-1]
                freq = pd.infer_freq(self.time_index)
                x_forecast = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=horizon,
                    freq=freq
                )
            else:
                x_hist = np.arange(self.n_timesteps)
                x_forecast = np.arange(self.n_timesteps, self.n_timesteps + horizon)
            
            # Plot historical component
            if historical_values is not None:
                axes[i].plot(x_hist, historical_values, 'b-', label='Historical')
            
            # Plot forecast component
            axes[i].plot(x_forecast, forecast_values, 'r-', label='Forecast')
            
            # Add labels
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Value')
            axes[i].set_title(f'{component_name.capitalize()} Component Forecast')
            axes[i].legend()
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


class SeasonalDecompositionModel(SeasonalModel):
    """
    Bayesian Seasonal Decomposition model.
    
    This class implements a Bayesian model for seasonal decomposition,
    separating a time series into trend, seasonal, and residual components.
    """
    
    def __init__(self, name: str = "seasonal_decomposition_model"):
        """
        Initialize a seasonal decomposition model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.trend_order = None
        self.seasonal_periods = None
        self.seasonal_harmonics = None
    
    def build_model(self, time_series: Union[np.ndarray, pd.Series],
                  trend_order: int = 1,
                  seasonal_periods: List[int] = [12],
                  seasonal_harmonics: Optional[Dict[int, int]] = None,
                  standardize: bool = True,
                  priors: Optional[Dict] = None) -> pm.Model:
        """
        Build a Bayesian Seasonal Decomposition model.
        
        Args:
            time_series: Time series data
            trend_order: Order of the polynomial trend
            seasonal_periods: List of seasonal periods
            seasonal_harmonics: Dictionary mapping seasonal periods to number of harmonics
            standardize: Whether to standardize the data
            priors: Dictionary of prior distributions for parameters
            
        Returns:
            PyMC model
        """
        # Preprocess data
        data, time_index = self.preprocess_data(time_series)
        
        # Store parameters
        self.trend_order = trend_order
        self.seasonal_periods = seasonal_periods
        
        # Set default seasonal harmonics if not provided
        if seasonal_harmonics is None:
            self.seasonal_harmonics = {period: period // 2 for period in seasonal_periods}
        else:
            self.seasonal_harmonics = seasonal_harmonics
        
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
                'trend_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 1}},
                'seasonal_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 0.5}},
                'sigma': {'dist': 'HalfNormal', 'params': {'sigma': 1}}
            }
        
        # Create model
        self.model = pm.Model()
        
        with self.model:
            # Priors for trend coefficients
            trend_prior = getattr(pm, priors['trend_coefficients']['dist'])
            trend_coefficients = trend_prior(
                'trend_coefficients', 
                **priors['trend_coefficients']['params'],
                shape=trend_order + 1
            )
            
            # Priors for seasonal coefficients
            seasonal_prior = getattr(pm, priors['seasonal_coefficients']['dist'])
            
            # Create dictionary to store seasonal coefficients
            seasonal_coefficients = {}
            
            for period in seasonal_periods:
                n_harmonics = self.seasonal_harmonics[period]
                
                # Sine and cosine coefficients for each harmonic
                seasonal_coefficients[period] = {}
                
                for h in range(1, n_harmonics + 1):
                    # Sine coefficient
                    seasonal_coefficients[period][f'sin_{h}'] = seasonal_prior(
                        f'seasonal_sin_coef_{period}_{h}',
                        **priors['seasonal_coefficients']['params']
                    )
                    
                    # Cosine coefficient
                    seasonal_coefficients[period][f'cos_{h}'] = seasonal_prior(
                        f'seasonal_cos_coef_{period}_{h}',
                        **priors['seasonal_coefficients']['params']
                    )
            
            # Prior for error term
            sigma_prior = getattr(pm, priors['sigma']['dist'])
            sigma = sigma_prior('sigma', **priors['sigma']['params'])
            
            # Create time index for model
            t = np.arange(len(data)) / len(data)
            
            # Compute trend component
            trend = pm.Deterministic('trend', tt.zeros(len(data)))
            
            for i in range(trend_order + 1):
                trend = trend + trend_coefficients[i] * t**i
            
            # Compute seasonal components
            seasonal = {}
            
            for period in seasonal_periods:
                seasonal[period] = pm.Deterministic(f'seasonal_{period}', tt.zeros(len(data)))
                n_harmonics = self.seasonal_harmonics[period]
                
                for h in range(1, n_harmonics + 1):
                    # Compute sine and cosine terms
                    sin_term = tt.sin(2 * np.pi * h * t * len(data) / period)
                    cos_term = tt.cos(2 * np.pi * h * t * len(data) / period)
                    
                    # Add harmonic to seasonal component
                    seasonal[period] = seasonal[period] + (
                        seasonal_coefficients[period][f'sin_{h}'] * sin_term +
                        seasonal_coefficients[period][f'cos_{h}'] * cos_term
                    )
            
            # Combine seasonal components
            total_seasonal = pm.Deterministic('total_seasonal', tt.zeros(len(data)))
            
            for period in seasonal_periods:
                total_seasonal = total_seasonal + seasonal[period]
            
            # Compute model prediction
            mu = pm.Deterministic('mu', trend + total_seasonal)
            
            # Likelihood
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
            
            # Set up forecast
            steps_ahead = 10  # Default forecast horizon
            forecast_t = (np.arange(len(data), len(data) + steps_ahead) / len(data))
            
            # Forecast trend
            trend_forecast = pm.Deterministic('trend_forecast', tt.zeros(steps_ahead))
            
            for i in range(trend_order + 1):
                trend_forecast = trend_forecast + trend_coefficients[i] * forecast_t**i
            
            # Forecast seasonal components
            seasonal_forecast = {}
            
            for period in seasonal_periods:
                seasonal_forecast[period] = pm.Deterministic(
                    f'seasonal_forecast_{period}', 
                    tt.zeros(steps_ahead)
                )
                n_harmonics = self.seasonal_harmonics[period]
                
                for h in range(1, n_harmonics + 1):
                    # Compute sine and cosine terms for forecast
                    sin_term = tt.sin(2 * np.pi * h * forecast_t * len(data) / period)
                    cos_term = tt.cos(2 * np.pi * h * forecast_t * len(data) / period)
                    
                    # Add harmonic to seasonal forecast
                    seasonal_forecast[period] = seasonal_forecast[period] + (
                        seasonal_coefficients[period][f'sin_{h}'] * sin_term +
                        seasonal_coefficients[period][f'cos_{h}'] * cos_term
                    )
            
            # Combine seasonal forecasts
            total_seasonal_forecast = pm.Deterministic(
                'total_seasonal_forecast', 
                tt.zeros(steps_ahead)
            )
            
            for period in seasonal_periods:
                total_seasonal_forecast = total_seasonal_forecast + seasonal_forecast[period]
            
            # Compute forecast
            forecast = pm.Deterministic(
                'forecast', 
                trend_forecast + total_seasonal_forecast
            )
        
        # Store standardization parameters for later use
        self.data_mean = data_mean
        self.data_std = data_std
        
        return self.model
    
    def extract_components(self) -> Dict[str, np.ndarray]:
        """
        Extract components from the model.
        
        Returns:
            Dictionary with components
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Initialize dictionary for components
        components = {}
        
        # Extract trend component
        if isinstance(self.trace, az.InferenceData):
            trend = self.trace.posterior['trend'].mean(dim=('chain', 'draw')).values
        else:
            trend = np.mean(self.trace['trend'], axis=0)
        
        components['trend'] = trend
        
        # Extract seasonal components
        for period in self.seasonal_periods:
            if isinstance(self.trace, az.InferenceData):
                seasonal = self.trace.posterior[f'seasonal_{period}'].mean(dim=('chain', 'draw')).values
            else:
                seasonal = np.mean(self.trace[f'seasonal_{period}'], axis=0)
            
            components[f'seasonal_{period}'] = seasonal
        
        # Extract total seasonal component
        if isinstance(self.trace, az.InferenceData):
            total_seasonal = self.trace.posterior['total_seasonal'].mean(dim=('chain', 'draw')).values
        else:
            total_seasonal = np.mean(self.trace['total_seasonal'], axis=0)
        
        components['seasonal'] = total_seasonal
        
        # Compute residual component
        if isinstance(self.trace, az.InferenceData):
            mu = self.trace.posterior['mu'].mean(dim=('chain', 'draw')).values
        else:
            mu = np.mean(self.trace['mu'], axis=0)
        
        residual = self.time_series - mu
        components['residual'] = residual
        
        # Unstandardize components if data was standardized
        if hasattr(self, 'data_mean') and hasattr(self, 'data_std'):
            for component in components:
                if component != 'residual':
                    components[component] = components[component] * self.data_std
            
            components['trend'] = components['trend'] + self.data_mean
        
        return components
    
    def forecast_components(self, 
                          horizon: int) -> Dict[str, np.ndarray]:
        """
        Forecast components for future time steps.
        
        Args:
            horizon: Number of time steps to forecast
            
        Returns:
            Dictionary with component forecasts
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Initialize dictionary for component forecasts
        component_forecasts = {}
        
        # Extract trend forecast
        if isinstance(self.trace, az.InferenceData):
            trend_forecast = self.trace.posterior['trend_forecast'].mean(dim=('chain', 'draw')).values
        else:
            trend_forecast = np.mean(self.trace['trend_forecast'], axis=0)
        
        # Truncate or extend trend forecast to match horizon
        if len(trend_forecast) < horizon:
            # Extend trend forecast using extrapolation
            t = np.arange(len(self.time_series), len(self.time_series) + horizon) / len(self.time_series)
            
            # Extract trend coefficients
            if isinstance(self.trace, az.InferenceData):
                trend_coefficients = self.trace.posterior['trend_coefficients'].mean(dim=('chain', 'draw')).values
            else:
                trend_coefficients = np.mean(self.trace['trend_coefficients'], axis=0)
            
            # Compute extended trend forecast
            extended_trend = np.zeros(horizon)
            
            for i in range(len(trend_coefficients)):
                extended_trend += trend_coefficients[i] * t**i
            
            trend_forecast = extended_trend
        elif len(trend_forecast) > horizon:
            trend_forecast = trend_forecast[:horizon]
        
        component_forecasts['trend'] = trend_forecast
        
        # Extract seasonal forecasts
        for period in self.seasonal_periods:
            if isinstance(self.trace, az.InferenceData):
                seasonal_forecast = self.trace.posterior[f'seasonal_forecast_{period}'].mean(dim=('chain', 'draw')).values
            else:
                seasonal_forecast = np.mean(self.trace[f'seasonal_forecast_{period}'], axis=0)
            
            # Truncate or extend seasonal forecast to match horizon
            if len(seasonal_forecast) < horizon:
                # Extend seasonal forecast using periodicity
                n_repeats = int(np.ceil(horizon / period))
                extended_seasonal = np.tile(seasonal_forecast[:period], n_repeats)[:horizon]
                seasonal_forecast = extended_seasonal
            elif len(seasonal_forecast) > horizon:
                seasonal_forecast = seasonal_forecast[:horizon]
            
            component_forecasts[f'seasonal_{period}'] = seasonal_forecast
        
        # Extract total seasonal forecast
        if isinstance(self.trace, az.InferenceData):
            total_seasonal_forecast = self.trace.posterior['total_seasonal_forecast'].mean(dim=('chain', 'draw')).values
        else:
            total_seasonal_forecast = np.mean(self.trace['total_seasonal_forecast'], axis=0)
        
        # Truncate or extend total seasonal forecast to match horizon
        if len(total_seasonal_forecast) < horizon:
            # Compute total seasonal forecast from individual components
            total_seasonal_forecast = np.zeros(horizon)
            
            for period in self.seasonal_periods:
                total_seasonal_forecast += component_forecasts[f'seasonal_{period}']
        elif len(total_seasonal_forecast) > horizon:
            total_seasonal_forecast = total_seasonal_forecast[:horizon]
        
        component_forecasts['seasonal'] = total_seasonal_forecast
        
        # Compute total forecast
        component_forecasts['total'] = component_forecasts['trend'] + component_forecasts['seasonal']
        
        # Unstandardize component forecasts if data was standardized
        if hasattr(self, 'data_mean') and hasattr(self, 'data_std'):
            for component in component_forecasts:
                if component != 'residual':
                    component_forecasts[component] = component_forecasts[component] * self.data_std
            
            component_forecasts['trend'] = component_forecasts['trend'] + self.data_mean
            component_forecasts['total'] = component_forecasts['total'] + self.data_mean
        
        return component_forecasts
    
    def plot_seasonal_harmonics(self, 
                              period: int,
                              figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot seasonal harmonics for a specific period.
        
        Args:
            period: Seasonal period to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        if period not in self.seasonal_periods:
            raise ValueError(f"Period {period} not found in seasonal periods.")
        
        # Get number of harmonics for this period
        n_harmonics = self.seasonal_harmonics[period]
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_harmonics, 1, figsize=figsize)
        
        # If only one harmonic, convert axes to list
        if n_harmonics == 1:
            axes = [axes]
        
        # Create time index for one period
        t = np.linspace(0, 1, period)
        
        # Extract harmonic coefficients
        harmonic_coefficients = {}
        
        for h in range(1, n_harmonics + 1):
            if isinstance(self.trace, az.InferenceData):
                sin_coef = self.trace.posterior[f'seasonal_sin_coef_{period}_{h}'].mean(dim=('chain', 'draw')).values
                cos_coef = self.trace.posterior[f'seasonal_cos_coef_{period}_{h}'].mean(dim=('chain', 'draw')).values
            else:
                sin_coef = np.mean(self.trace[f'seasonal_sin_coef_{period}_{h}'])
                cos_coef = np.mean(self.trace[f'seasonal_cos_coef_{period}_{h}'])
            
            harmonic_coefficients[h] = {'sin': sin_coef, 'cos': cos_coef}
        
        # Plot each harmonic
        for h in range(1, n_harmonics + 1):
            # Compute harmonic
            sin_term = harmonic_coefficients[h]['sin'] * np.sin(2 * np.pi * h * t)
            cos_term = harmonic_coefficients[h]['cos'] * np.cos(2 * np.pi * h * t)
            harmonic = sin_term + cos_term
            
            # Plot harmonic
            axes[h-1].plot(t, harmonic)
            
            # Add labels
            axes[h-1].set_xlabel('Cycle Phase')
            axes[h-1].set_ylabel('Value')
            axes[h-1].set_title(f'Harmonic {h} (Period {period})')
            
            # Add vertical lines at cycle boundaries
            axes[h-1].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            axes[h-1].axvline(x=1, color='gray', linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Generate sample data with trend and multiple seasonal patterns
    np.random.seed(42)
    n_samples = 365 * 3  # 3 years of daily data
    
    # Create time index
    time_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate trend
    trend = 0.01 * np.arange(n_samples) + 10
    
    # Generate weekly seasonality
    weekly_period = 7
    weekly_seasonality = 3 * np.sin(2 * np.pi * np.arange(n_samples) / weekly_period)
    
    # Generate monthly seasonality
    monthly_period = 30
    monthly_seasonality = 5 * np.sin(2 * np.pi * np.arange(n_samples) / monthly_period)
    
    # Generate annual seasonality
    annual_period = 365
    annual_seasonality = 10 * np.sin(2 * np.pi * np.arange(n_samples) / annual_period)
    
    # Add noise
    noise = np.random.normal(0, 1, n_samples)
    
    # Combine components
    ts = trend + weekly_seasonality + monthly_seasonality + annual_seasonality + noise
    
    # Create pandas Series
    time_series = pd.Series(ts, index=time_index)
    
    # Create and fit seasonal decomposition model
    model = SeasonalDecompositionModel()
    model.build_model(
        time_series,
        trend_order=2,
        seasonal_periods=[7, 30, 365],
        seasonal_harmonics={7: 3, 30: 5, 365: 10}
    )
    model.sample(draws=1000, tune=1000)
    
    # Extract components
    components = model.extract_components()
    
    # Plot components
    model.plot_components()
    
    # Plot seasonal harmonics
    model.plot_seasonal_harmonics(period=7)
    model.plot_seasonal_harmonics(period=30)
    model.plot_seasonal_harmonics(period=365)
    
    # Forecast components
    forecast_horizon = 90  # 3 months
    component_forecasts = model.forecast_components(forecast_horizon)
    
    # Plot component forecasts
    model.plot_component_forecasts(forecast_horizon)
    
    # Generate forecast
    forecast = model.forecast(horizon=forecast_horizon)
    
    # Plot forecast
    model.plot_forecast()
    
    plt.show()
