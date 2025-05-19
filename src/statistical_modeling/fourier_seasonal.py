"""
Fourier Seasonal Models Module for the Cosmic Market Oracle.

This module implements Bayesian Fourier-based seasonal models for time series analysis,
providing specialized models for analyzing complex seasonal patterns in financial 
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

from src.statistical_modeling.seasonal_models import SeasonalModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FourierSeasonalModel(SeasonalModel):
    """
    Bayesian Fourier Seasonal model.
    
    This class implements a Bayesian model using Fourier terms to model
    complex seasonal patterns in time series data.
    """
    
    def __init__(self, name: str = "fourier_seasonal_model"):
        """
        Initialize a Fourier seasonal model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.trend_order = None
        self.seasonal_periods = None
        self.fourier_orders = None
        self.regression_terms = None
    
    def build_model(self, time_series: Union[np.ndarray, pd.Series],
                  trend_order: int = 1,
                  seasonal_periods: List[int] = [12],
                  fourier_orders: Optional[Dict[int, int]] = None,
                  regression_terms: Optional[Dict[str, np.ndarray]] = None,
                  standardize: bool = True,
                  priors: Optional[Dict] = None) -> pm.Model:
        """
        Build a Bayesian Fourier Seasonal model.
        
        Args:
            time_series: Time series data
            trend_order: Order of the polynomial trend
            seasonal_periods: List of seasonal periods
            fourier_orders: Dictionary mapping seasonal periods to Fourier orders
            regression_terms: Dictionary of additional regression terms
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
        
        # Set default Fourier orders if not provided
        if fourier_orders is None:
            self.fourier_orders = {period: min(period // 2, 10) for period in seasonal_periods}
        else:
            self.fourier_orders = fourier_orders
        
        # Store regression terms
        self.regression_terms = regression_terms
        
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
                'fourier_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 0.5}},
                'regression_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 1}},
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
            
            # Priors for Fourier coefficients
            fourier_prior = getattr(pm, priors['fourier_coefficients']['dist'])
            
            # Create dictionary to store Fourier coefficients
            fourier_coefficients = {}
            
            for period in seasonal_periods:
                fourier_order = self.fourier_orders[period]
                
                # Sine and cosine coefficients for each Fourier term
                fourier_coefficients[period] = {}
                
                for k in range(1, fourier_order + 1):
                    # Sine coefficient
                    fourier_coefficients[period][f'sin_{k}'] = fourier_prior(
                        f'fourier_sin_coef_{period}_{k}',
                        **priors['fourier_coefficients']['params']
                    )
                    
                    # Cosine coefficient
                    fourier_coefficients[period][f'cos_{k}'] = fourier_prior(
                        f'fourier_cos_coef_{period}_{k}',
                        **priors['fourier_coefficients']['params']
                    )
            
            # Priors for regression coefficients if regression terms provided
            if regression_terms is not None:
                regression_prior = getattr(pm, priors['regression_coefficients']['dist'])
                regression_coefficients = regression_prior(
                    'regression_coefficients',
                    **priors['regression_coefficients']['params'],
                    shape=len(regression_terms)
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
            
            # Compute Fourier components
            fourier = {}
            
            for period in seasonal_periods:
                fourier[period] = pm.Deterministic(f'fourier_{period}', tt.zeros(len(data)))
                fourier_order = self.fourier_orders[period]
                
                for k in range(1, fourier_order + 1):
                    # Compute sine and cosine terms
                    sin_term = tt.sin(2 * np.pi * k * t * len(data) / period)
                    cos_term = tt.cos(2 * np.pi * k * t * len(data) / period)
                    
                    # Add Fourier term to component
                    fourier[period] = fourier[period] + (
                        fourier_coefficients[period][f'sin_{k}'] * sin_term +
                        fourier_coefficients[period][f'cos_{k}'] * cos_term
                    )
            
            # Combine Fourier components
            total_fourier = pm.Deterministic('total_fourier', tt.zeros(len(data)))
            
            for period in seasonal_periods:
                total_fourier = total_fourier + fourier[period]
            
            # Compute regression component if regression terms provided
            if regression_terms is not None:
                regression = pm.Deterministic('regression', tt.zeros(len(data)))
                
                for i, (term_name, term_values) in enumerate(regression_terms.items()):
                    regression = regression + regression_coefficients[i] * term_values
            else:
                regression = pm.Deterministic('regression', tt.zeros(len(data)))
            
            # Compute model prediction
            mu = pm.Deterministic('mu', trend + total_fourier + regression)
            
            # Likelihood
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
            
            # Set up forecast
            steps_ahead = 10  # Default forecast horizon
            forecast_t = (np.arange(len(data), len(data) + steps_ahead) / len(data))
            
            # Forecast trend
            trend_forecast = pm.Deterministic('trend_forecast', tt.zeros(steps_ahead))
            
            for i in range(trend_order + 1):
                trend_forecast = trend_forecast + trend_coefficients[i] * forecast_t**i
            
            # Forecast Fourier components
            fourier_forecast = {}
            
            for period in seasonal_periods:
                fourier_forecast[period] = pm.Deterministic(
                    f'fourier_forecast_{period}', 
                    tt.zeros(steps_ahead)
                )
                fourier_order = self.fourier_orders[period]
                
                for k in range(1, fourier_order + 1):
                    # Compute sine and cosine terms for forecast
                    sin_term = tt.sin(2 * np.pi * k * forecast_t * len(data) / period)
                    cos_term = tt.cos(2 * np.pi * k * forecast_t * len(data) / period)
                    
                    # Add Fourier term to forecast
                    fourier_forecast[period] = fourier_forecast[period] + (
                        fourier_coefficients[period][f'sin_{k}'] * sin_term +
                        fourier_coefficients[period][f'cos_{k}'] * cos_term
                    )
            
            # Combine Fourier forecasts
            total_fourier_forecast = pm.Deterministic(
                'total_fourier_forecast', 
                tt.zeros(steps_ahead)
            )
            
            for period in seasonal_periods:
                total_fourier_forecast = total_fourier_forecast + fourier_forecast[period]
            
            # Forecast regression component if regression terms provided
            if regression_terms is not None:
                # This is a placeholder; in a real implementation, you would need
                # future values of regression terms for forecasting
                regression_forecast = pm.Deterministic(
                    'regression_forecast',
                    tt.zeros(steps_ahead)
                )
            else:
                regression_forecast = pm.Deterministic(
                    'regression_forecast',
                    tt.zeros(steps_ahead)
                )
            
            # Compute forecast
            forecast = pm.Deterministic(
                'forecast', 
                trend_forecast + total_fourier_forecast + regression_forecast
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
        
        # Extract Fourier components
        for period in self.seasonal_periods:
            if isinstance(self.trace, az.InferenceData):
                fourier = self.trace.posterior[f'fourier_{period}'].mean(dim=('chain', 'draw')).values
            else:
                fourier = np.mean(self.trace[f'fourier_{period}'], axis=0)
            
            components[f'fourier_{period}'] = fourier
        
        # Extract total Fourier component
        if isinstance(self.trace, az.InferenceData):
            total_fourier = self.trace.posterior['total_fourier'].mean(dim=('chain', 'draw')).values
        else:
            total_fourier = np.mean(self.trace['total_fourier'], axis=0)
        
        components['seasonal'] = total_fourier
        
        # Extract regression component if available
        if isinstance(self.trace, az.InferenceData):
            regression = self.trace.posterior['regression'].mean(dim=('chain', 'draw')).values
        else:
            regression = np.mean(self.trace['regression'], axis=0)
        
        components['regression'] = regression
        
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
        
        # Extract Fourier forecasts
        for period in self.seasonal_periods:
            if isinstance(self.trace, az.InferenceData):
                fourier_forecast = self.trace.posterior[f'fourier_forecast_{period}'].mean(dim=('chain', 'draw')).values
            else:
                fourier_forecast = np.mean(self.trace[f'fourier_forecast_{period}'], axis=0)
            
            # Truncate or extend Fourier forecast to match horizon
            if len(fourier_forecast) < horizon:
                # Extend Fourier forecast using periodicity
                t = np.arange(len(self.time_series), len(self.time_series) + horizon) / len(self.time_series)
                
                # Extract Fourier coefficients
                fourier_coefficients = {}
                fourier_order = self.fourier_orders[period]
                
                for k in range(1, fourier_order + 1):
                    if isinstance(self.trace, az.InferenceData):
                        sin_coef = self.trace.posterior[f'fourier_sin_coef_{period}_{k}'].mean(dim=('chain', 'draw')).values
                        cos_coef = self.trace.posterior[f'fourier_cos_coef_{period}_{k}'].mean(dim=('chain', 'draw')).values
                    else:
                        sin_coef = np.mean(self.trace[f'fourier_sin_coef_{period}_{k}'])
                        cos_coef = np.mean(self.trace[f'fourier_cos_coef_{period}_{k}'])
                    
                    fourier_coefficients[f'sin_{k}'] = sin_coef
                    fourier_coefficients[f'cos_{k}'] = cos_coef
                
                # Compute extended Fourier forecast
                extended_fourier = np.zeros(horizon)
                
                for k in range(1, fourier_order + 1):
                    sin_term = np.sin(2 * np.pi * k * t * len(self.time_series) / period)
                    cos_term = np.cos(2 * np.pi * k * t * len(self.time_series) / period)
                    
                    extended_fourier += (
                        fourier_coefficients[f'sin_{k}'] * sin_term +
                        fourier_coefficients[f'cos_{k}'] * cos_term
                    )
                
                fourier_forecast = extended_fourier
            elif len(fourier_forecast) > horizon:
                fourier_forecast = fourier_forecast[:horizon]
            
            component_forecasts[f'fourier_{period}'] = fourier_forecast
        
        # Extract total Fourier forecast
        if isinstance(self.trace, az.InferenceData):
            total_fourier_forecast = self.trace.posterior['total_fourier_forecast'].mean(dim=('chain', 'draw')).values
        else:
            total_fourier_forecast = np.mean(self.trace['total_fourier_forecast'], axis=0)
        
        # Truncate or extend total Fourier forecast to match horizon
        if len(total_fourier_forecast) < horizon:
            # Compute total Fourier forecast from individual components
            total_fourier_forecast = np.zeros(horizon)
            
            for period in self.seasonal_periods:
                total_fourier_forecast += component_forecasts[f'fourier_{period}']
        elif len(total_fourier_forecast) > horizon:
            total_fourier_forecast = total_fourier_forecast[:horizon]
        
        component_forecasts['seasonal'] = total_fourier_forecast
        
        # Extract regression forecast if available
        if isinstance(self.trace, az.InferenceData):
            regression_forecast = self.trace.posterior['regression_forecast'].mean(dim=('chain', 'draw')).values
        else:
            regression_forecast = np.mean(self.trace['regression_forecast'], axis=0)
        
        # Truncate or extend regression forecast to match horizon
        if len(regression_forecast) < horizon:
            # This is a placeholder; in a real implementation, you would need
            # future values of regression terms for forecasting
            regression_forecast = np.zeros(horizon)
        elif len(regression_forecast) > horizon:
            regression_forecast = regression_forecast[:horizon]
        
        component_forecasts['regression'] = regression_forecast
        
        # Compute total forecast
        component_forecasts['total'] = (
            component_forecasts['trend'] + 
            component_forecasts['seasonal'] + 
            component_forecasts['regression']
        )
        
        # Unstandardize component forecasts if data was standardized
        if hasattr(self, 'data_mean') and hasattr(self, 'data_std'):
            for component in component_forecasts:
                component_forecasts[component] = component_forecasts[component] * self.data_std
            
            component_forecasts['trend'] = component_forecasts['trend'] + self.data_mean
            component_forecasts['total'] = component_forecasts['total'] + self.data_mean
        
        return component_forecasts
    
    def plot_fourier_spectrum(self, 
                            period: int,
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot Fourier spectrum for a specific period.
        
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
        
        # Get Fourier order for this period
        fourier_order = self.fourier_orders[period]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract Fourier coefficients
        sin_coefficients = []
        cos_coefficients = []
        
        for k in range(1, fourier_order + 1):
            if isinstance(self.trace, az.InferenceData):
                sin_coef = self.trace.posterior[f'fourier_sin_coef_{period}_{k}'].mean(dim=('chain', 'draw')).values
                cos_coef = self.trace.posterior[f'fourier_cos_coef_{period}_{k}'].mean(dim=('chain', 'draw')).values
            else:
                sin_coef = np.mean(self.trace[f'fourier_sin_coef_{period}_{k}'])
                cos_coef = np.mean(self.trace[f'fourier_cos_coef_{period}_{k}'])
            
            sin_coefficients.append(sin_coef)
            cos_coefficients.append(cos_coef)
        
        # Compute amplitude and phase
        amplitudes = np.sqrt(np.array(sin_coefficients)**2 + np.array(cos_coefficients)**2)
        
        # Plot amplitude spectrum
        frequencies = np.arange(1, fourier_order + 1) / period
        ax.bar(frequencies, amplitudes, width=0.01)
        
        # Add labels
        ax.set_xlabel('Frequency (cycles per time unit)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Fourier Spectrum (Period {period})')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def plot_fourier_components(self, 
                              period: int,
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot individual Fourier components for a specific period.
        
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
        
        # Get Fourier order for this period
        fourier_order = self.fourier_orders[period]
        
        # Create figure with subplots
        fig, axes = plt.subplots(fourier_order, 1, figsize=figsize)
        
        # If only one Fourier term, convert axes to list
        if fourier_order == 1:
            axes = [axes]
        
        # Create time index
        if self.time_index is not None and isinstance(self.time_index, pd.DatetimeIndex):
            t = self.time_index
        else:
            t = np.arange(self.n_timesteps)
        
        # Extract Fourier coefficients
        fourier_coefficients = {}
        
        for k in range(1, fourier_order + 1):
            if isinstance(self.trace, az.InferenceData):
                sin_coef = self.trace.posterior[f'fourier_sin_coef_{period}_{k}'].mean(dim=('chain', 'draw')).values
                cos_coef = self.trace.posterior[f'fourier_cos_coef_{period}_{k}'].mean(dim=('chain', 'draw')).values
            else:
                sin_coef = np.mean(self.trace[f'fourier_sin_coef_{period}_{k}'])
                cos_coef = np.mean(self.trace[f'fourier_cos_coef_{period}_{k}'])
            
            fourier_coefficients[k] = {'sin': sin_coef, 'cos': cos_coef}
        
        # Compute normalized time
        t_norm = np.arange(self.n_timesteps) / self.n_timesteps
        
        # Plot each Fourier component
        for k in range(1, fourier_order + 1):
            # Compute component
            sin_term = fourier_coefficients[k]['sin'] * np.sin(2 * np.pi * k * t_norm * self.n_timesteps / period)
            cos_term = fourier_coefficients[k]['cos'] * np.cos(2 * np.pi * k * t_norm * self.n_timesteps / period)
            component = sin_term + cos_term
            
            # Plot component
            axes[k-1].plot(t, component)
            
            # Add labels
            axes[k-1].set_xlabel('Time')
            axes[k-1].set_ylabel('Value')
            axes[k-1].set_title(f'Fourier Component {k} (Period {period})')
        
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
    
    # Create regression terms
    regression_terms = {
        'temperature': 5 * np.sin(2 * np.pi * np.arange(n_samples) / 365 + np.pi/4) + np.random.normal(0, 0.5, n_samples),
        'holiday': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    }
    
    # Create and fit Fourier seasonal model
    model = FourierSeasonalModel()
    model.build_model(
        time_series,
        trend_order=2,
        seasonal_periods=[7, 30, 365],
        fourier_orders={7: 3, 30: 5, 365: 10},
        regression_terms=regression_terms
    )
    model.sample(draws=1000, tune=1000)
    
    # Extract components
    components = model.extract_components()
    
    # Plot components
    model.plot_components()
    
    # Plot Fourier spectrum
    model.plot_fourier_spectrum(period=7)
    model.plot_fourier_spectrum(period=30)
    model.plot_fourier_spectrum(period=365)
    
    # Plot Fourier components
    model.plot_fourier_components(period=7)
    
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
