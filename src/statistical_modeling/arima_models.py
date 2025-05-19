"""
ARIMA Models Module for the Cosmic Market Oracle.

This module implements Bayesian ARIMA (Autoregressive Integrated Moving Average) models
for time series analysis, providing specialized models for analyzing financial and 
astrological time series data.
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
from src.statistical_modeling.ar_models import ARModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MAModel(TimeSeriesModel):
    """
    Bayesian Moving Average (MA) model.
    
    This class implements a Bayesian MA(q) model for time series analysis.
    """
    
    def __init__(self, name: str = "ma_model"):
        """
        Initialize an MA model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.order = None
        self.ma_coefficients = None
    
    def build_model(self, time_series: Union[np.ndarray, pd.Series], 
                  order: int = 1,
                  differencing: int = 0,
                  standardize: bool = True,
                  priors: Optional[Dict] = None) -> pm.Model:
        """
        Build a Bayesian MA(q) model.
        
        Args:
            time_series: Time series data
            order: Order of the MA model (q)
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
                'ma_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 0.5}},
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
            
            # Priors for MA coefficients
            ma_prior = getattr(pm, priors['ma_coefficients']['dist'])
            ma_coefficients = ma_prior(
                'ma_coefficients', 
                **priors['ma_coefficients']['params'],
                shape=order
            )
            
            # Prior for error term
            sigma_prior = getattr(pm, priors['sigma']['dist'])
            sigma = sigma_prior('sigma', **priors['sigma']['params'])
            
            # Set up MA model
            mu = pm.Deterministic('mu', tt.zeros(len(data)))
            
            # Initialize with intercept
            mu = tt.set_subtensor(mu[:], intercept)
            
            # Initialize errors
            errors = tt.zeros(len(data) + order)
            
            # Likelihood
            y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
            
            # Set up forecast
            steps_ahead = 10  # Default forecast horizon
            forecast = pm.Deterministic('forecast', tt.zeros(steps_ahead))
            
            # Initialize forecast with intercept
            forecast = tt.set_subtensor(forecast[:], intercept)
        
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
            ma_coefficients = self.trace.posterior['ma_coefficients'].values
            sigma = self.trace.posterior['sigma'].values.flatten()
        else:
            intercept = self.trace['intercept']
            ma_coefficients = self.trace['ma_coefficients']
            sigma = self.trace['sigma']
        
        # Reshape MA coefficients
        ma_coefficients = np.reshape(ma_coefficients, (-1, self.order))
        
        # Generate forecasts
        n_samples = len(intercept)
        forecasts = np.zeros((n_samples, horizon))
        
        for i in range(n_samples):
            forecast_values = np.zeros(horizon)
            errors = np.zeros(horizon + self.order)
            
            # Generate random errors for future time steps
            if include_uncertainty:
                errors[-self.order:] = np.random.normal(0, sigma[i], self.order)
            
            for t in range(horizon):
                # Compute prediction
                pred = intercept[i]
                for j in range(self.order):
                    if t-j-1 >= 0:
                        # Use previous forecast errors
                        pred += ma_coefficients[i, j] * errors[t-j-1]
                    else:
                        # Use initial errors
                        pred += ma_coefficients[i, j] * errors[self.order+t-j-1]
                
                # Add process uncertainty if requested
                if include_uncertainty:
                    errors[t] = np.random.normal(0, sigma[i])
                
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
    
    def plot_ma_coefficients(self, 
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot posterior distributions of MA coefficients.
        
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
            ma_coefficients = self.trace.posterior['ma_coefficients'].values
        else:
            ma_coefficients = self.trace['ma_coefficients']
        
        # Reshape coefficients
        ma_coefficients = np.reshape(ma_coefficients, (-1, self.order))
        
        # Compute mean and credible intervals
        coef_mean = np.mean(ma_coefficients, axis=0)
        coef_hdi = az.hdi(ma_coefficients)
        
        # Create labels
        labels = [f"MA({i+1})" for i in range(self.order)]
        
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
        ax.set_title('Posterior Distributions of MA Coefficients')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


class ARMAModel(TimeSeriesModel):
    """
    Bayesian Autoregressive Moving Average (ARMA) model.
    
    This class implements a Bayesian ARMA(p,q) model for time series analysis.
    """
    
    def __init__(self, name: str = "arma_model"):
        """
        Initialize an ARMA model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.ar_order = None
        self.ma_order = None
        self.ar_coefficients = None
        self.ma_coefficients = None
    
    def build_model(self, time_series: Union[np.ndarray, pd.Series], 
                  ar_order: int = 1,
                  ma_order: int = 1,
                  differencing: int = 0,
                  standardize: bool = True,
                  priors: Optional[Dict] = None) -> pm.Model:
        """
        Build a Bayesian ARMA(p,q) model.
        
        Args:
            time_series: Time series data
            ar_order: Order of the AR component (p)
            ma_order: Order of the MA component (q)
            differencing: Order of differencing
            standardize: Whether to standardize the data
            priors: Dictionary of prior distributions for parameters
            
        Returns:
            PyMC model
        """
        # Preprocess data
        data, time_index = self.preprocess_data(time_series, differencing)
        
        # Store orders
        self.ar_order = ar_order
        self.ma_order = ma_order
        
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
                'ma_coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 0.5}},
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
            if ar_order > 0:
                ar_prior = getattr(pm, priors['ar_coefficients']['dist'])
                ar_coefficients = ar_prior(
                    'ar_coefficients', 
                    **priors['ar_coefficients']['params'],
                    shape=ar_order
                )
            
            # Priors for MA coefficients
            if ma_order > 0:
                ma_prior = getattr(pm, priors['ma_coefficients']['dist'])
                ma_coefficients = ma_prior(
                    'ma_coefficients', 
                    **priors['ma_coefficients']['params'],
                    shape=ma_order
                )
            
            # Prior for error term
            sigma_prior = getattr(pm, priors['sigma']['dist'])
            sigma = sigma_prior('sigma', **priors['sigma']['params'])
            
            # Set up ARMA model
            mu = pm.Deterministic('mu', tt.zeros(len(data)))
            
            # Initialize with intercept
            max_lag = max(ar_order, ma_order)
            mu = tt.set_subtensor(mu[:max_lag], intercept)
            
            # Initialize errors
            errors = tt.zeros(len(data) + ma_order)
            
            # Compute ARMA predictions
            for t in range(max_lag, len(data)):
                pred = intercept
                
                # Add AR terms
                for i in range(ar_order):
                    pred = pred + ar_coefficients[i] * data[t-i-1]
                
                # Add MA terms
                for i in range(ma_order):
                    if t-i-1 >= 0:
                        pred = pred + ma_coefficients[i] * (data[t-i-1] - mu[t-i-1])
                
                mu = tt.set_subtensor(mu[t], pred)
            
            # Likelihood
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
            
            # Set up forecast
            steps_ahead = 10  # Default forecast horizon
            forecast = pm.Deterministic('forecast', tt.zeros(steps_ahead))
            
            # Initialize forecast with intercept
            forecast = tt.set_subtensor(forecast[:], intercept)
        
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
            if self.ar_order > 0:
                ar_coefficients = self.trace.posterior['ar_coefficients'].values
                ar_coefficients = np.reshape(ar_coefficients, (-1, self.ar_order))
            else:
                ar_coefficients = None
            
            if self.ma_order > 0:
                ma_coefficients = self.trace.posterior['ma_coefficients'].values
                ma_coefficients = np.reshape(ma_coefficients, (-1, self.ma_order))
            else:
                ma_coefficients = None
            
            sigma = self.trace.posterior['sigma'].values.flatten()
        else:
            intercept = self.trace['intercept']
            
            if self.ar_order > 0:
                ar_coefficients = self.trace['ar_coefficients']
                ar_coefficients = np.reshape(ar_coefficients, (-1, self.ar_order))
            else:
                ar_coefficients = None
            
            if self.ma_order > 0:
                ma_coefficients = self.trace['ma_coefficients']
                ma_coefficients = np.reshape(ma_coefficients, (-1, self.ma_order))
            else:
                ma_coefficients = None
            
            sigma = self.trace['sigma']
        
        # Get last values from the time series
        last_values = self.time_series[-self.ar_order:] if self.ar_order > 0 else None
        
        # Compute last errors
        if self.ma_order > 0:
            # Extract predicted values
            if isinstance(self.trace, az.InferenceData):
                y_pred = self.trace.posterior['mu'].mean(dim=('chain', 'draw')).values
            else:
                y_pred = np.mean(self.trace['mu'], axis=0)
            
            # Compute errors
            last_errors = self.time_series[-self.ma_order:] - y_pred[-self.ma_order:]
        else:
            last_errors = None
        
        # Generate forecasts
        n_samples = len(intercept)
        forecasts = np.zeros((n_samples, horizon))
        
        for i in range(n_samples):
            forecast_values = np.zeros(horizon)
            
            # Initialize errors
            if self.ma_order > 0:
                errors = np.zeros(horizon + self.ma_order)
                errors[:self.ma_order] = last_errors
            
            for t in range(horizon):
                # Compute prediction
                pred = intercept[i]
                
                # Add AR terms
                if self.ar_order > 0:
                    for j in range(self.ar_order):
                        if t-j-1 < 0:
                            # Use historical data
                            pred += ar_coefficients[i, j] * last_values[self.ar_order+t-j-1]
                        else:
                            # Use forecasted values
                            pred += ar_coefficients[i, j] * forecast_values[t-j-1]
                
                # Add MA terms
                if self.ma_order > 0:
                    for j in range(self.ma_order):
                        pred += ma_coefficients[i, j] * errors[t+self.ma_order-j-1]
                
                # Add process uncertainty if requested
                if include_uncertainty:
                    error = np.random.normal(0, sigma[i])
                    if self.ma_order > 0:
                        errors[t+self.ma_order] = error
                else:
                    error = 0
                
                forecast_values[t] = pred + error
            
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
    
    def plot_coefficients(self, 
                        figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot posterior distributions of ARMA coefficients.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot AR coefficients
        if self.ar_order > 0:
            # Extract coefficient samples
            if isinstance(self.trace, az.InferenceData):
                ar_coefficients = self.trace.posterior['ar_coefficients'].values
            else:
                ar_coefficients = self.trace['ar_coefficients']
            
            # Reshape coefficients
            ar_coefficients = np.reshape(ar_coefficients, (-1, self.ar_order))
            
            # Compute mean and credible intervals
            ar_coef_mean = np.mean(ar_coefficients, axis=0)
            ar_coef_hdi = az.hdi(ar_coefficients)
            
            # Create labels
            ar_labels = [f"AR({i+1})" for i in range(self.ar_order)]
            
            # Plot coefficients
            axes[0].errorbar(
                x=ar_coef_mean,
                y=ar_labels,
                xerr=np.abs(ar_coef_hdi - ar_coef_mean[:, np.newaxis]).T,
                fmt='o',
                capsize=5
            )
            
            # Add vertical line at zero
            axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            
            # Add labels
            axes[0].set_xlabel('Coefficient Value')
            axes[0].set_ylabel('Lag')
            axes[0].set_title('Posterior Distributions of AR Coefficients')
        else:
            axes[0].text(0.5, 0.5, 'No AR coefficients',
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=axes[0].transAxes)
        
        # Plot MA coefficients
        if self.ma_order > 0:
            # Extract coefficient samples
            if isinstance(self.trace, az.InferenceData):
                ma_coefficients = self.trace.posterior['ma_coefficients'].values
            else:
                ma_coefficients = self.trace['ma_coefficients']
            
            # Reshape coefficients
            ma_coefficients = np.reshape(ma_coefficients, (-1, self.ma_order))
            
            # Compute mean and credible intervals
            ma_coef_mean = np.mean(ma_coefficients, axis=0)
            ma_coef_hdi = az.hdi(ma_coefficients)
            
            # Create labels
            ma_labels = [f"MA({i+1})" for i in range(self.ma_order)]
            
            # Plot coefficients
            axes[1].errorbar(
                x=ma_coef_mean,
                y=ma_labels,
                xerr=np.abs(ma_coef_hdi - ma_coef_mean[:, np.newaxis]).T,
                fmt='o',
                capsize=5
            )
            
            # Add vertical line at zero
            axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            
            # Add labels
            axes[1].set_xlabel('Coefficient Value')
            axes[1].set_ylabel('Lag')
            axes[1].set_title('Posterior Distributions of MA Coefficients')
        else:
            axes[1].text(0.5, 0.5, 'No MA coefficients',
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=axes[1].transAxes)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Generate sample data with ARMA structure
    np.random.seed(42)
    n_samples = 200
    
    # Create time index
    time_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate ARMA(1,1) process
    ar_coef = 0.7
    ma_coef = 0.4
    sigma = 0.5
    
    # Initialize time series and errors
    ts = np.zeros(n_samples)
    errors = np.random.normal(0, sigma, n_samples)
    
    # Generate ARMA(1,1) process
    ts[0] = errors[0]
    for t in range(1, n_samples):
        ts[t] = ar_coef * ts[t-1] + errors[t] + ma_coef * errors[t-1]
    
    # Create pandas Series
    time_series = pd.Series(ts, index=time_index)
    
    # Create and fit MA model
    ma_model = MAModel()
    ma_model.build_model(time_series, order=2)
    ma_model.sample(draws=1000, tune=1000)
    
    # Plot MA coefficients
    ma_model.plot_ma_coefficients()
    
    # Generate forecast
    forecast = ma_model.forecast(horizon=30)
    
    # Plot forecast
    ma_model.plot_forecast()
    
    # Create and fit ARMA model
    arma_model = ARMAModel()
    arma_model.build_model(time_series, ar_order=1, ma_order=1)
    arma_model.sample(draws=1000, tune=1000)
    
    # Plot coefficients
    arma_model.plot_coefficients()
    
    # Generate forecast
    forecast = arma_model.forecast(horizon=30)
    
    # Plot forecast
    arma_model.plot_forecast()
    
    plt.show()
