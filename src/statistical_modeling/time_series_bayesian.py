"""
Time Series Bayesian Module for the Cosmic Market Oracle.

This module implements Bayesian models for time series analysis,
providing a foundation for analyzing financial and astrological time series data.
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

from src.statistical_modeling.bayesian_core import BayesianModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesModel(BayesianModel):
    """
    Base class for Bayesian time series models.
    
    This class extends the BayesianModel class to provide functionality
    for building and analyzing time series models.
    """
    
    def __init__(self, name: str = "time_series_model"):
        """
        Initialize a time series model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.time_series = None
        self.time_index = None
        self.n_timesteps = None
        self.forecast_horizon = None
        self.forecast_results = None
    
    def build_model(self, *args, **kwargs) -> pm.Model:
        """
        Build the time series model.
        
        This is a placeholder method that should be overridden by subclasses.
        
        Returns:
            PyMC model
        """
        raise NotImplementedError("Subclasses must implement build_model method")
    
    def preprocess_data(self, time_series: Union[np.ndarray, pd.Series], 
                      differencing: int = 0) -> Tuple[np.ndarray, Optional[pd.DatetimeIndex]]:
        """
        Preprocess time series data.
        
        Args:
            time_series: Time series data
            differencing: Order of differencing
            
        Returns:
            Tuple of (preprocessed_data, time_index)
        """
        # Store original time index if available
        time_index = None
        if isinstance(time_series, pd.Series):
            time_index = time_series.index
            time_series = time_series.values
        
        # Store original data
        original_data = time_series.copy()
        
        # Apply differencing if requested
        if differencing > 0:
            for _ in range(differencing):
                time_series = np.diff(time_series)
        
        # Store preprocessed data
        self.time_series = time_series
        self.time_index = time_index
        self.n_timesteps = len(time_series)
        
        return time_series, time_index
    
    def forecast(self, 
               horizon: int, 
               return_samples: bool = False,
               num_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Generate forecasts for future time steps.
        
        Args:
            horizon: Number of time steps to forecast
            return_samples: Whether to return posterior samples
            num_samples: Number of posterior samples to generate
            
        Returns:
            Dictionary with forecast results
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been built or sampled yet.")
        
        # Store forecast horizon
        self.forecast_horizon = horizon
        
        # Generate posterior predictive samples
        with self.model:
            # This is a placeholder - actual implementation would depend on the specific model
            # and would need to be overridden by subclasses
            posterior_predictive = pm.sample_posterior_predictive(
                self.trace,
                var_names=['forecast'],
                random_seed=42
            )
        
        # Extract forecast samples
        if isinstance(posterior_predictive, az.InferenceData):
            forecast_samples = posterior_predictive.posterior_predictive['forecast'].values
        else:
            forecast_samples = posterior_predictive['forecast']
        
        # Compute forecast statistics
        forecast_mean = np.mean(forecast_samples, axis=0)
        forecast_std = np.std(forecast_samples, axis=0)
        forecast_lower = np.percentile(forecast_samples, 2.5, axis=0)
        forecast_upper = np.percentile(forecast_samples, 97.5, axis=0)
        
        # Store forecast results
        self.forecast_results = {
            'mean': forecast_mean,
            'std': forecast_std,
            'lower': forecast_lower,
            'upper': forecast_upper
        }
        
        if return_samples:
            self.forecast_results['samples'] = forecast_samples
        
        return self.forecast_results
    
    def plot_forecast(self, 
                    include_history: bool = True,
                    n_history: Optional[int] = None,
                    figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot forecast results.
        
        Args:
            include_history: Whether to include historical data
            n_history: Number of historical time steps to include
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.forecast_results is None:
            raise ValueError("No forecast results available. Call forecast first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine x-axis values
        if self.time_index is not None and isinstance(self.time_index, pd.DatetimeIndex):
            # Use datetime index if available
            last_date = self.time_index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=self.forecast_horizon,
                freq=pd.infer_freq(self.time_index)
            )
            x_forecast = forecast_dates
            
            if include_history:
                if n_history is None:
                    x_history = self.time_index
                else:
                    x_history = self.time_index[-n_history:]
        else:
            # Use integer index
            last_idx = self.n_timesteps - 1
            x_forecast = np.arange(last_idx + 1, last_idx + 1 + self.forecast_horizon)
            
            if include_history:
                if n_history is None:
                    x_history = np.arange(self.n_timesteps)
                else:
                    x_history = np.arange(last_idx - n_history + 1, last_idx + 1)
        
        # Plot historical data if requested
        if include_history:
            if n_history is None:
                history_data = self.time_series
            else:
                history_data = self.time_series[-n_history:]
            
            ax.plot(x_history, history_data, 'b-', label='Historical Data')
        
        # Plot forecast
        ax.plot(x_forecast, self.forecast_results['mean'], 'r-', label='Forecast')
        
        # Plot confidence intervals
        ax.fill_between(
            x_forecast,
            self.forecast_results['lower'],
            self.forecast_results['upper'],
            color='r',
            alpha=0.2,
            label='95% Credible Interval'
        )
        
        # Add labels and legend
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Time Series Forecast')
        ax.legend()
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def compute_forecast_metrics(self, 
                               actual: np.ndarray) -> Dict[str, float]:
        """
        Compute forecast accuracy metrics.
        
        Args:
            actual: Actual values for the forecast period
            
        Returns:
            Dictionary with forecast metrics
        """
        if self.forecast_results is None:
            raise ValueError("No forecast results available. Call forecast first.")
        
        if len(actual) != len(self.forecast_results['mean']):
            raise ValueError(f"Length of actual values ({len(actual)}) does not match "
                           f"forecast horizon ({len(self.forecast_results['mean'])})")
        
        # Compute forecast errors
        errors = actual - self.forecast_results['mean']
        abs_errors = np.abs(errors)
        squared_errors = errors**2
        
        # Compute metrics
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(squared_errors))
        mape = np.mean(np.abs(errors / actual)) * 100
        
        # Compute coverage probability
        in_interval = np.logical_and(
            actual >= self.forecast_results['lower'],
            actual <= self.forecast_results['upper']
        )
        coverage = np.mean(in_interval) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'coverage': coverage
        }
    
    def plot_residuals(self, 
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot residual diagnostics.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been built or sampled yet.")
        
        # Generate posterior predictive samples for historical data
        with self.model:
            posterior_predictive = pm.sample_posterior_predictive(
                self.trace,
                var_names=['y'],
                random_seed=42
            )
        
        # Extract predicted values
        if isinstance(posterior_predictive, az.InferenceData):
            y_pred = posterior_predictive.posterior_predictive['y'].mean(dim=('chain', 'draw')).values
        else:
            y_pred = np.mean(posterior_predictive['y'], axis=0)
        
        # Compute residuals
        residuals = self.time_series - y_pred
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot residuals vs. time
        if self.time_index is not None and isinstance(self.time_index, pd.DatetimeIndex):
            axes[0, 0].plot(self.time_index, residuals, 'o')
        else:
            axes[0, 0].plot(np.arange(self.n_timesteps), residuals, 'o')
        
        axes[0, 0].axhline(y=0, color='r', linestyle='-')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs. Time')
        
        # Plot residuals vs. fitted values
        axes[0, 1].scatter(y_pred, residuals)
        axes[0, 1].axhline(y=0, color='r', linestyle='-')
        axes[0, 1].set_xlabel('Fitted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs. Fitted')
        
        # Plot residual histogram
        axes[1, 0].hist(residuals, bins=20, density=True, alpha=0.6)
        
        # Add normal density for comparison
        xmin, xmax = axes[1, 0].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        mean = np.mean(residuals)
        std = np.std(residuals)
        p = stats.norm.pdf(x, mean, std)
        axes[1, 0].plot(x, p, 'r-', linewidth=2)
        
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Residual Histogram')
        
        # Plot residual Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Residual Q-Q Plot')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def plot_components(self, 
                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot time series components (trend, seasonal, residual).
        
        This is a placeholder method that should be overridden by subclasses
        that implement decomposition.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        raise NotImplementedError("Component plotting not implemented for this model")
    
    def compute_autocorrelation(self, 
                              lags: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute autocorrelation function (ACF) for the time series.
        
        Args:
            lags: Number of lags to compute
            
        Returns:
            Tuple of (lags, autocorrelation)
        """
        from statsmodels.tsa.stattools import acf
        
        # Compute autocorrelation
        acf_values = acf(self.time_series, nlags=lags, fft=True)
        lag_values = np.arange(lags + 1)
        
        return lag_values, acf_values
    
    def compute_partial_autocorrelation(self, 
                                      lags: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute partial autocorrelation function (PACF) for the time series.
        
        Args:
            lags: Number of lags to compute
            
        Returns:
            Tuple of (lags, partial_autocorrelation)
        """
        from statsmodels.tsa.stattools import pacf
        
        # Compute partial autocorrelation
        pacf_values = pacf(self.time_series, nlags=lags, method='ols')
        lag_values = np.arange(lags + 1)
        
        return lag_values, pacf_values
    
    def plot_autocorrelation(self, 
                           lags: int = 40,
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot autocorrelation function (ACF) for the time series.
        
        Args:
            lags: Number of lags to compute
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Compute autocorrelation
        lag_values, acf_values = self.compute_autocorrelation(lags)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot autocorrelation
        ax.stem(lag_values, acf_values, basefmt='b-')
        
        # Add confidence intervals (95%)
        conf_level = 1.96 / np.sqrt(self.n_timesteps)
        ax.axhline(y=conf_level, color='r', linestyle='--', alpha=0.7)
        ax.axhline(y=-conf_level, color='r', linestyle='--', alpha=0.7)
        
        # Add labels
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('Autocorrelation Function (ACF)')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def plot_partial_autocorrelation(self, 
                                   lags: int = 40,
                                   figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot partial autocorrelation function (PACF) for the time series.
        
        Args:
            lags: Number of lags to compute
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Compute partial autocorrelation
        lag_values, pacf_values = self.compute_partial_autocorrelation(lags)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot partial autocorrelation
        ax.stem(lag_values, pacf_values, basefmt='b-')
        
        # Add confidence intervals (95%)
        conf_level = 1.96 / np.sqrt(self.n_timesteps)
        ax.axhline(y=conf_level, color='r', linestyle='--', alpha=0.7)
        ax.axhline(y=-conf_level, color='r', linestyle='--', alpha=0.7)
        
        # Add labels
        ax.set_xlabel('Lag')
        ax.set_ylabel('Partial Autocorrelation')
        ax.set_title('Partial Autocorrelation Function (PACF)')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def plot_diagnostics(self, 
                       lags: int = 40,
                       figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot comprehensive diagnostics for the time series model.
        
        Args:
            lags: Number of lags for autocorrelation
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # Plot original time series
        if self.time_index is not None and isinstance(self.time_index, pd.DatetimeIndex):
            axes[0, 0].plot(self.time_index, self.time_series)
        else:
            axes[0, 0].plot(np.arange(self.n_timesteps), self.time_series)
        
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Original Time Series')
        
        # Plot autocorrelation
        lag_values, acf_values = self.compute_autocorrelation(lags)
        axes[0, 1].stem(lag_values, acf_values, basefmt='b-')
        
        # Add confidence intervals (95%)
        conf_level = 1.96 / np.sqrt(self.n_timesteps)
        axes[0, 1].axhline(y=conf_level, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].axhline(y=-conf_level, color='r', linestyle='--', alpha=0.7)
        
        axes[0, 1].set_xlabel('Lag')
        axes[0, 1].set_ylabel('Autocorrelation')
        axes[0, 1].set_title('Autocorrelation Function (ACF)')
        
        # Plot partial autocorrelation
        lag_values, pacf_values = self.compute_partial_autocorrelation(lags)
        axes[1, 0].stem(lag_values, pacf_values, basefmt='b-')
        
        # Add confidence intervals (95%)
        axes[1, 0].axhline(y=conf_level, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].axhline(y=-conf_level, color='r', linestyle='--', alpha=0.7)
        
        axes[1, 0].set_xlabel('Lag')
        axes[1, 0].set_ylabel('Partial Autocorrelation')
        axes[1, 0].set_title('Partial Autocorrelation Function (PACF)')
        
        # Plot histogram
        axes[1, 1].hist(self.time_series, bins=20, density=True, alpha=0.6)
        
        # Add normal density for comparison
        xmin, xmax = axes[1, 1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        mean = np.mean(self.time_series)
        std = np.std(self.time_series)
        p = stats.norm.pdf(x, mean, std)
        axes[1, 1].plot(x, p, 'r-', linewidth=2)
        
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Histogram')
        
        # Plot Q-Q plot
        from scipy import stats
        stats.probplot(self.time_series, dist="norm", plot=axes[2, 0])
        axes[2, 0].set_title('Q-Q Plot')
        
        # Plot seasonal decomposition if available
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Infer frequency
            if self.time_index is not None and isinstance(self.time_index, pd.DatetimeIndex):
                freq = pd.infer_freq(self.time_index)
                if freq is not None:
                    # Convert frequency to integer
                    if freq == 'D':
                        period = 7  # Weekly seasonality
                    elif freq == 'M':
                        period = 12  # Monthly seasonality
                    elif freq == 'Q':
                        period = 4  # Quarterly seasonality
                    elif freq == 'A':
                        period = 1  # Annual seasonality
                    else:
                        period = 12  # Default
                else:
                    period = 12  # Default
            else:
                period = 12  # Default
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                self.time_series, 
                model='additive', 
                period=period
            )
            
            # Plot trend component
            if self.time_index is not None and isinstance(self.time_index, pd.DatetimeIndex):
                axes[2, 1].plot(self.time_index, decomposition.trend)
            else:
                axes[2, 1].plot(np.arange(self.n_timesteps), decomposition.trend)
            
            axes[2, 1].set_xlabel('Time')
            axes[2, 1].set_ylabel('Trend')
            axes[2, 1].set_title('Trend Component')
        except:
            axes[2, 1].text(0.5, 0.5, 'Seasonal decomposition not available',
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=axes[2, 1].transAxes)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # This is a placeholder for example usage
    # Actual implementation would depend on specific subclasses
    pass
