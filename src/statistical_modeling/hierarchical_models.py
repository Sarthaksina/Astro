"""
Hierarchical Models Module for the Cosmic Market Oracle.

This module implements Bayesian hierarchical models for multi-level data analysis,
particularly useful for analyzing nested data structures such as time series data
across multiple timeframes.
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


class HierarchicalModel(BayesianModel):
    """
    Base class for Bayesian hierarchical models.
    
    This class extends the BayesianModel class to provide functionality
    for building and analyzing multi-level models.
    """
    
    def __init__(self, name: str = "hierarchical_model"):
        """
        Initialize a hierarchical model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.group_data = None
        self.group_ids = None
        self.n_groups = None
        self.group_level_params = {}
    
    def build_model(self, *args, **kwargs) -> pm.Model:
        """
        Build the hierarchical model.
        
        This is a placeholder method that should be overridden by subclasses.
        
        Returns:
            PyMC model
        """
        raise NotImplementedError("Subclasses must implement build_model method")
    
    def extract_group_params(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract group-level parameters from the trace.
        
        Returns:
            Dictionary of group-level parameters
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        group_params = {}
        
        # Extract group-level parameters from trace
        for param_name in self.group_level_params:
            if isinstance(self.trace, az.InferenceData):
                param_samples = self.trace.posterior[param_name].values
            else:
                param_samples = self.trace[param_name]
            
            # Compute mean and credible intervals for each group
            means = np.mean(param_samples, axis=(0, 1))
            hdi = az.hdi(param_samples)
            
            group_params[param_name] = {
                'mean': means,
                'hdi_lower': hdi[:, 0],
                'hdi_upper': hdi[:, 1]
            }
        
        return group_params
    
    def plot_group_params(self, 
                        param_name: str,
                        group_names: Optional[List[str]] = None,
                        figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot group-level parameters.
        
        Args:
            param_name: Name of the parameter to plot
            group_names: Names of groups (if None, uses group indices)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if param_name not in self.group_level_params:
            raise ValueError(f"Parameter {param_name} is not a group-level parameter")
        
        # Extract group parameters
        group_params = self.extract_group_params()
        
        # Set default group names if not provided
        if group_names is None:
            group_names = [f"Group {i+1}" for i in range(self.n_groups)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot group parameters with credible intervals
        ax.errorbar(
            x=group_params[param_name]['mean'],
            y=group_names,
            xerr=np.vstack([
                group_params[param_name]['mean'] - group_params[param_name]['hdi_lower'],
                group_params[param_name]['hdi_upper'] - group_params[param_name]['mean']
            ]),
            fmt='o',
            capsize=5
        )
        
        # Add global mean if available
        if f"{param_name}_mu" in self.group_level_params:
            global_mean = self.extract_group_params()[f"{param_name}_mu"]['mean']
            ax.axvline(x=global_mean, color='red', linestyle='--', 
                      label=f"Global Mean: {global_mean:.3f}")
            ax.legend()
        
        # Add labels
        ax.set_xlabel(f"{param_name} Value")
        ax.set_ylabel("Group")
        ax.set_title(f"Group-level {param_name} Parameters")
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def plot_group_effects(self, 
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot all group-level effects.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Extract group parameters
        group_params = self.extract_group_params()
        
        # Count number of parameters (excluding global parameters)
        n_params = sum(1 for param in self.group_level_params 
                      if not param.endswith('_mu') and not param.endswith('_sigma'))
        
        # Create figure
        fig, axes = plt.subplots(1, n_params, figsize=figsize)
        
        # Handle case with only one parameter
        if n_params == 1:
            axes = [axes]
        
        # Plot each parameter
        param_idx = 0
        for param_name in self.group_level_params:
            if param_name.endswith('_mu') or param_name.endswith('_sigma'):
                continue
                
            # Sort groups by parameter value
            sorted_idx = np.argsort(group_params[param_name]['mean'])
            
            # Plot sorted group parameters with credible intervals
            axes[param_idx].errorbar(
                x=np.arange(self.n_groups),
                y=group_params[param_name]['mean'][sorted_idx],
                yerr=np.vstack([
                    group_params[param_name]['mean'][sorted_idx] - group_params[param_name]['hdi_lower'][sorted_idx],
                    group_params[param_name]['hdi_upper'][sorted_idx] - group_params[param_name]['mean'][sorted_idx]
                ]),
                fmt='o',
                capsize=5
            )
            
            # Add global mean if available
            if f"{param_name}_mu" in self.group_level_params:
                global_mean = self.extract_group_params()[f"{param_name}_mu"]['mean']
                axes[param_idx].axhline(y=global_mean, color='red', linestyle='--', 
                                      label=f"Global Mean: {global_mean:.3f}")
                axes[param_idx].legend()
            
            # Add labels
            axes[param_idx].set_xlabel("Group Index (sorted)")
            axes[param_idx].set_ylabel(f"{param_name} Value")
            axes[param_idx].set_title(f"Group-level {param_name}")
            
            param_idx += 1
        
        # Add overall title
        fig.suptitle("Group-level Effects", fontsize=16)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        
        return fig
    
    def compute_icc(self, param_name: str) -> Dict[str, float]:
        """
        Compute Intraclass Correlation Coefficient (ICC) for a parameter.
        
        The ICC measures the proportion of variance explained by group membership.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Dictionary with ICC statistics
        """
        if f"{param_name}_sigma" not in self.group_level_params:
            raise ValueError(f"Parameter {param_name}_sigma not found in group-level parameters")
        
        if "sigma" not in self.summary.index:
            raise ValueError("Residual standard deviation (sigma) not found in model summary")
        
        # Extract group-level and residual standard deviations
        group_sd = self.summary.loc[f"{param_name}_sigma", "mean"]
        residual_sd = self.summary.loc["sigma", "mean"]
        
        # Compute ICC
        icc = group_sd**2 / (group_sd**2 + residual_sd**2)
        
        return {
            'icc': icc,
            'group_variance': group_sd**2,
            'residual_variance': residual_sd**2,
            'total_variance': group_sd**2 + residual_sd**2
        }
    
    def compute_shrinkage(self, param_name: str) -> np.ndarray:
        """
        Compute shrinkage factors for group-level parameters.
        
        Shrinkage measures how much each group's estimate is pulled toward the global mean.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Array of shrinkage factors for each group
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Extract group-level and global parameters
        group_params = self.extract_group_params()
        
        if f"{param_name}_mu" not in self.group_level_params:
            raise ValueError(f"Global mean parameter {param_name}_mu not found")
        
        # Get global mean
        global_mean = group_params[f"{param_name}_mu"]['mean']
        
        # Get group means
        group_means = group_params[param_name]['mean']
        
        # Compute raw group means from data (no pooling estimates)
        # This is a placeholder - actual implementation would depend on the specific model
        # and would need to be overridden by subclasses
        no_pooling_means = group_means  # Placeholder
        
        # Compute shrinkage factors
        # Shrinkage = 1 - (distance from global mean) / (distance from no pooling estimate)
        shrinkage = 1 - np.abs(group_means - global_mean) / np.abs(no_pooling_means - global_mean)
        
        return shrinkage
    
    def predict_for_group(self, 
                        X_new: np.ndarray, 
                        group_id: int,
                        **kwargs) -> Dict[str, np.ndarray]:
        """
        Make predictions for a specific group.
        
        This is a placeholder method that should be overridden by subclasses.
        
        Args:
            X_new: New predictor variables
            group_id: ID of the group to predict for
            **kwargs: Additional arguments for prediction
            
        Returns:
            Dictionary with predictions
        """
        raise NotImplementedError("Subclasses must implement predict_for_group method")


class VaryingInterceptModel(HierarchicalModel):
    """
    Hierarchical linear regression model with varying intercepts.
    
    This model allows intercepts to vary by group while keeping slopes fixed.
    """
    
    def __init__(self, name: str = "varying_intercept_model"):
        """
        Initialize a varying intercept model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
    
    def build_model(self, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray,
                  priors: Optional[Dict] = None) -> pm.Model:
        """
        Build a hierarchical linear regression model with varying intercepts.
        
        Args:
            X: Predictor variables (n_samples, n_features)
            y: Target variable (n_samples,)
            group_ids: Group IDs for each sample (n_samples,)
            priors: Dictionary of prior distributions for parameters
            
        Returns:
            PyMC model
        """
        # Store data
        self.X = X
        self.y = y
        self.group_ids = group_ids
        
        # Get dimensions
        n_samples, n_features = X.shape
        self.n_groups = len(np.unique(group_ids))
        
        # Set default priors if not provided
        if priors is None:
            priors = {
                'intercept_mu': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 10}},
                'intercept_sigma': {'dist': 'HalfNormal', 'params': {'sigma': 5}},
                'coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 1}},
                'sigma': {'dist': 'HalfNormal', 'params': {'sigma': 1}}
            }
        
        # Create model
        self.model = pm.Model()
        
        with self.model:
            # Priors for global intercept
            intercept_mu_prior = getattr(pm, priors['intercept_mu']['dist'])
            intercept_mu = intercept_mu_prior(
                'intercept_mu', **priors['intercept_mu']['params']
            )
            
            # Prior for intercept standard deviation
            intercept_sigma_prior = getattr(pm, priors['intercept_sigma']['dist'])
            intercept_sigma = intercept_sigma_prior(
                'intercept_sigma', **priors['intercept_sigma']['params']
            )
            
            # Group-level intercepts
            intercepts = pm.Normal(
                'intercepts',
                mu=intercept_mu,
                sigma=intercept_sigma,
                shape=self.n_groups
            )
            
            # Priors for coefficients
            coef_prior = getattr(pm, priors['coefficients']['dist'])
            coefficients = coef_prior(
                'coefficients', 
                **priors['coefficients']['params'],
                shape=n_features
            )
            
            # Prior for error term
            sigma_prior = getattr(pm, priors['sigma']['dist'])
            sigma = sigma_prior('sigma', **priors['sigma']['params'])
            
            # Linear predictor
            mu = intercepts[group_ids] + pm.math.dot(X, coefficients)
            
            # Likelihood
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
        
        # Store group-level parameters
        self.group_level_params = {
            'intercepts': intercepts,
            'intercept_mu': intercept_mu,
            'intercept_sigma': intercept_sigma
        }
        
        return self.model
    
    def predict(self, X_new: np.ndarray, group_ids: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Make predictions for new data.
        
        Args:
            X_new: New predictor variables (n_samples, n_features)
            group_ids: Group IDs for new samples (if None, uses global mean intercept)
            
        Returns:
            Dictionary with predicted mean and standard deviation
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Get posterior samples
        if isinstance(self.trace, az.InferenceData):
            coefficients = self.trace.posterior['coefficients'].values
            if group_ids is not None:
                intercepts = self.trace.posterior['intercepts'].values
            else:
                intercept_mu = self.trace.posterior['intercept_mu'].values
            sigma = self.trace.posterior['sigma'].values
        else:
            coefficients = self.trace['coefficients']
            if group_ids is not None:
                intercepts = self.trace['intercepts']
            else:
                intercept_mu = self.trace['intercept_mu']
            sigma = self.trace['sigma']
        
        # Reshape coefficients for matrix multiplication
        coefficients = np.reshape(coefficients, (-1, X_new.shape[1]))
        
        # Compute predictions
        if group_ids is not None:
            # Use group-specific intercepts
            intercepts = np.reshape(intercepts, (-1, self.n_groups))
            group_intercepts = intercepts[:, group_ids]
            predictions = group_intercepts + np.dot(coefficients, X_new.T)
        else:
            # Use global mean intercept
            intercept_mu = np.reshape(intercept_mu, (-1, 1))
            predictions = intercept_mu + np.dot(coefficients, X_new.T)
        
        # Compute mean and standard deviation of predictions
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
        
        return {
            'mean': pred_mean,
            'std': pred_std
        }
    
    def predict_for_group(self, 
                        X_new: np.ndarray, 
                        group_id: int) -> Dict[str, np.ndarray]:
        """
        Make predictions for a specific group.
        
        Args:
            X_new: New predictor variables (n_samples, n_features)
            group_id: ID of the group to predict for
            
        Returns:
            Dictionary with predicted mean and standard deviation
        """
        # Create group IDs array with the specified group ID
        group_ids = np.full(X_new.shape[0], group_id)
        
        # Make predictions
        return self.predict(X_new, group_ids)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_groups = 8
    n_samples_per_group = 30
    n_samples = n_groups * n_samples_per_group
    n_features = 2
    
    # Generate group IDs
    group_ids = np.repeat(np.arange(n_groups), n_samples_per_group)
    
    # Generate true group intercepts
    true_intercept_mu = 2.5
    true_intercept_sigma = 1.0
    true_intercepts = true_intercept_mu + true_intercept_sigma * np.random.randn(n_groups)
    
    # Generate true coefficients
    true_coefficients = np.array([1.0, -0.5])
    
    # Generate predictors
    X = np.random.randn(n_samples, n_features)
    
    # Generate target
    y = true_intercepts[group_ids] + np.dot(X, true_coefficients) + 0.5 * np.random.randn(n_samples)
    
    # Create and fit varying intercept model
    model = VaryingInterceptModel()
    model.build_model(X, y, group_ids)
    model.sample(draws=1000, tune=1000)
    
    # Compute diagnostics
    diagnostics = model.compute_diagnostics()
    print("Model Diagnostics:")
    print(f"R-hat max: {max(diagnostics['r_hat'].values()):.3f}")
    print(f"ESS min: {min(diagnostics['ess'].values()):.1f}")
    
    # Extract group parameters
    group_params = model.extract_group_params()
    print("\nGroup Intercepts:")
    for i in range(n_groups):
        print(f"Group {i+1}: {group_params['intercepts']['mean'][i]:.3f} "
              f"({group_params['intercepts']['hdi_lower'][i]:.3f}, "
              f"{group_params['intercepts']['hdi_upper'][i]:.3f})")
    
    # Plot group parameters
    model.plot_group_params('intercepts')
    
    # Compute ICC
    icc = model.compute_icc('intercepts')
    print(f"\nIntraclass Correlation Coefficient: {icc['icc']:.3f}")
    
    # Make predictions for a specific group
    X_new = np.random.randn(10, n_features)
    predictions = model.predict_for_group(X_new, group_id=0)
    print("\nPredictions for Group 1:")
    for i in range(len(predictions['mean'])):
        print(f"Sample {i+1}: {predictions['mean'][i]:.3f} ± {predictions['std'][i]:.3f}")
    
    plt.show()
