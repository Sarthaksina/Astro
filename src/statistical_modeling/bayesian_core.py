"""
Core Bayesian Module for the Cosmic Market Oracle.

This module implements basic Bayesian inference and model building functionality,
providing a foundation for more complex Bayesian models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Callable
import matplotlib.pyplot as plt
# import logging # Removed
from ..utils.logger import get_logger # Changed path for src
import pymc as pm
import arviz as az
import theano.tensor as tt
from scipy import stats

# Configure logging
# logging.basicConfig( # Removed
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = get_logger(__name__) # Changed


class BayesianModel:
    """
    Base class for Bayesian models.
    
    This class provides core functionality for building, sampling from,
    and analyzing Bayesian models.
    """
    
    def __init__(self, name: str = "bayesian_model"):
        """
        Initialize a Bayesian model.
        
        Args:
            name: Name of the model
        """
        self.name = name
        self.model = None
        self.trace = None
        self.summary = None
        self.diagnostics = {}
    
    def build_model(self, *args, **kwargs) -> pm.Model:
        """
        Build the Bayesian model.
        
        This is a placeholder method that should be overridden by subclasses.
        
        Returns:
            PyMC model
        """
        raise NotImplementedError("Subclasses must implement build_model method")
    
    def sample(self, 
              draws: int = 1000, 
              tune: int = 1000, 
              chains: int = 2,
              target_accept: float = 0.8,
              return_inferencedata: bool = True,
              **kwargs) -> Union[az.InferenceData, pm.MultiTrace]:
        """
        Sample from the posterior distribution of the model.
        
        Args:
            draws: Number of posterior samples to draw
            tune: Number of tuning samples
            chains: Number of MCMC chains
            target_accept: Target acceptance rate
            return_inferencedata: Whether to return InferenceData object
            **kwargs: Additional arguments to pass to pm.sample
            
        Returns:
            Trace of posterior samples
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model first.")
        
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=return_inferencedata,
                **kwargs
            )
        
        return self.trace
    
    def compute_diagnostics(self) -> Dict:
        """
        Compute diagnostics for the model.
        
        Returns:
            Dictionary of diagnostic metrics
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Compute summary statistics
        self.summary = az.summary(self.trace)
        
        # Compute diagnostics
        self.diagnostics = {
            'r_hat': self.summary['r_hat'].to_dict(),
            'ess': self.summary['ess_bulk'].to_dict(),
            'mcse': self.summary['mcse_mean'].to_dict()
        }
        
        # Check for convergence issues
        r_hat_max = self.summary['r_hat'].max()
        if r_hat_max > 1.1:
            logger.warning(f"R-hat values > 1.1 detected (max: {r_hat_max:.3f}). "
                          "This indicates potential convergence issues.")
        
        # Check for low effective sample size
        ess_min = self.summary['ess_bulk'].min()
        if ess_min < 100:
            logger.warning(f"Low effective sample size detected (min: {ess_min:.1f}). "
                          "Consider increasing the number of draws.")
        
        return self.diagnostics
    
    def plot_trace(self, 
                 var_names: Optional[List[str]] = None,
                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot trace and posterior distributions.
        
        Args:
            var_names: Variables to plot (if None, plots all)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Create trace plot
        ax = az.plot_trace(self.trace, var_names=var_names, figsize=figsize)
        
        # Get the figure
        fig = plt.gcf()
        
        return fig
    
    def plot_posterior(self, 
                     var_names: Optional[List[str]] = None,
                     figsize: Tuple[int, int] = (12, 8),
                     kind: str = 'kde') -> plt.Figure:
        """
        Plot posterior distributions.
        
        Args:
            var_names: Variables to plot (if None, plots all)
            figsize: Figure size
            kind: Type of plot ('kde' or 'hist')
            
        Returns:
            Matplotlib figure
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Create posterior plot
        ax = az.plot_posterior(self.trace, var_names=var_names, figsize=figsize, kind=kind)
        
        # Get the figure
        fig = plt.gcf()
        
        return fig
    
    def plot_forest(self, 
                  var_names: Optional[List[str]] = None,
                  figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot forest plot of posterior distributions.
        
        Args:
            var_names: Variables to plot (if None, plots all)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Create forest plot
        ax = az.plot_forest(self.trace, var_names=var_names, figsize=figsize)
        
        # Get the figure
        fig = plt.gcf()
        
        return fig
    
    def plot_pair(self, 
                var_names: Optional[List[str]] = None,
                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot pairwise relationships between parameters.
        
        Args:
            var_names: Variables to plot (if None, plots all)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Create pair plot
        ax = az.plot_pair(self.trace, var_names=var_names, figsize=figsize)
        
        # Get the figure
        fig = plt.gcf()
        
        return fig
    
    def get_posterior_predictive(self, 
                               samples: int = 1000,
                               var_names: Optional[List[str]] = None) -> az.InferenceData:
        """
        Generate posterior predictive samples.
        
        Args:
            samples: Number of posterior predictive samples
            var_names: Names of variables to sample (if None, samples all)
            
        Returns:
            Posterior predictive samples
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been built or sampled yet.")
        
        with self.model:
            posterior_predictive = pm.sample_posterior_predictive(
                self.trace, 
                var_names=var_names,
                random_seed=42,
                return_inferencedata=True
            )
        
        return posterior_predictive
    
    def plot_posterior_predictive(self, 
                                data: np.ndarray,
                                var_name: str,
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot posterior predictive check.
        
        Args:
            data: Observed data
            var_name: Name of the variable to check
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Generate posterior predictive samples
        posterior_predictive = self.get_posterior_predictive(var_names=[var_name])
        
        # Create posterior predictive plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram of observed data
        ax.hist(data, bins=30, alpha=0.5, label='Observed', density=True)
        
        # Extract posterior predictive samples
        pp_samples = posterior_predictive.posterior_predictive[var_name].values.flatten()
        
        # Plot histogram of posterior predictive samples
        ax.hist(pp_samples, bins=30, alpha=0.5, label='Posterior Predictive', density=True)
        
        # Add labels and legend
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior Predictive Check for {var_name}')
        ax.legend()
        
        return fig
    
    def compute_waic(self) -> Dict:
        """
        Compute Widely Applicable Information Criterion (WAIC).
        
        Returns:
            Dictionary with WAIC and related metrics
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been built or sampled yet.")
        
        # Compute WAIC
        waic = az.waic(self.trace)
        
        return {
            'waic': waic.waic,
            'p_waic': waic.p_waic,
            'waic_se': waic.waic_se
        }
    
    def compute_loo(self) -> Dict:
        """
        Compute Leave-One-Out Cross-Validation (LOO).
        
        Returns:
            Dictionary with LOO and related metrics
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been built or sampled yet.")
        
        # Compute LOO
        loo = az.loo(self.trace)
        
        return {
            'loo': loo.loo,
            'p_loo': loo.p_loo,
            'loo_se': loo.loo_se
        }
    
    def compare_models(self, other_models: List['BayesianModel'], 
                     method: str = 'waic') -> pd.DataFrame:
        """
        Compare this model with other models.
        
        Args:
            other_models: List of other Bayesian models to compare
            method: Comparison method ('waic' or 'loo')
            
        Returns:
            DataFrame with comparison results
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been built or sampled yet.")
        
        # Collect all traces
        traces = {self.name: self.trace}
        for model in other_models:
            if model.trace is None:
                raise ValueError(f"Model {model.name} has not been sampled yet.")
            traces[model.name] = model.trace
        
        # Compare models
        if method == 'waic':
            comparison = az.compare(traces, ic='waic')
        elif method == 'loo':
            comparison = az.compare(traces, ic='loo')
        else:
            raise ValueError(f"Unknown comparison method: {method}")
        
        return comparison


class LinearModel(BayesianModel):
    """
    Bayesian linear regression model.
    """
    
    def __init__(self, name: str = "linear_model"):
        """
        Initialize a Bayesian linear regression model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.coefficients = None
    
    def build_model(self, X: np.ndarray, y: np.ndarray, 
                  add_intercept: bool = True,
                  priors: Optional[Dict] = None) -> pm.Model:
        """
        Build a Bayesian linear regression model.
        
        Args:
            X: Predictor variables (n_samples, n_features)
            y: Target variable (n_samples,)
            add_intercept: Whether to add an intercept term
            priors: Dictionary of prior distributions for parameters
            
        Returns:
            PyMC model
        """
        # Store data
        self.X = X
        self.y = y
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Set default priors if not provided
        if priors is None:
            priors = {
                'intercept': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 10}},
                'coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 1}},
                'sigma': {'dist': 'HalfNormal', 'params': {'sigma': 1}}
            }
        
        # Create model
        self.model = pm.Model()
        
        with self.model:
            # Priors for intercept
            if add_intercept:
                intercept_prior = getattr(pm, priors['intercept']['dist'])
                intercept = intercept_prior(
                    'intercept', **priors['intercept']['params']
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
            if add_intercept:
                mu = intercept + pm.math.dot(X, coefficients)
            else:
                mu = pm.math.dot(X, coefficients)
            
            # Likelihood
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
        
        return self.model
    
    def predict(self, X_new: np.ndarray, add_intercept: bool = True) -> Dict[str, np.ndarray]:
        """
        Make predictions for new data.
        
        Args:
            X_new: New predictor variables (n_samples, n_features)
            add_intercept: Whether to add an intercept term
            
        Returns:
            Dictionary with predicted mean and standard deviation
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Get posterior samples
        if isinstance(self.trace, az.InferenceData):
            coefficients = self.trace.posterior['coefficients'].values
            if add_intercept:
                intercept = self.trace.posterior['intercept'].values
            sigma = self.trace.posterior['sigma'].values
        else:
            coefficients = self.trace['coefficients']
            if add_intercept:
                intercept = self.trace['intercept']
            sigma = self.trace['sigma']
        
        # Reshape coefficients for matrix multiplication
        coefficients = np.reshape(coefficients, (-1, X_new.shape[1]))
        
        # Compute predictions
        if add_intercept:
            intercept = np.reshape(intercept, (-1, 1))
            predictions = intercept + np.dot(coefficients, X_new.T)
        else:
            predictions = np.dot(coefficients, X_new.T)
        
        # Compute mean and standard deviation of predictions
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
        
        return {
            'mean': pred_mean,
            'std': pred_std
        }
    
    def plot_coefficients(self, 
                        feature_names: Optional[List[str]] = None,
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot posterior distributions of coefficients.
        
        Args:
            feature_names: Names of features (if None, uses default names)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Set default feature names if not provided
        if feature_names is None:
            feature_names = [f"X{i+1}" for i in range(self.X.shape[1])]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract coefficient samples
        if isinstance(self.trace, az.InferenceData):
            coefficients = self.trace.posterior['coefficients'].values
        else:
            coefficients = self.trace['coefficients']
        
        # Reshape coefficients
        coefficients = np.reshape(coefficients, (-1, self.X.shape[1]))
        
        # Compute mean and credible intervals
        coef_mean = np.mean(coefficients, axis=0)
        coef_hdi = az.hdi(coefficients)
        
        # Plot coefficients
        ax.errorbar(
            x=coef_mean,
            y=feature_names,
            xerr=np.abs(coef_hdi - coef_mean[:, np.newaxis]).T,
            fmt='o',
            capsize=5
        )
        
        # Add vertical line at zero
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add labels
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Feature')
        ax.set_title('Posterior Distributions of Coefficients')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def plot_predictions(self, 
                       X_new: np.ndarray, 
                       y_new: Optional[np.ndarray] = None,
                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot predictions for new data.
        
        Args:
            X_new: New predictor variables (n_samples, n_features)
            y_new: True values for new data (if available)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Make predictions
        predictions = self.predict(X_new)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by predicted mean for better visualization
        sort_idx = np.argsort(predictions['mean'])
        x = np.arange(len(predictions['mean']))
        
        # Plot predictions with uncertainty
        ax.errorbar(
            x=x,
            y=predictions['mean'][sort_idx],
            yerr=2 * predictions['std'][sort_idx],  # 2 standard deviations (95% CI)
            fmt='o',
            capsize=5,
            label='Predicted'
        )
        
        # Plot true values if available
        if y_new is not None:
            ax.scatter(
                x=x,
                y=y_new[sort_idx],
                color='red',
                marker='x',
                label='True'
            )
        
        # Add labels and legend
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.set_title('Predictions with Uncertainty')
        ax.legend()
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


class LogisticModel(BayesianModel):
    """
    Bayesian logistic regression model.
    """
    
    def __init__(self, name: str = "logistic_model"):
        """
        Initialize a Bayesian logistic regression model.
        
        Args:
            name: Name of the model
        """
        super().__init__(name=name)
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.coefficients = None
    
    def build_model(self, X: np.ndarray, y: np.ndarray, 
                  add_intercept: bool = True,
                  priors: Optional[Dict] = None) -> pm.Model:
        """
        Build a Bayesian logistic regression model.
        
        Args:
            X: Predictor variables (n_samples, n_features)
            y: Binary target variable (n_samples,)
            add_intercept: Whether to add an intercept term
            priors: Dictionary of prior distributions for parameters
            
        Returns:
            PyMC model
        """
        # Store data
        self.X = X
        self.y = y
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Set default priors if not provided
        if priors is None:
            priors = {
                'intercept': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 10}},
                'coefficients': {'dist': 'Normal', 'params': {'mu': 0, 'sigma': 1}}
            }
        
        # Create model
        self.model = pm.Model()
        
        with self.model:
            # Priors for intercept
            if add_intercept:
                intercept_prior = getattr(pm, priors['intercept']['dist'])
                intercept = intercept_prior(
                    'intercept', **priors['intercept']['params']
                )
            
            # Priors for coefficients
            coef_prior = getattr(pm, priors['coefficients']['dist'])
            coefficients = coef_prior(
                'coefficients', 
                **priors['coefficients']['params'],
                shape=n_features
            )
            
            # Linear predictor
            if add_intercept:
                logit_p = intercept + pm.math.dot(X, coefficients)
            else:
                logit_p = pm.math.dot(X, coefficients)
            
            # Likelihood
            likelihood = pm.Bernoulli('y', logit_p=logit_p, observed=y)
        
        return self.model
    
    def predict_proba(self, X_new: np.ndarray, add_intercept: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict probabilities for new data.
        
        Args:
            X_new: New predictor variables (n_samples, n_features)
            add_intercept: Whether to add an intercept term
            
        Returns:
            Dictionary with predicted probabilities
        """
        if self.trace is None:
            raise ValueError("No samples available. Call sample first.")
        
        # Get posterior samples
        if isinstance(self.trace, az.InferenceData):
            coefficients = self.trace.posterior['coefficients'].values
            if add_intercept:
                intercept = self.trace.posterior['intercept'].values
        else:
            coefficients = self.trace['coefficients']
            if add_intercept:
                intercept = self.trace['intercept']
        
        # Reshape coefficients for matrix multiplication
        coefficients = np.reshape(coefficients, (-1, X_new.shape[1]))
        
        # Compute logits
        if add_intercept:
            intercept = np.reshape(intercept, (-1, 1))
            logits = intercept + np.dot(coefficients, X_new.T)
        else:
            logits = np.dot(coefficients, X_new.T)
        
        # Convert to probabilities
        probas = 1 / (1 + np.exp(-logits))
        
        # Compute mean and standard deviation of probabilities
        proba_mean = np.mean(probas, axis=0)
        proba_std = np.std(probas, axis=0)
        
        return {
            'mean': proba_mean,
            'std': proba_std
        }
    
    def predict(self, X_new: np.ndarray, threshold: float = 0.5, 
              add_intercept: bool = True) -> np.ndarray:
        """
        Make binary predictions for new data.
        
        Args:
            X_new: New predictor variables (n_samples, n_features)
            threshold: Probability threshold for positive class
            add_intercept: Whether to add an intercept term
            
        Returns:
            Binary predictions
        """
        # Predict probabilities
        probas = self.predict_proba(X_new, add_intercept)
        
        # Convert to binary predictions
        predictions = (probas['mean'] > threshold).astype(int)
        
        return predictions
    
    def plot_roc_curve(self, 
                     X_test: np.ndarray, 
                     y_test: np.ndarray,
                     figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
        """
        Plot ROC curve for test data.
        
        Args:
            X_test: Test predictor variables (n_samples, n_features)
            y_test: Test target variable (n_samples,)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import roc_curve, auc
        
        # Predict probabilities
        probas = self.predict_proba(X_test)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, probas['mean'])
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve
        ax.plot(
            fpr, tpr, 
            lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})'
        )
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Add labels and legend
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        
        return fig


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    # Generate predictors
    X = np.random.randn(n_samples, n_features)
    
    # Generate true coefficients
    true_intercept = 2.5
    true_coefficients = np.array([1.0, -0.5, 0.2])
    
    # Generate target for linear regression
    y_linear = true_intercept + np.dot(X, true_coefficients) + 0.5 * np.random.randn(n_samples)
    
    # Generate target for logistic regression
    logits = true_intercept + np.dot(X, true_coefficients)
    p = 1 / (1 + np.exp(-logits))
    y_logistic = np.random.binomial(1, p)
    
    # Create and fit linear model
    linear_model = LinearModel(name="linear_example")
    linear_model.build_model(X, y_linear)
    linear_model.sample(draws=1000, tune=1000)
    
    # Compute diagnostics
    linear_diagnostics = linear_model.compute_diagnostics()
    print("Linear Model Diagnostics:")
    print(f"R-hat max: {max(linear_diagnostics['r_hat'].values()):.3f}")
    print(f"ESS min: {min(linear_diagnostics['ess'].values()):.1f}")
    
    # Plot results
    linear_model.plot_trace()
    linear_model.plot_coefficients(feature_names=["X1", "X2", "X3"])
    
    # Create and fit logistic model
    logistic_model = LogisticModel(name="logistic_example")
    logistic_model.build_model(X, y_logistic)
    logistic_model.sample(draws=1000, tune=1000)
    
    # Compute diagnostics
    logistic_diagnostics = logistic_model.compute_diagnostics()
    print("\nLogistic Model Diagnostics:")
    print(f"R-hat max: {max(logistic_diagnostics['r_hat'].values()):.3f}")
    print(f"ESS min: {min(logistic_diagnostics['ess'].values()):.1f}")
    
    # Plot results
    logistic_model.plot_trace()
    logistic_model.plot_coefficients(feature_names=["X1", "X2", "X3"])
    
    # Compare models
    linear_model2 = LinearModel(name="linear_alternative")
    linear_model2.build_model(X[:, :2], y_linear)  # Using only first 2 features
    linear_model2.sample(draws=1000, tune=1000)
    
    comparison = linear_model.compare_models([linear_model2])
    print("\nModel Comparison:")
    print(comparison)
    
    plt.show()
