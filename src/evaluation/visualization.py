#!/usr/bin/env python
# Cosmic Market Oracle - Evaluation Visualization

"""
Visualization tools for the Evaluation and Validation Framework.

This module provides tools for visualizing evaluation results, including
performance across market regimes, metric distributions, and robustness heatmaps.

This module consolidates visualization functionality from multiple sources:
- Basic performance visualization (predictions vs actuals, error distributions)
- Market regime comparison visualization (previously in regime_visualizer.py)
- Validation results visualization (cross-validation, walk-forward validation)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import logging
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class PerformanceVisualizer:
    """
    Visualizer for model performance metrics.
    
    This class provides methods for visualizing various performance metrics
    for prediction models, including time series plots, error distributions,
    and scatter plots of predicted vs. actual values.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8),
                style: str = "whitegrid",
                save_dir: Optional[str] = None):
        """
        Initialize the performance visualizer.
        
        Args:
            figsize: Figure size (width, height)
            style: Seaborn style
            save_dir: Directory to save figures (optional)
        """
        self.figsize = figsize
        self.style = style
        self.save_dir = save_dir
        
        # Create save directory if specified
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Set style
        sns.set_style(style)
        
    def plot_predictions_vs_actuals(self, predictions: np.ndarray,
                                  actuals: np.ndarray,
                                  dates: Optional[pd.DatetimeIndex] = None,
                                  title: str = "Predictions vs. Actuals",
                                  save_as: Optional[str] = None) -> Figure:
        """
        Plot predictions vs. actuals over time.
        
        Args:
            predictions: Predicted values
            actuals: Actual values
            dates: Date index (optional)
            title: Plot title
            save_as: Filename to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create x-axis values
        x = dates if dates is not None else np.arange(len(actuals))
        
        # Plot actuals and predictions
        ax.plot(x, actuals, label="Actual", color="blue", linewidth=2)
        ax.plot(x, predictions, label="Predicted", color="red", linewidth=1.5, alpha=0.8)
        
        # Add labels and title
        ax.set_xlabel("Date" if dates is not None else "Time")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if dates
        if dates is not None:
            plt.xticks(rotation=45)
            fig.tight_layout()
            
        # Save figure if specified
        if save_as and self.save_dir:
            save_path = os.path.join(self.save_dir, save_as)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")
            
        return fig
        
    def plot_error_distribution(self, predictions: np.ndarray,
                              actuals: np.ndarray,
                              title: str = "Error Distribution",
                              save_as: Optional[str] = None) -> Figure:
        """
        Plot distribution of prediction errors.
        
        Args:
            predictions: Predicted values
            actuals: Actual values
            title: Plot title
            save_as: Filename to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Calculate errors
        errors = actuals - predictions
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot histogram with KDE
        sns.histplot(errors, kde=True, ax=ax)
        
        # Add mean and std lines
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        ax.axvline(mean_error, color="red", linestyle="--", 
                 label=f"Mean: {mean_error:.4f}")
        ax.axvline(mean_error + std_error, color="green", linestyle=":", 
                 label=f"±Std: {std_error:.4f}")
        ax.axvline(mean_error - std_error, color="green", linestyle=":")
        
        # Add labels and title
        ax.set_xlabel("Error")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend()
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if specified
        if save_as and self.save_dir:
            save_path = os.path.join(self.save_dir, save_as)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")
            
        return fig
        
    def plot_scatter_predictions(self, predictions: np.ndarray,
                               actuals: np.ndarray,
                               title: str = "Predicted vs. Actual Values",
                               save_as: Optional[str] = None) -> Figure:
        """
        Create scatter plot of predicted vs. actual values.
        
        Args:
            predictions: Predicted values
            actuals: Actual values
            title: Plot title
            save_as: Filename to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot scatter
        ax.scatter(actuals, predictions, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(np.min(actuals), np.min(predictions))
        max_val = max(np.max(actuals), np.max(predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Prediction")
        
        # Add regression line
        if len(actuals) > 1:
            z = np.polyfit(actuals, predictions, 1)
            p = np.poly1d(z)
            ax.plot(np.array([min_val, max_val]), p(np.array([min_val, max_val])), 
                  'r-', label=f"Fit: y={z[0]:.3f}x+{z[1]:.3f}")
        
        # Add labels and title
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(title)
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if specified
        if save_as and self.save_dir:
            save_path = os.path.join(self.save_dir, save_as)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")
            
        return fig
        
    def plot_metric_over_time(self, dates: pd.DatetimeIndex,
                            metric_values: np.ndarray,
                            metric_name: str,
                            title: Optional[str] = None,
                            save_as: Optional[str] = None) -> Figure:
        """
        Plot a metric over time.
        
        Args:
            dates: Date index
            metric_values: Metric values
            metric_name: Name of the metric
            title: Plot title (optional)
            save_as: Filename to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot metric values
        ax.plot(dates, metric_values, marker='o', linestyle='-')
        
        # Add labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel(metric_name)
        ax.set_title(title or f"{metric_name} Over Time")
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_value = np.mean(metric_values)
        ax.axhline(mean_value, color="red", linestyle="--", 
                 label=f"Mean: {mean_value:.4f}")
        
        ax.legend()
        fig.tight_layout()
        
        # Save figure if specified
        if save_as and self.save_dir:
            save_path = os.path.join(self.save_dir, save_as)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")
            
        return fig
        
    def plot_validation_results(self, validation_results: Dict[str, Any],
                              metric_name: str = "rmse",
                              title: Optional[str] = None,
                              save_as: Optional[str] = None) -> Figure:
        """
        Plot validation results.
        
        Args:
            validation_results: Validation results dictionary
            metric_name: Name of the metric to plot
            title: Plot title (optional)
            save_as: Filename to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Check validation method
        method = validation_results.get("method")
        
        if method == "walk_forward":
            return self._plot_walk_forward_results(validation_results, metric_name, title, save_as)
        elif method == "cross_market":
            return self._plot_cross_market_results(validation_results, metric_name, title, save_as)
        elif method == "temporal":
            return self._plot_temporal_results(validation_results, metric_name, title, save_as)
        else:
            logger.warning(f"Unknown validation method: {method}")
            return None
            
    def _plot_walk_forward_results(self, validation_results: Dict[str, Any],
                                 metric_name: str,
                                 title: Optional[str] = None,
                                 save_as: Optional[str] = None) -> Figure:
        """
        Plot walk-forward validation results.
        
        Args:
            validation_results: Validation results dictionary
            metric_name: Name of the metric to plot
            title: Plot title (optional)
            save_as: Filename to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Extract fold results
        fold_results = validation_results.get("fold_results", [])
        
        if not fold_results:
            logger.warning("No fold results found.")
            return None
            
        # Extract metric values and dates
        metric_values = []
        fold_numbers = []
        test_starts = []
        
        for fold in fold_results:
            if metric_name in fold.get("metrics", {}):
                metric_values.append(fold["metrics"][metric_name])
                fold_numbers.append(fold["fold"])
                test_starts.append(fold["test_start"])
                
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot metric values
        ax.plot(fold_numbers, metric_values, marker='o', linestyle='-')
        
        # Add labels and title
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric_name)
        ax.set_title(title or f"{metric_name} Across Validation Folds")
        
        # Add fold dates as secondary x-axis
        if test_starts:
            ax2 = ax.twiny()
            ax2.set_xticks(fold_numbers)
            ax2.set_xticklabels([str(d.date()) for d in test_starts], rotation=45)
            ax2.set_xlabel("Test Start Date")
            
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_value = np.mean(metric_values)
        ax.axhline(mean_value, color="red", linestyle="--", 
                 label=f"Mean: {mean_value:.4f}")
        
        ax.legend()
        fig.tight_layout()
        
        # Save figure if specified
        if save_as and self.save_dir:
            save_path = os.path.join(self.save_dir, save_as)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")
            
        return fig


class RegimeVisualizer:
    """
    Visualizer for comparing model performance across market regimes.
    
    This class provides methods for visualizing how prediction models perform
    across different market regimes, including bull, bear, sideways, and volatile.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8),
                style: str = "whitegrid",
                save_dir: Optional[str] = None):
        """
        Initialize the regime comparison visualizer.
        
        Args:
            figsize: Figure size (width, height)
            style: Seaborn style
            save_dir: Directory to save figures (optional)
        """
        self.figsize = figsize
        self.style = style
        self.save_dir = save_dir
        
        # Create save directory if specified
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Set style
        sns.set_style(style)
        
        # Define regime colors
        self.regime_colors = {
            "bull": "green",
            "bear": "red",
            "sideways": "blue",
            "volatile": "purple",
            "unknown": "gray"
        }
        
    def plot_regime_metrics(self, regime_metrics: Dict[str, Dict[str, float]],
                          metric_names: List[str] = None,
                          title: str = "Performance Across Market Regimes",
                          save_as: Optional[str] = None) -> Figure:
        """
        Plot metrics across different market regimes.
        
        Args:
            regime_metrics: Dictionary of regime-specific metrics
            metric_names: List of metric names to plot (optional)
            title: Plot title
            save_as: Filename to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        if not regime_metrics:
            logger.warning("No regime metrics provided.")
            return None
            
        # Default metrics if not specified
        if metric_names is None:
            # Use first regime's metrics as reference
            first_regime = next(iter(regime_metrics))
            metric_names = list(regime_metrics[first_regime].keys())
            
        # Create dataframe for plotting
        data = []
        
        for regime, metrics in regime_metrics.items():
            for metric in metric_names:
                if metric in metrics:
                    data.append({
                        "Regime": regime,
                        "Metric": metric,
                        "Value": metrics[metric]
                    })
                    
        if not data:
            logger.warning("No valid metrics found for plotting.")
            return None
            
        df = pd.DataFrame(data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot grouped bar chart
        sns.barplot(x="Metric", y="Value", hue="Regime", data=df, ax=ax,
                  palette={r: self.regime_colors.get(r, "gray") for r in df["Regime"].unique()})
        
        # Add labels and title
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title(title)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if specified
        if save_as and self.save_dir:
            save_path = os.path.join(self.save_dir, save_as)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")
            
        return fig
    
    def plot_regime_performance_radar(self, regime_metrics: Dict[str, Dict[str, float]],
                                    metric_names: List[str] = None,
                                    title: str = "Regime Performance Radar",
                                    save_as: Optional[str] = None) -> Figure:
        """
        Plot radar chart of performance across regimes.
        
        Args:
            regime_metrics: Dictionary of regime-specific metrics
            metric_names: List of metric names to plot (optional)
            title: Plot title
            save_as: Filename to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        if not regime_metrics:
            logger.warning("No regime metrics provided.")
            return None
            
        # Default metrics if not specified
        if metric_names is None:
            # Use first regime's metrics as reference
            first_regime = next(iter(regime_metrics))
            metric_names = list(regime_metrics[first_regime].keys())
            
        # Create figure
        fig = plt.figure(figsize=self.figsize)
        
        # Number of variables
        N = len(metric_names)
        
        # Create angles for radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create subplot with polar projection
        ax = fig.add_subplot(111, polar=True)
        
        # Add metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        
        # Plot each regime
        for regime, metrics in regime_metrics.items():
            # Get values for this regime
            values = [metrics.get(metric, 0) for metric in metric_names]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=regime,
                  color=self.regime_colors.get(regime, "gray"))
            ax.fill(angles, values, alpha=0.1, color=self.regime_colors.get(regime, "gray"))
            
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        plt.title(title)
        
        # Save figure if specified
        if save_as and self.save_dir:
            save_path = os.path.join(self.save_dir, save_as)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")
            
        return fig
    
    def plot_regime_transitions(self, predictions: np.ndarray,
                              actuals: np.ndarray,
                              regimes: np.ndarray,
                              dates: Optional[pd.DatetimeIndex] = None,
                              title: str = "Performance During Regime Transitions",
                              save_as: Optional[str] = None) -> Figure:
        """
        Plot performance during regime transitions.
        
        Args:
            predictions: Predicted values
            actuals: Actual values
            regimes: Array of regime labels
            dates: Date index (optional)
            title: Plot title
            save_as: Filename to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create x-axis values
        x = dates if dates is not None else np.arange(len(actuals))
        
        # Calculate errors
        errors = np.abs(actuals - predictions)
        
        # Plot errors
        ax.plot(x, errors, color="black", alpha=0.7, label="Absolute Error")
        
        # Find regime transitions
        transitions = np.where(np.roll(regimes, 1) != regimes)[0]
        
        # Plot vertical lines at transitions
        for t in transitions:
            if t > 0 and t < len(x):
                ax.axvline(x[t], color="red", linestyle="--", alpha=0.5)
                
                # Add regime label
                if t < len(regimes):
                    regime = regimes[t]
                    ax.text(x[t], ax.get_ylim()[1] * 0.9, regime,
                          rotation=90, verticalalignment="top",
                          color=self.regime_colors.get(regime, "gray"))
        
        # Color background by regime
        prev_t = 0
        for t in np.append(transitions, len(x)):
            if t > 0:
                regime = regimes[prev_t:t][0] if prev_t < len(regimes) else "unknown"
                ax.axvspan(x[prev_t], x[min(t, len(x)-1)],
                         alpha=0.1, color=self.regime_colors.get(regime, "gray"))
                prev_t = t
        
        # Add labels and title
        ax.set_xlabel("Date" if dates is not None else "Time")
        ax.set_ylabel("Absolute Error")
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        # Rotate x-axis labels if dates
        if dates is not None:
            plt.xticks(rotation=45)
            
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if specified
        if save_as and self.save_dir:
            save_path = os.path.join(self.save_dir, save_as)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")
            
        return fig
    
    def plot_regime_error_distributions(self, predictions: np.ndarray,
                                      actuals: np.ndarray,
                                      regimes: np.ndarray,
                                      title: str = "Error Distributions by Regime",
                                      save_as: Optional[str] = None) -> Figure:
        """
        Plot error distributions for different regimes.
        
        Args:
            predictions: Predicted values
            actuals: Actual values
            regimes: Array of regime labels
            title: Plot title
            save_as: Filename to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Calculate errors
        errors = actuals - predictions
        
        # Get unique regimes
        unique_regimes = np.unique(regimes)
        n_regimes = len(unique_regimes)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, n_regimes, figsize=(self.figsize[0] * n_regimes / 2, self.figsize[1]))
        
        # Handle case with only one regime
        if n_regimes == 1:
            axes = [axes]
            
        # Plot error distribution for each regime
        for i, regime in enumerate(unique_regimes):
            # Get errors for this regime
            regime_errors = errors[regimes == regime]
            
            if len(regime_errors) > 0:
                # Plot distribution
                sns.histplot(regime_errors, kde=True, ax=axes[i], 
                           color=self.regime_colors.get(regime, "gray"))
                
                # Add mean and std lines
                mean_error = np.mean(regime_errors)
                std_error = np.std(regime_errors)
                
                axes[i].axvline(mean_error, color="red", linestyle="--", 
                              label=f"Mean: {mean_error:.4f}")
                axes[i].axvline(mean_error + std_error, color="green", linestyle=":", 
                              label=f"±Std: {std_error:.4f}")
                axes[i].axvline(mean_error - std_error, color="green", linestyle=":")
                
                # Add labels
                axes[i].set_title(f"{regime} Regime")
                axes[i].set_xlabel("Error")
                
                if i == 0:
                    axes[i].set_ylabel("Frequency")
                    
                axes[i].legend()
                
        # Add overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        
        # Save figure if specified
        if save_as and self.save_dir:
            save_path = os.path.join(self.save_dir, save_as)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")
            
        return fig
        
    def plot_regime_metric_heatmap(self, regime_metrics: Dict[str, Dict[str, float]],
                                 metric_names: List[str] = None,
                                 title: str = "Metric Heatmap by Regime",
                                 save_as: Optional[str] = None) -> Figure:
        """
        Plot heatmap of metrics across regimes.
        
        Args:
            regime_metrics: Dictionary of regime-specific metrics
            metric_names: List of metric names to plot (optional)
            title: Plot title
            save_as: Filename to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        if not regime_metrics:
            logger.warning("No regime metrics provided.")
            return None
            
        # Default metrics if not specified
        if metric_names is None:
            # Use first regime's metrics as reference
            first_regime = next(iter(regime_metrics))
            metric_names = list(regime_metrics[first_regime].keys())
            
        # Create data matrix for heatmap
        regimes = list(regime_metrics.keys())
        data = np.zeros((len(regimes), len(metric_names)))
        
        for i, regime in enumerate(regimes):
            for j, metric in enumerate(metric_names):
                data[i, j] = regime_metrics[regime].get(metric, np.nan)
                
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot heatmap
        sns.heatmap(data, annot=True, fmt=".3f", cmap="viridis",
                  xticklabels=metric_names, yticklabels=regimes, ax=ax)
        
        # Add labels and title
        ax.set_title(title)
        ax.set_xlabel("Metric")
        ax.set_ylabel("Regime")
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure if specified
        if save_as and self.save_dir:
            save_path = os.path.join(self.save_dir, save_as)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")
            
        return fig
