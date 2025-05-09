# Cosmic Market Oracle - Feature Importance Module

"""
This module provides methods for analyzing and visualizing the importance of features
in prediction models. It includes techniques for interpreting model-specific feature
importance, permutation importance, and SHAP values for explainable AI.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import shap


class FeatureImportanceAnalyzer:
    """Analyzes and visualizes feature importance in prediction models."""
    
    def __init__(self):
        """
        Initialize the feature importance analyzer.
        """
        pass
    
    def get_model_feature_importance(self, model: Any, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract feature importance from a trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            X: Feature matrix used for training
            
        Returns:
            DataFrame with feature importance scores
        """
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_permutation_importance(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                                 n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
        """
        Calculate permutation feature importance.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            n_repeats: Number of times to permute each feature
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with permutation importance scores
        """
        # Calculate permutation importance
        perm_importance = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
        
        # Create DataFrame with permutation importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        })
        
        # Sort by mean importance
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    def get_shap_importance(self, model: Any, X: pd.DataFrame, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate SHAP values for feature importance.
        
        Args:
            model: Trained model
            X: Feature matrix
            sample_size: Optional number of samples to use for SHAP calculation
            
        Returns:
            DataFrame with SHAP importance scores
        """
        # Sample data if requested
        if sample_size is not None and sample_size < len(X):
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X
        
        # Create explainer based on model type
        if isinstance(model, RandomForestRegressor):
            explainer = shap.TreeExplainer(model)
        else:
            # Default to KernelExplainer for other model types
            explainer = shap.KernelExplainer(model.predict, X_sample)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different return types from different explainers
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Take first output for multi-output models
        
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame with SHAP importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mean_abs_shap
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Store SHAP values for later visualization
        self.shap_values_ = shap_values
        self.shap_data_ = X_sample
        
        return importance_df
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, title: str = 'Feature Importance',
                              top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot feature importance scores.
        
        Args:
            importance_df: DataFrame with feature importance scores
            title: Plot title
            top_n: Number of top features to show
            figsize: Figure size (width, height)
        """
        # Select top N features
        plot_df = importance_df.head(top_n)
        
        # Create horizontal bar plot
        plt.figure(figsize=figsize)
        plt.barh(plot_df['feature'], plot_df['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(title)
        plt.gca().invert_yaxis()  # Display highest importance at the top
        plt.tight_layout()
        plt.show()
    
    def plot_permutation_importance(self, importance_df: pd.DataFrame, 
                                  top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot permutation importance scores with error bars.
        
        Args:
            importance_df: DataFrame with permutation importance scores
            top_n: Number of top features to show
            figsize: Figure size (width, height)
        """
        # Select top N features
        plot_df = importance_df.head(top_n)
        
        # Create horizontal bar plot with error bars
        plt.figure(figsize=figsize)
        plt.barh(plot_df['feature'], plot_df['importance_mean'], 
                xerr=plot_df['importance_std'], capsize=5)
        plt.xlabel('Permutation Importance')
        plt.ylabel('Feature')
        plt.title('Permutation Feature Importance')
        plt.gca().invert_yaxis()  # Display highest importance at the top
        plt.tight_layout()
        plt.show()
    
    def plot_shap_summary(self, top_n: Optional[int] = None) -> None:
        """
        Plot SHAP summary plot for feature importance.
        
        Args:
            top_n: Optional number of top features to show
        """
        # Check if SHAP values have been calculated
        if not hasattr(self, 'shap_values_') or not hasattr(self, 'shap_data_'):
            raise ValueError("SHAP values not calculated. Run get_shap_importance() first.")
        
        # Select top N features if specified
        if top_n is not None:
            # Calculate mean absolute SHAP value for each feature
            mean_abs_shap = np.abs(self.shap_values_).mean(axis=0)
            
            # Get indices of top N features
            top_indices = np.argsort(mean_abs_shap)[-top_n:]
            
            # Select top features
            shap_values = self.shap_values_[:, top_indices]
            features = self.shap_data_.iloc[:, top_indices]
        else:
            shap_values = self.shap_values_
            features = self.shap_data_
        
        # Create SHAP summary plot
        shap.summary_plot(shap_values, features)
    
    def plot_shap_dependence(self, feature_name: str) -> None:
        """
        Plot SHAP dependence plot for a specific feature.
        
        Args:
            feature_name: Name of the feature to plot
        """
        # Check if SHAP values have been calculated
        if not hasattr(self, 'shap_values_') or not hasattr(self, 'shap_data_'):
            raise ValueError("SHAP values not calculated. Run get_shap_importance() first.")
        
        # Check if feature exists
        if feature_name not in self.shap_data_.columns:
            raise ValueError(f"Feature '{feature_name}' not found in data")
        
        # Get feature index
        feature_idx = list(self.shap_data_.columns).index(feature_name)
        
        # Create SHAP dependence plot
        shap.dependence_plot(feature_idx, self.shap_values_, self.shap_data_)
    
    def compare_feature_importance_methods(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                         top_n: int = 20, figsize: Tuple[int, int] = (15, 10)) -> pd.DataFrame:
        """
        Compare different feature importance methods.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            top_n: Number of top features to show
            figsize: Figure size (width, height)
            
        Returns:
            DataFrame with combined importance scores
        """
        # Calculate model-specific importance if available
        try:
            model_importance = self.get_model_feature_importance(model, X)
            has_model_importance = True
        except:
            has_model_importance = False
        
        # Calculate permutation importance
        perm_importance = self.get_permutation_importance(model, X, y)
        
        # Calculate SHAP importance (with sample for efficiency)
        sample_size = min(1000, len(X))
        shap_importance = self.get_shap_importance(model, X, sample_size=sample_size)
        
        # Create combined DataFrame
        combined = pd.DataFrame({'feature': X.columns})
        
        if has_model_importance:
            combined = combined.merge(model_importance[['feature', 'importance']], 
                                     on='feature', how='left', suffixes=('', '_model'))
            combined.rename(columns={'importance': 'model_importance'}, inplace=True)
        
        combined = combined.merge(perm_importance[['feature', 'importance_mean']], 
                                 on='feature', how='left')
        combined.rename(columns={'importance_mean': 'permutation_importance'}, inplace=True)
        
        combined = combined.merge(shap_importance[['feature', 'importance']], 
                                 on='feature', how='left', suffixes=('', '_shap'))
        combined.rename(columns={'importance': 'shap_importance'}, inplace=True)
        
        # Calculate average rank across methods
        rank_cols = []
        
        if has_model_importance:
            combined['rank_model'] = combined['model_importance'].rank(ascending=False)
            rank_cols.append('rank_model')
            
        combined['rank_perm'] = combined['permutation_importance'].rank(ascending=False)
        combined['rank_shap'] = combined['shap_importance'].rank(ascending=False)
        rank_cols.extend(['rank_perm', 'rank_shap'])
        
        combined['avg_rank'] = combined[rank_cols].mean(axis=1)
        combined = combined.sort_values('avg_rank')
        
        # Plot comparison
        plt.figure(figsize=figsize)
        
        # Select top N features by average rank
        plot_df = combined.head(top_n)
        
        # Set up plot
        width = 0.25
        x = np.arange(len(plot_df))
        
        # Create grouped bar chart
        ax = plt.subplot(111)
        bars = []
        
        if has_model_importance:
            bars.append(ax.barh(x - width, plot_df['model_importance'], width, label='Model Importance'))
            
        bars.append(ax.barh(x, plot_df['permutation_importance'], width, label='Permutation Importance'))
        bars.append(ax.barh(x + width, plot_df['shap_importance'], width, label='SHAP Importance'))
        
        # Add labels and legend
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importance Comparison')
        ax.set_yticks(x)
        ax.set_yticklabels(plot_df['feature'])
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return combined
    
    def analyze_feature_importance_by_category(self, importance_df: pd.DataFrame, 
                                             category_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Analyze feature importance grouped by category.
        
        Args:
            importance_df: DataFrame with feature importance scores
            category_mapping: Dictionary mapping feature names to categories
            
        Returns:
            DataFrame with importance by category
        """
        # Add category column
        importance_df['category'] = importance_df['feature'].map(lambda x: next(
            (cat for feat_pattern, cat in category_mapping.items() if feat_pattern in x), 'Other'))
        
        # Group by category and sum importance
        category_importance = importance_df.groupby('category')['importance'].sum().reset_index()
        
        # Sort by importance
        category_importance = category_importance.sort_values('importance', ascending=False)
        
        return category_importance
    
    def plot_category_importance(self, category_importance: pd.DataFrame, 
                               figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot importance by feature category.
        
        Args:
            category_importance: DataFrame with importance by category
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        plt.pie(category_importance['importance'], labels=category_importance['category'], 
               autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Feature Importance by Category')
        plt.tight_layout()
        plt.show()