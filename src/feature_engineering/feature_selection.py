# Cosmic Market Oracle - Feature Selection Module

"""
This module provides methods for selecting the most relevant features for prediction models.
It includes techniques for dimensionality reduction, feature importance ranking,
and elimination of redundant or irrelevant features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


class FeatureSelector:
    """Provides methods for selecting the most relevant features for prediction models."""
    
    def __init__(self):
        """
        Initialize the feature selector.
        """
        pass
    
    def select_by_correlation(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> pd.DataFrame:
        """
        Select features based on correlation with the target variable.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        # Calculate correlation with target
        corr_with_target = pd.DataFrame()
        corr_with_target['feature'] = X.columns
        corr_with_target['correlation'] = [abs(np.corrcoef(X[col], y)[0, 1]) for col in X.columns]
        
        # Sort by absolute correlation
        corr_with_target = corr_with_target.sort_values('correlation', ascending=False)
        
        # Select top k features
        selected_features = corr_with_target['feature'].head(k).tolist()
        
        return X[selected_features]
    
    def select_by_mutual_information(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> pd.DataFrame:
        """
        Select features based on mutual information with the target variable.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        # Apply mutual information feature selection
        selector = SelectKBest(mutual_info_regression, k=k)
        selector.fit(X, y)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        return X[selected_features]
    
    def select_by_random_forest(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> pd.DataFrame:
        """
        Select features based on importance scores from a Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        # Train a Random Forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = pd.DataFrame()
        importances['feature'] = X.columns
        importances['importance'] = rf.feature_importances_
        
        # Sort by importance
        importances = importances.sort_values('importance', ascending=False)
        
        # Select top k features
        selected_features = importances['feature'].head(k).tolist()
        
        return X[selected_features]
    
    def select_by_lasso(self, X: pd.DataFrame, y: pd.Series, alpha: float = 0.01) -> pd.DataFrame:
        """
        Select features using Lasso regularization.
        
        Args:
            X: Feature matrix
            y: Target variable
            alpha: Regularization strength
            
        Returns:
            DataFrame with selected features
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train a Lasso model
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_scaled, y)
        
        # Get feature coefficients
        coefficients = pd.DataFrame()
        coefficients['feature'] = X.columns
        coefficients['coefficient'] = lasso.coef_
        
        # Select features with non-zero coefficients
        selected_features = coefficients[coefficients['coefficient'] != 0]['feature'].tolist()
        
        return X[selected_features]
    
    def select_by_recursive_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                      estimator=None, k: int = 20) -> pd.DataFrame:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature matrix
            y: Target variable
            estimator: Estimator to use (defaults to RandomForestRegressor)
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        # Use Random Forest as default estimator
        if estimator is None:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Apply RFE
        selector = RFE(estimator=estimator, n_features_to_select=k, step=1)
        selector.fit(X, y)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        return X[selected_features]
    
    def select_by_pca(self, X: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """
        Transform features using Principal Component Analysis.
        
        Args:
            X: Feature matrix
            n_components: Number of components to keep
            
        Returns:
            DataFrame with PCA components
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create DataFrame with PCA components
        columns = [f'PC{i+1}' for i in range(n_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=columns, index=X.index)
        
        # Store explained variance ratio for reference
        self.pca_explained_variance_ = pca.explained_variance_ratio_
        self.pca_components_ = pca.components_
        
        return X_pca_df
    
    def remove_multicollinearity(self, X: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """
        Remove highly correlated features to reduce multicollinearity.
        
        Args:
            X: Feature matrix
            threshold: Correlation threshold for removal
            
        Returns:
            DataFrame with reduced multicollinearity
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Create upper triangle mask
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Drop highly correlated features
        X_reduced = X.drop(to_drop, axis=1)
        
        return X_reduced
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'random_forest', 
                       **kwargs) -> pd.DataFrame:
        """
        Select features using the specified method.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method
            **kwargs: Additional arguments for the selection method
            
        Returns:
            DataFrame with selected features
        """
        # Map method names to selection functions
        method_map = {
            'correlation': self.select_by_correlation,
            'mutual_info': self.select_by_mutual_information,
            'random_forest': self.select_by_random_forest,
            'lasso': self.select_by_lasso,
            'rfe': self.select_by_recursive_elimination,
            'pca': self.select_by_pca
        }
        
        # Check if method is valid
        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Available methods: {list(method_map.keys())}")
        
        # Apply the selected method
        if method == 'pca':
            return method_map[method](X, **kwargs)
        else:
            return method_map[method](X, y, **kwargs)
    
    def select_features_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                               methods: List[str] = ['correlation', 'random_forest', 'mutual_info'],
                               k: int = 20) -> pd.DataFrame:
        """
        Select features using an ensemble of methods and voting.
        
        Args:
            X: Feature matrix
            y: Target variable
            methods: List of feature selection methods to use
            k: Number of features to select from each method
            
        Returns:
            DataFrame with selected features
        """
        all_selected_features = []
        feature_votes = {feature: 0 for feature in X.columns}
        
        # Apply each method and collect selected features
        for method in methods:
            if method == 'correlation':
                selected = self.select_by_correlation(X, y, k=k).columns.tolist()
            elif method == 'mutual_info':
                selected = self.select_by_mutual_information(X, y, k=k).columns.tolist()
            elif method == 'random_forest':
                selected = self.select_by_random_forest(X, y, k=k).columns.tolist()
            elif method == 'lasso':
                selected = self.select_by_lasso(X, y).columns.tolist()
            elif method == 'rfe':
                selected = self.select_by_recursive_elimination(X, y, k=k).columns.tolist()
            else:
                continue
                
            all_selected_features.extend(selected)
            
            # Count votes for each feature
            for feature in selected:
                feature_votes[feature] += 1
        
        # Sort features by vote count
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        
        # Select top k features by votes
        final_features = [feature for feature, votes in sorted_features[:k] if votes > 0]
        
        return X[final_features]