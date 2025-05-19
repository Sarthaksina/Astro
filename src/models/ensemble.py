# Cosmic Market Oracle - Ensemble Models

"""
This module implements various ensemble models for market prediction,
specializing in gradient boosting methods with customizations for
astrological and financial data.

Implemented models include:
- XGBoost with custom split criteria for cyclical data
- LightGBM with specialized feature interaction detection
- CatBoost for handling categorical astrological variables
- Stacked models with astrological segmentation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import required libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold

from src.models.advanced_architectures import TransformerModel, GNNModel, HybridCNNTransformerModel, NeuralODEModel


class EnsembleModel(nn.Module):
    """
    Ensemble model combining multiple neural network architectures.
    
    This model combines predictions from multiple model architectures
    using a weighted average or learned aggregation.
    
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        model_configs (Dict): Configuration for each model type
        aggregation (str): Aggregation method ('weighted' or 'learned')
        weights (List[float]): Optional weights for weighted aggregation
    """
    def __init__(self, input_dim: int, output_dim: int, model_configs: Dict[str, Dict],
                 aggregation: str = 'weighted', weights: Optional[List[float]] = None):
        super(EnsembleModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregation = aggregation
        
        # Create models
        self.models = nn.ModuleDict()
        for model_name, config in model_configs.items():
            if model_name == 'transformer':
                self.models[model_name] = TransformerModel(
                    input_dim=input_dim,
                    d_model=config.get('d_model', 256),
                    nhead=config.get('nhead', 8),
                    num_layers=config.get('num_layers', 6),
                    dim_feedforward=config.get('dim_feedforward', 1024),
                    output_dim=output_dim,
                    dropout=config.get('dropout', 0.1)
                )
            elif model_name == 'gnn':
                self.models[model_name] = GNNModel(
                    input_dim=input_dim,
                    hidden_dim=config.get('hidden_dim', 256),
                    output_dim=output_dim,
                    num_layers=config.get('num_layers', 3),
                    dropout=config.get('dropout', 0.1)
                )
            elif model_name == 'hybrid':
                self.models[model_name] = HybridCNNTransformerModel(
                    input_dim=input_dim,
                    cnn_channels=config.get('cnn_channels', [64, 128, 256]),
                    d_model=config.get('d_model', 256),
                    nhead=config.get('nhead', 8),
                    num_layers=config.get('num_layers', 4),
                    output_dim=output_dim,
                    dropout=config.get('dropout', 0.1)
                )
            elif model_name == 'neural_ode':
                self.models[model_name] = NeuralODEModel(
                    input_dim=input_dim,
                    hidden_dim=config.get('hidden_dim', 128),
                    output_dim=output_dim,
                    augment_dim=config.get('augment_dim', 5),
                    time_steps=config.get('time_steps', 10)
                )
        
        # Set up aggregation
        if aggregation == 'weighted':
            if weights is None:
                # Equal weights
                self.weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models), requires_grad=False)
            else:
                # Normalize weights
                weights_tensor = torch.tensor(weights, dtype=torch.float32)
                self.weights = nn.Parameter(weights_tensor / weights_tensor.sum(), requires_grad=False)
        else:  # learned
            # Learnable weights
            self.weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))
            
            # Attention-based aggregation
            self.attention = nn.Sequential(
                nn.Linear(output_dim * len(self.models), 128),
                nn.ReLU(),
                nn.Linear(128, len(self.models)),
                nn.Softmax(dim=1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ensemble model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Get predictions from each model
        outputs = []
        for model in self.models.values():
            outputs.append(model(x))
        
        # Aggregate predictions
        if self.aggregation == 'weighted':
            # Simple weighted average
            ensemble_output = torch.zeros_like(outputs[0])
            for i, output in enumerate(outputs):
                ensemble_output += output * self.weights[i]
        else:  # learned
            # Concatenate outputs
            concat_outputs = torch.cat(outputs, dim=1)
            
            # Calculate attention weights
            attention_weights = self.attention(concat_outputs)
            
            # Weighted sum
            ensemble_output = torch.zeros_like(outputs[0])
            for i, output in enumerate(outputs):
                ensemble_output += output * attention_weights[:, i].unsqueeze(1)
        
        return ensemble_output
    
    def get_model_weights(self) -> Dict[str, float]:
        """
        Get the weights for each model in the ensemble.
        
        Returns:
            Dictionary mapping model names to weights
        """
        if self.aggregation == 'weighted':
            return {name: self.weights[i].item() for i, name in enumerate(self.models.keys())}
        else:
            # For learned aggregation, we can't directly return weights
            # as they depend on the input
            return {name: float('nan') for name in self.models.keys()}


class CyclicalXGBoostModel(BaseEstimator, RegressorMixin):
    """
    XGBoost model with custom split criteria for cyclical data.
    
    This model extends XGBoost to better handle cyclical features like
    planetary positions and astrological angles.
    
    Args:
        cyclical_features (List[str]): List of cyclical feature names
        xgb_params (Dict): XGBoost parameters
        num_boost_round (int): Number of boosting rounds
    """
    def __init__(self, cyclical_features: List[str] = None, 
                 xgb_params: Dict = None, num_boost_round: int = 100):
        self.cyclical_features = cyclical_features or []
        self.xgb_params = xgb_params or {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1
        }
        self.num_boost_round = num_boost_round
        self.model = None
        self.feature_names = None
    
    def _preprocess_cyclical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform cyclical features into sine and cosine components.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features with cyclical features converted to sin/cos
        """
        X_processed = X.copy()
        
        for feature in self.cyclical_features:
            if feature in X.columns:
                # Normalize to [0, 2π]
                normalized = (X[feature] * 2 * np.pi / 360.0) if feature.endswith('_deg') else (X[feature] * 2 * np.pi)
                
                # Create sine and cosine features
                X_processed[f"{feature}_sin"] = np.sin(normalized)
                X_processed[f"{feature}_cos"] = np.cos(normalized)
                
                # Drop original feature
                X_processed = X_processed.drop(feature, axis=1)
        
        return X_processed
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'CyclicalXGBoostModel':
        """
        Fit the model to the data.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Fitted model
        """
        # Preprocess cyclical features
        X_processed = self._preprocess_cyclical_features(X)
        self.feature_names = X_processed.columns.tolist()
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_processed, label=y, feature_names=self.feature_names)
        
        # Train model
        self.model = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=self.num_boost_round
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        # Preprocess cyclical features
        X_processed = self._preprocess_cyclical_features(X)
        
        # Create DMatrix
        dtest = xgb.DMatrix(X_processed, feature_names=self.feature_names)
        
        # Make predictions
        return self.model.predict(dtest)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        importance = self.model.get_score(importance_type='gain')
        return pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        }).sort_values('Importance', ascending=False)


class AstroLightGBM(BaseEstimator, RegressorMixin):
    """
    LightGBM model with specialized feature interaction detection for astrological data.
    
    This model is designed to capture complex interactions between planetary positions
    and market movements.
    
    Args:
        lgb_params (Dict): LightGBM parameters
        num_boost_round (int): Number of boosting rounds
        interaction_constraints (List[List[int]]): Feature interaction constraints
    """
    def __init__(self, lgb_params: Dict = None, num_boost_round: int = 100,
                 interaction_constraints: List[List[int]] = None):
        self.lgb_params = lgb_params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        self.num_boost_round = num_boost_round
        self.interaction_constraints = interaction_constraints
        self.model = None
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'AstroLightGBM':
        """
        Fit the model to the data.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Fitted model
        """
        self.feature_names = X.columns.tolist()
        
        # Create dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Add interaction constraints if provided
        params = self.lgb_params.copy()
        if self.interaction_constraints:
            params['interaction_constraints'] = self.interaction_constraints
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.num_boost_round
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        importance = self.model.feature_importance(importance_type='gain')
        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)


class AstroCatBoost(BaseEstimator, RegressorMixin):
    """
    CatBoost model for handling categorical astrological variables.
    
    This model is particularly suited for handling categorical features like
    zodiac signs, nakshatras, and other astrological classifications.
    
    Args:
        cat_features (List[str]): List of categorical feature names
        cb_params (Dict): CatBoost parameters
        num_boost_round (int): Number of boosting rounds
    """
    def __init__(self, cat_features: List[str] = None, 
                 cb_params: Dict = None, num_boost_round: int = 100):
        self.cat_features = cat_features or []
        self.cb_params = cb_params or {
            'loss_function': 'RMSE',
            'depth': 6,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False
        }
        self.num_boost_round = num_boost_round
        self.model = None
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'AstroCatBoost':
        """
        Fit the model to the data.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Fitted model
        """
        self.feature_names = X.columns.tolist()
        
        # Identify categorical feature indices
        cat_feature_indices = [X.columns.get_loc(feat) for feat in self.cat_features if feat in X.columns]
        
        # Create CatBoost Pool
        train_data = cb.Pool(X, label=y, cat_features=cat_feature_indices)
        
        # Initialize and train model
        self.model = cb.CatBoostRegressor(**self.cb_params, iterations=self.num_boost_round)
        self.model.fit(train_data, verbose=False)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        importance = self.model.get_feature_importance()
        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)


class AstrologicalStackedEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacked ensemble model with astrological segmentation.
    
    This model creates specialized sub-models for different astrological
    conditions and combines their predictions using a meta-learner.
    
    Args:
        base_models (List[Tuple[str, BaseEstimator]]): List of base models
        meta_model (BaseEstimator): Meta-learner model
        segmentation_func (Callable): Function to segment data based on astrological conditions
        cv (int): Number of cross-validation folds
    """
    def __init__(self, base_models: List[Tuple[str, BaseEstimator]], 
                 meta_model: BaseEstimator, 
                 segmentation_func: Callable[[pd.DataFrame], np.ndarray] = None,
                 cv: int = 5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.segmentation_func = segmentation_func
        self.cv = cv
        self.stacking_model = None
        self.segments = None
        self.segment_models = {}
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'AstrologicalStackedEnsemble':
        """
        Fit the model to the data.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Fitted model
        """
        # If segmentation function is provided, create segment-specific models
        if self.segmentation_func is not None:
            self.segments = self.segmentation_func(X)
            unique_segments = np.unique(self.segments)
            
            # Train a separate stacking model for each segment
            for segment in unique_segments:
                segment_mask = (self.segments == segment)
                X_segment = X[segment_mask]
                y_segment = y[segment_mask]
                
                if len(X_segment) > 0:
                    # Create and fit stacking model for this segment
                    stacking_model = StackingRegressor(
                        estimators=self.base_models,
                        final_estimator=self.meta_model,
                        cv=self.cv,
                        n_jobs=-1
                    )
                    stacking_model.fit(X_segment, y_segment)
                    self.segment_models[segment] = stacking_model
        else:
            # Create and fit a single stacking model
            self.stacking_model = StackingRegressor(
                estimators=self.base_models,
                final_estimator=self.meta_model,
                cv=self.cv,
                n_jobs=-1
            )
            self.stacking_model.fit(X, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if self.segmentation_func is not None:
            # Segment the data
            segments = self.segmentation_func(X)
            predictions = np.zeros(len(X))
            
            # Make predictions for each segment
            for segment, model in self.segment_models.items():
                segment_mask = (segments == segment)
                X_segment = X[segment_mask]
                
                if len(X_segment) > 0:
                    predictions[segment_mask] = model.predict(X_segment)
            
            return predictions
        else:
            # Use the single stacking model
            return self.stacking_model.predict(X)


class BayesianModelAveraging(BaseEstimator, RegressorMixin):
    """
    Bayesian Model Averaging with astrological priors.
    
    This model combines predictions from multiple models using Bayesian averaging,
    with the ability to incorporate prior knowledge about astrological conditions.
    
    Args:
        models (List[BaseEstimator]): List of models to average
        prior_weights (List[float]): Prior weights for each model
    """
    def __init__(self, models: List[BaseEstimator], prior_weights: List[float] = None):
        self.models = models
        self.prior_weights = prior_weights
        self.posterior_weights = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'BayesianModelAveraging':
        """
        Fit the model to the data and compute posterior weights.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Fitted model
        """
        n_models = len(self.models)
        
        # Fit each model
        for model in self.models:
            model.fit(X, y)
        
        # Compute log-likelihood for each model
        log_likelihoods = np.zeros(n_models)
        for i, model in enumerate(self.models):
            predictions = model.predict(X)
            # Compute mean squared error
            mse = np.mean((y - predictions) ** 2)
            # Convert to log-likelihood assuming Gaussian errors
            log_likelihoods[i] = -0.5 * len(y) * np.log(2 * np.pi * mse) - 0.5 * len(y)
        
        # Apply prior weights if provided
        if self.prior_weights is not None:
            log_posterior = log_likelihoods + np.log(self.prior_weights)
        else:
            # Uniform prior
            log_posterior = log_likelihoods
        
        # Normalize to get posterior weights
        max_log_posterior = np.max(log_posterior)
        posterior = np.exp(log_posterior - max_log_posterior)
        self.posterior_weights = posterior / np.sum(posterior)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        # Get predictions from each model
        predictions = np.zeros((len(X), len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        # Weighted average of predictions
        return np.dot(predictions, self.posterior_weights)
    
    def get_model_weights(self) -> pd.DataFrame:
        """
        Get the posterior weights for each model.
        
        Returns:
            DataFrame with model weights
        """
        if self.posterior_weights is None:
            raise ValueError("Model not fitted yet")
        
        return pd.DataFrame({
            'Model': [str(model) for model in self.models],
            'Weight': self.posterior_weights
        })