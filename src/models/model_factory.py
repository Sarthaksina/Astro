# Cosmic Market Oracle - Model Factory

"""
This module provides a unified interface for creating and configuring
different types of models for market prediction based on astrological data.

The factory pattern allows for easy model selection and configuration
based on experiment parameters.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

# Import model implementations
from src.models.time_series import AttentionBiLSTM, TemporalConvNet, WaveNetModel
# from src.models.transformers import AstroEconomicTransformer, AstroEventDetectionTransformer # Removed
from src.models.advanced_architectures import (
    TransformerModel,
    GNNModel,
    HybridCNNTransformerModel,
    NeuralODEModel
)
from src.models.ensemble import EnsembleModel


class ModelFactory:
    """
    Factory class for creating and configuring market prediction models.
    
    This class provides a unified interface for creating different types of models
    based on configuration parameters.
    
    Args:
        config (Dict): Configuration dictionary with model parameters
    """
    def __init__(self, config: Dict):
        self.config = config
        self.model_type = config.get('model_type', 'lstm')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_model(self) -> torch.nn.Module:
        """
        Create and configure a model based on the configuration.
        
        Returns:
            Configured model instance
        """
        if self.model_type == 'lstm':
            return self._create_lstm_model()
        elif self.model_type == 'tcn':
            return self._create_tcn_model()
        elif self.model_type == 'wavenet':
            return self._create_wavenet_model()
        # elif self.model_type == 'transformer': # Removed
            # return self._create_transformer_model() # Removed
        # elif self.model_type == 'event_transformer': # Removed
            # return self._create_event_transformer_model() # Removed
        elif self.model_type == 'advanced_transformer':
            return self._create_advanced_transformer_model()
        elif self.model_type == 'gnn':
            return self._create_gnn_model()
        elif self.model_type == 'hybrid':
            return self._create_hybrid_model()
        elif self.model_type == 'neural_ode':
            return self._create_neural_ode_model()
        elif self.model_type == 'ensemble':
            # Create ensemble model directly using the EnsembleModel from ensemble.py
            model_config = self.config.get('ensemble_config', {})
            
            # Get model configurations for each sub-model
            model_configs = {}
            for model_name in model_config.get('models', ['transformer', 'gnn', 'hybrid']):
                if model_name == 'transformer':
                    model_configs[model_name] = self.config.get('advanced_transformer_config', {})
                elif model_name == 'gnn':
                    model_configs[model_name] = self.config.get('gnn_config', {})
                elif model_name == 'hybrid':
                    model_configs[model_name] = self.config.get('hybrid_config', {})
                elif model_name == 'neural_ode':
                    model_configs[model_name] = self.config.get('neural_ode_config', {})
            
            model = EnsembleModel(
                input_dim=model_config.get('input_dim', 64),
                output_dim=model_config.get('output_dim', 1),
                model_configs=model_configs,
                aggregation=model_config.get('aggregation', 'weighted'),
                weights=model_config.get('weights', None)
            )
            
            return model.to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_lstm_model(self) -> AttentionBiLSTM:
        """
        Create and configure an LSTM model.
        
        Returns:
            Configured LSTM model
        """
        model_config = self.config.get('lstm_config', {})
        
        model = AttentionBiLSTM(
            input_dim=model_config.get('input_dim', 32),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_layers=model_config.get('num_layers', 2),
            output_dim=model_config.get('output_dim', 1),
            dropout=model_config.get('dropout', 0.2)
        )
        
        return model.to(self.device)
    
    def _create_tcn_model(self) -> TemporalConvNet:
        """
        Create and configure a Temporal Convolutional Network model.
        
        Returns:
            Configured TCN model
        """
        model_config = self.config.get('tcn_config', {})
        
        model = TemporalConvNet(
            input_dim=model_config.get('input_dim', 32),
            num_channels=model_config.get('num_channels', [64, 128, 256, 512]),
            kernel_size=model_config.get('kernel_size', 3),
            dropout=model_config.get('dropout', 0.2)
        )
        
        return model.to(self.device)
    
    def _create_wavenet_model(self) -> WaveNetModel:
        """
        Create and configure a WaveNet model.
        
        Returns:
            Configured WaveNet model
        """
        model_config = self.config.get('wavenet_config', {})
        
        model = WaveNetModel(
            input_dim=model_config.get('input_dim', 32),
            residual_channels=model_config.get('residual_channels', 32),
            skip_channels=model_config.get('skip_channels', 32),
            dilation_layers=model_config.get('dilation_layers', 10),
            output_dim=model_config.get('output_dim', 1)
        )
        
        return model.to(self.device)
    
    # Removed _create_transformer_model method
    
    # Removed _create_event_transformer_model method
        
    def _create_advanced_transformer_model(self) -> TransformerModel:
        """
        Create and configure an advanced Transformer model with positional encoding.
        
        Returns:
            Configured advanced transformer model
        """
        model_config = self.config.get('advanced_transformer_config', {})
        
        model = TransformerModel(
            input_dim=model_config.get('input_dim', 64),
            output_dim=model_config.get('output_dim', 1),
            d_model=model_config.get('d_model', 128),
            nhead=model_config.get('num_heads', 8),
            num_layers=model_config.get('num_layers', 4),
            dim_feedforward=model_config.get('dim_feedforward', 512),
            dropout=model_config.get('dropout', 0.1),
            activation=model_config.get('activation', 'gelu'),
            max_seq_length=model_config.get('max_seq_length', 1000)
        )
        
        return model.to(self.device)
    
    def _create_gnn_model(self) -> GNNModel:
        """
        Create and configure a Graph Neural Network model for planetary relationships.
        
        Returns:
            Configured GNN model
        """
        model_config = self.config.get('gnn_config', {})
        
        model = GNNModel(
            input_dim=model_config.get('input_dim', 32),
            output_dim=model_config.get('output_dim', 1),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_layers=model_config.get('num_layers', 3),
            dropout=model_config.get('dropout', 0.2),
            gnn_type=model_config.get('gnn_type', 'gat'),
            aggregation=model_config.get('aggregation', 'mean'),
            use_edge_features=model_config.get('use_edge_features', True),
            graph_construction=model_config.get('graph_construction', 'distance')
        )
        
        return model.to(self.device)
    
    def _create_hybrid_model(self) -> HybridCNNTransformerModel:
        """
        Create and configure a hybrid CNN-Transformer model.
        
        Returns:
            Configured hybrid model
        """
        model_config = self.config.get('hybrid_config', {})
        
        model = HybridCNNTransformerModel(
            input_dim=model_config.get('input_dim', 64),
            output_dim=model_config.get('output_dim', 1),
            cnn_filters=model_config.get('cnn_filters', 64),
            cnn_kernel_size=model_config.get('cnn_kernel_size', 3),
            transformer_dim=model_config.get('transformer_dim', 128),
            transformer_heads=model_config.get('transformer_heads', 4),
            transformer_layers=model_config.get('transformer_layers', 2),
            dropout=model_config.get('dropout', 0.1),
            use_residual=model_config.get('use_residual', True)
        )
        
        return model.to(self.device)
    
    def _create_neural_ode_model(self) -> NeuralODEModel:
        """
        Create and configure a Neural ODE model for continuous-time dynamics.
        
        Returns:
            Configured Neural ODE model
        """
        model_config = self.config.get('neural_ode_config', {})
        
        model = NeuralODEModel(
            input_dim=model_config.get('input_dim', 64),
            output_dim=model_config.get('output_dim', 1),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_layers=model_config.get('num_layers', 3),
            dropout=model_config.get('dropout', 0.1),
            solver=model_config.get('solver', 'dopri5'),
            rtol=model_config.get('rtol', 1e-3),
            atol=model_config.get('atol', 1e-4)
        )
        
        return model.to(self.device)
    
    # _create_ensemble_model method removed - now directly using EnsembleModel from ensemble.py in create_model method


def create_model_from_config(config: Dict) -> torch.nn.Module:
    """
    Convenience function to create a model from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        Configured model instance
    """
    factory = ModelFactory(config)
    return factory.create_model()


def get_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model based on configuration.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary with optimizer parameters
        
    Returns:
        Configured optimizer
    """
    optimizer_config = config.get('optimizer_config', {})
    optimizer_type = optimizer_config.get('type', 'adam')
    lr = optimizer_config.get('learning_rate', 0.001)
    weight_decay = optimizer_config.get('weight_decay', 0.0)
    
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration dictionary with scheduler parameters
        
    Returns:
        Configured scheduler or None if not specified
    """
    scheduler_config = config.get('scheduler_config', {})
    scheduler_type = scheduler_config.get('type', None)
    
    if scheduler_type is None:
        return None
    
    if scheduler_type.lower() == 'step':
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type.lower() == 'cosine':
        T_max = scheduler_config.get('T_max', 100)
        eta_min = scheduler_config.get('eta_min', 0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type.lower() == 'plateau':
        patience = scheduler_config.get('patience', 10)
        factor = scheduler_config.get('factor', 0.1)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")