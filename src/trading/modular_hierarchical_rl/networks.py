"""
Shared neural network components for the modular hierarchical RL framework.

This module contains base classes and common components used by both
the strategic planner and tactical executor networks.
"""

import torch
import torch.nn as nn
import numpy as np

class BaseNetwork(nn.Module):
    """
    Base class for all neural network components.
    
    Provides common initialization and weight initialization methods.
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def _init_output_weights(self, layer):
        """Special initialization for output layers."""
        nn.init.orthogonal_(layer.weight, gain=0.01)
        nn.init.constant_(layer.bias, 0.0)

class SharedNetwork(BaseNetwork):
    """
    Shared network component used by both actor and critic.
    
    This network processes the combined state and goal input.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__(hidden_dim)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        """Forward pass through shared layers."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x
