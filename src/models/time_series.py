# Cosmic Market Oracle - Time Series Models

"""
This module implements various time series models for market prediction,
specializing in capturing temporal patterns in financial and astrological data.

Implemented models include:
- Bidirectional LSTMs with attention mechanisms
- Temporal Convolutional Networks (TCN)
- WaveNet-inspired architectures for cyclical pattern detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class AttentionBiLSTM(nn.Module):
    """
    Bidirectional LSTM with attention mechanism for time series forecasting.
    
    This model is designed to capture temporal dependencies in market data
    while paying special attention to significant astrological events.
    
    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden units in LSTM
        num_layers (int): Number of LSTM layers
        output_dim (int): Number of output features
        dropout (float): Dropout probability
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2):
        super(AttentionBiLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            output: Predicted values
            attention_weights: Attention weights for interpretability
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Apply dropout and pass through final layer
        context_vector = self.dropout(context_vector)
        output = self.fc(context_vector)
        
        return output, attention_weights


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for time series forecasting.
    
    Implements dilated causal convolutions to efficiently capture long-range
    dependencies in time series data without the recurrent connections of RNNs.
    
    Args:
        input_dim (int): Number of input features
        num_channels (List[int]): Number of channels in each layer
        kernel_size (int): Kernel size for all convolutions
        dropout (float): Dropout probability
    """
    def __init__(self, input_dim: int, num_channels: List[int], 
                 kernel_size: int = 3, dropout: float = 0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i  # Exponentially increasing dilation
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size, 
                    stride=1, dilation=dilation, 
                    padding=(kernel_size-1) * dilation, 
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, num_channels[-1], seq_len)
        """
        return self.network(x)


class TemporalBlock(nn.Module):
    """
    Temporal block for the Temporal Convolutional Network.
    
    Consists of two dilated causal convolutions with residual connection.
    
    Args:
        n_inputs (int): Number of input channels
        n_outputs (int): Number of output channels
        kernel_size (int): Kernel size for convolutions
        stride (int): Stride for convolutions
        dilation (int): Dilation factor for convolutions
        padding (int): Padding size
        dropout (float): Dropout probability
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, 
                 stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        
        # First dilated convolution
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        
        # Second dilated convolution
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        
        # Residual connection if input and output dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        # Regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Normalization
        self.batch_norm1 = nn.BatchNorm1d(n_outputs)
        self.batch_norm2 = nn.BatchNorm1d(n_outputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the temporal block.
        
        Args:
            x: Input tensor of shape (batch_size, n_inputs, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, n_outputs, seq_len)
        """
        # First convolution branch
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution branch
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class WaveNetModel(nn.Module):
    """
    WaveNet-inspired architecture for cyclical pattern detection in market data.
    
    This model uses dilated causal convolutions with skip connections to efficiently
    model long-range dependencies in time series data, particularly suited for
    detecting cyclical patterns in astrological and market data.
    
    Args:
        input_dim (int): Number of input features
        residual_channels (int): Number of channels in residual layers
        skip_channels (int): Number of channels in skip connections
        dilation_layers (int): Number of dilation layers
        output_dim (int): Number of output features
    """
    def __init__(self, input_dim: int, residual_channels: int = 32, 
                 skip_channels: int = 32, dilation_layers: int = 10, 
                 output_dim: int = 1):
        super(WaveNetModel, self).__init__()
        
        # Initial causal convolution to convert input to residual channels
        self.causal_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=residual_channels,
            kernel_size=2,
            padding=1,  # Causal padding
        )
        
        # Dilated convolution stack
        self.dilated_stack = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # Create dilated convolution layers with increasing dilation
        for i in range(dilation_layers):
            dilation = 2 ** i
            
            # Dilated convolution with gated activation
            self.dilated_stack.append(
                nn.Conv1d(
                    in_channels=residual_channels,
                    out_channels=residual_channels * 2,  # For gated activation
                    kernel_size=2,
                    dilation=dilation,
                    padding=dilation  # Causal padding
                )
            )
            
            # Skip connection
            self.skip_connections.append(
                nn.Conv1d(
                    in_channels=residual_channels,
                    out_channels=skip_channels,
                    kernel_size=1
                )
            )
        
        # Final 1x1 convolution layers
        self.final_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.final_conv2 = nn.Conv1d(skip_channels, output_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the WaveNet model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, output_dim, seq_len)
        """
        # Initial causal convolution
        x = self.causal_conv(x)
        skip_sum = 0
        
        # Dilated convolution stack
        for i, (dilated_conv, skip_conv) in enumerate(zip(self.dilated_stack, self.skip_connections)):
            # Dilated convolution with gated activation
            dilated_out = dilated_conv(x)
            
            # Split for gated activation
            filter_out, gate_out = torch.chunk(dilated_out, 2, dim=1)
            
            # Apply gated activation (sigmoid * tanh)
            gated_out = torch.sigmoid(gate_out) * torch.tanh(filter_out)
            
            # Skip connection
            skip = skip_conv(gated_out)
            skip_sum = skip + skip_sum
            
            # Residual connection
            x = x + gated_out
        
        # Final 1x1 convolution layers
        x = F.relu(skip_sum)
        x = F.relu(self.final_conv1(x))
        x = self.final_conv2(x)
        
        return x