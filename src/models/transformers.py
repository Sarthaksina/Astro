# Cosmic Market Oracle - Transformer Models

"""
This module implements transformer-based models for market prediction,
specializing in attention mechanisms that can capture relationships
between astrological events and market movements.

Implemented models include:
- Custom positional encodings based on astrological cycles
- Multi-head attention mechanisms for planetary relationships
- Astro-economic event detection with specialized tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union


class AstrologicalPositionalEncoding(nn.Module):
    """
    Custom positional encoding based on astrological cycles.
    
    This encoding incorporates knowledge of planetary cycles and
    astrological periods to create more meaningful position representations.
    
    Args:
        d_model (int): Embedding dimension
        max_len (int): Maximum sequence length
        planetary_cycles (Dict[str, float]): Dictionary mapping planet names to their cycle periods
    """
    def __init__(self, d_model: int, max_len: int = 5000, 
                 planetary_cycles: Dict[str, float] = None):
        super(AstrologicalPositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Default planetary cycles in days if not provided
        self.planetary_cycles = planetary_cycles or {
            'Sun': 365.25,  # Solar year
            'Moon': 29.53,   # Lunar month
            'Mercury': 87.97,
            'Venus': 224.7,
            'Mars': 686.98,
            'Jupiter': 4332.59,
            'Saturn': 10759.22,
            'Lunar_Node': 6798.38,  # Nodal cycle (18.6 years)
        }
        
        # Create positional encoding buffer
        self.register_buffer('pe', self._create_encoding())
    
    def _create_encoding(self) -> torch.Tensor:
        """
        Create the positional encoding tensor incorporating astrological cycles.
        
        Returns:
            Positional encoding tensor of shape (max_len, d_model)
        """
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        
        # Allocate dimensions to different planetary cycles
        dim_per_planet = self.d_model // (2 * len(self.planetary_cycles))
        remaining_dims = self.d_model - (2 * dim_per_planet * len(self.planetary_cycles))
        
        current_dim = 0
        for i, (planet, cycle) in enumerate(self.planetary_cycles.items()):
            # Calculate number of dimensions for this planet
            if i == len(self.planetary_cycles) - 1:
                planet_dims = dim_per_planet * 2 + remaining_dims
            else:
                planet_dims = dim_per_planet * 2
            
            # Create sine and cosine encodings based on planetary cycle
            div_term = torch.exp(torch.arange(0, planet_dims, 2).float() * (-math.log(10000.0) / planet_dims))
            pe[:, current_dim:current_dim+planet_dims:2] = torch.sin(position * div_term * (2 * math.pi / cycle))
            pe[:, current_dim+1:current_dim+planet_dims:2] = torch.cos(position * div_term * (2 * math.pi / cycle))
            
            current_dim += planet_dims
        
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class PlanetaryMultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism specializing in different planetary relationships.
    
    Each attention head can focus on relationships between specific planets
    or astrological factors.
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
        planetary_bias (bool): Whether to use planetary bias in attention
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, 
                 planetary_bias: bool = True):
        super(PlanetaryMultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.planetary_bias = planetary_bias
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Planetary bias - learnable parameters for each head to focus on specific relationships
        if planetary_bias:
            self.planet_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """
        Initialize the weights for the linear projections.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the attention mechanism.
        
        Args:
            query: Query tensor of shape (batch_size, query_len, d_model)
            key: Key tensor of shape (batch_size, key_len, d_model)
            value: Value tensor of shape (batch_size, value_len, d_model)
            mask: Optional mask tensor of shape (batch_size, query_len, key_len)
            
        Returns:
            output: Attention output
            attention_weights: Attention weights for interpretability
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply planetary bias if enabled
        if self.planetary_bias:
            scores = scores + self.planet_bias
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(context)
        
        return output, attention_weights


class AstroEconomicTransformer(nn.Module):
    """
    Transformer model for astro-economic event detection and market prediction.
    
    This model combines astrological data with market data using specialized
    attention mechanisms and positional encodings.
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        d_ff (int): Dimension of feedforward network
        max_seq_len (int): Maximum sequence length
        dropout (float): Dropout probability
        num_market_features (int): Number of market features
        num_astro_features (int): Number of astrological features
        output_dim (int): Number of output features
        planetary_cycles (Dict[str, float]): Dictionary of planetary cycles
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8, num_layers: int = 6,
                 d_ff: int = 1024, max_seq_len: int = 365, dropout: float = 0.1,
                 num_market_features: int = 10, num_astro_features: int = 20,
                 output_dim: int = 1, planetary_cycles: Dict[str, float] = None):
        super(AstroEconomicTransformer, self).__init__()
        
        # Feature embedding layers
        self.market_embedding = nn.Linear(num_market_features, d_model // 2)
        self.astro_embedding = nn.Linear(num_astro_features, d_model // 2)
        
        # Positional encoding
        self.pos_encoding = AstrologicalPositionalEncoding(
            d_model, max_seq_len, planetary_cycles
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            AstroEconomicTransformerLayer(
                d_model, num_heads, d_ff, dropout
            ) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, market_data: torch.Tensor, astro_data: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer model.
        
        Args:
            market_data: Market features of shape (batch_size, seq_len, num_market_features)
            astro_data: Astrological features of shape (batch_size, seq_len, num_astro_features)
            mask: Optional mask tensor
            
        Returns:
            Predicted values
        """
        # Embed features
        market_embedded = self.market_embedding(market_data)
        astro_embedded = self.astro_embedding(astro_data)
        
        # Concatenate embeddings
        x = torch.cat([market_embedded, astro_embedded], dim=-1)
        
        # Apply positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Apply output layer (using the last token for prediction)
        output = self.output_layer(x[:, -1, :])
        
        return output


class AstroEconomicTransformerLayer(nn.Module):
    """
    Transformer layer for the AstroEconomic Transformer.
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Dimension of feedforward network
        dropout (float): Dropout probability
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(AstroEconomicTransformerLayer, self).__init__()
        
        # Multi-head attention
        self.attention = PlanetaryMultiHeadAttention(d_model, num_heads, dropout)
        
        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer normalization
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward network with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class AstroEventDetectionTransformer(nn.Module):
    """
    Transformer model for detecting significant astrological events that impact markets.
    
    This model uses specialized tokens to identify important astrological configurations
    and their potential market impact.
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        num_astro_events (int): Number of astrological event types to detect
        max_seq_len (int): Maximum sequence length
        dropout (float): Dropout probability
        num_features (int): Number of input features
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8, num_layers: int = 4,
                 num_astro_events: int = 10, max_seq_len: int = 365, 
                 dropout: float = 0.1, num_features: int = 30):
        super(AstroEventDetectionTransformer, self).__init__()
        
        # Feature embedding
        self.feature_embedding = nn.Linear(num_features, d_model)
        
        # Event token embeddings (learnable)
        self.event_tokens = nn.Parameter(torch.randn(num_astro_events, d_model))
        
        # Positional encoding
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Event detection head
        self.event_detector = nn.Linear(d_model, num_astro_events)
        
        # Market impact prediction head
        self.market_predictor = nn.Linear(d_model + num_astro_events, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the event detection transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            
        Returns:
            market_prediction: Predicted market movement
            event_probabilities: Detected astrological event probabilities
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed features
        x = self.feature_embedding(x)
        
        # Add positional encoding
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_encoding(positions)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer encoder
        encoder_output = self.transformer_encoder(x)
        
        # Use the output corresponding to the last token for event detection
        last_hidden = encoder_output[:, -1, :]
        
        # Detect astrological events
        event_logits = self.event_detector(last_hidden)
        event_probabilities = torch.sigmoid(event_logits)
        
        # Concatenate event probabilities with hidden state for market prediction
        market_features = torch.cat([last_hidden, event_probabilities], dim=-1)
        
        # Predict market movement
        market_prediction = self.market_predictor(market_features)
        
        return market_prediction, event_probabilities