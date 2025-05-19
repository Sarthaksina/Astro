#!/usr/bin/env python
# Cosmic Market Oracle - Advanced Model Architectures

"""
Advanced deep learning model architectures for the Cosmic Market Oracle.

This module implements state-of-the-art neural network architectures optimized for
astrological and financial time series data, including Transformers, Graph Neural
Networks, and hybrid architectures.
"""

import math
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE
from torch_geometric.data import Data, Batch

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    
    Adds positional information to the input embeddings to provide
    sequence order information to the self-attention mechanism.
    
    Can incorporate astrological cycles for specialized encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, use_astro_cycles: bool = False, 
                 planetary_cycles: Dict[str, float] = None):
        """
        Initialize the positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            use_astro_cycles: Whether to use astrological cycles for encoding
            planetary_cycles: Dictionary mapping planet names to their cycle periods
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.use_astro_cycles = use_astro_cycles
        
        # Default planetary cycles in days if not provided and using astro cycles
        self.planetary_cycles = planetary_cycles or {
            'Sun': 365.25,     # Solar year
            'Moon': 29.53,      # Lunar month
            'Mercury': 87.97,
            'Venus': 224.7,
            'Mars': 686.98,
            'Jupiter': 4332.59,
            'Saturn': 10759.22,
            'Lunar_Node': 6798.38,  # Nodal cycle (18.6 years)
        }
        
        # Create positional encoding
        pe = self._create_encoding()
        
        # Add batch dimension and register as buffer
        self.register_buffer("pe", pe)
    
    def _create_encoding(self) -> torch.Tensor:
        """
        Create the positional encoding tensor.
        
        Returns:
            Positional encoding tensor of shape (1, max_len, d_model)
        """
        if not self.use_astro_cycles:
            # Standard sinusoidal encoding
            pe = torch.zeros(self.max_len, self.d_model)
            position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
            )
            
            # Apply sine to even indices
            pe[:, 0::2] = torch.sin(position * div_term)
            
            # Apply cosine to odd indices
            pe[:, 1::2] = torch.cos(position * div_term)
            
            return pe.unsqueeze(0)
        else:
            # Astrological cycle-based encoding
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
                div_term = torch.exp(torch.arange(0, planet_dims, 2).float() * 
                                    (-math.log(10000.0) / planet_dims))
                pe[:, current_dim:current_dim+planet_dims:2] = torch.sin(
                    position * div_term * (2 * math.pi / cycle))
                pe[:, current_dim+1:current_dim+planet_dims:2] = torch.cos(
                    position * div_term * (2 * math.pi / cycle))
                
                current_dim += planet_dims
            
            return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    """
    Transformer-based model for time series prediction.
    
    Uses self-attention mechanisms to capture dependencies between
    different time steps and features in the input sequence.
    
    Can handle both standard time series data and specialized
    astrological-economic data with separate feature streams.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_length: int = 1000,
        use_astro_encoding: bool = False,
        separate_streams: bool = False,
        market_features: int = 0,
        astro_features: int = 0,
        planetary_cycles: Dict[str, float] = None,
    ):
        """
        Initialize the Transformer model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function
            max_seq_length: Maximum sequence length
            use_astro_encoding: Whether to use astrological positional encoding
            separate_streams: Whether to use separate streams for market and astro data
            market_features: Number of market features (only used if separate_streams=True)
            astro_features: Number of astrological features (only used if separate_streams=True)
            planetary_cycles: Dictionary of planetary cycles for positional encoding
        """
        super().__init__()
        
        self.separate_streams = separate_streams
        
        if separate_streams:
            # Separate projections for market and astrological data
            self.market_projection = nn.Linear(market_features, d_model // 2)
            self.astro_projection = nn.Linear(astro_features, d_model // 2)
            self.combined_projection = nn.Linear(d_model, d_model)
        else:
            # Single input projection
            self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model, 
            max_seq_length, 
            use_astro_cycles=use_astro_encoding,
            planetary_cycles=planetary_cycles
        )
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            getattr(nn, activation.upper())(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                market_data: Optional[torch.Tensor] = None, 
                astro_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Transformer model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim] (used if separate_streams=False)
            mask: Optional mask for padding
            market_data: Market features tensor (used if separate_streams=True)
            astro_data: Astrological features tensor (used if separate_streams=True)
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        if self.separate_streams:
            # Process separate market and astrological data streams
            if market_data is None or astro_data is None:
                raise ValueError("Both market_data and astro_data must be provided when using separate streams")
                
            # Project each stream
            market_features = self.market_projection(market_data)
            astro_features = self.astro_projection(astro_data)
            
            # Concatenate along feature dimension
            x = torch.cat([market_features, astro_features], dim=-1)
            
            # Apply combined projection
            x = self.combined_projection(x)
        else:
            # Project input to embedding dimension
            x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use the output corresponding to the last time step
        x = x[:, -1, :]
        
        # Project to output dimension
        x = self.output_projection(x)
        
        return x


class PlanetaryGraphConstruction:
    """
    Constructs a graph representation of planetary relationships.
    
    Creates nodes for planets and edges based on aspects, distances,
    or other relationships between planets.
    """
    
    def __init__(
        self,
        construction_type: str = "distance",
        threshold: float = 10.0,
        k: int = 5,
        use_aspects: bool = True,
        aspect_types: Optional[List[str]] = None,
    ):
        """
        Initialize the graph construction module.
        
        Args:
            construction_type: Type of graph construction (distance, threshold, knn)
            threshold: Distance threshold for creating edges
            k: Number of nearest neighbors for knn
            use_aspects: Whether to use astrological aspects for edge creation
            aspect_types: List of aspect types to consider
        """
        self.construction_type = construction_type
        self.threshold = threshold
        self.k = k
        self.use_aspects = use_aspects
        
        if aspect_types is None:
            self.aspect_types = [
                "conjunction", "opposition", "trine", "square", "sextile"
            ]
        else:
            self.aspect_types = aspect_types
        
        # Aspect angles in degrees
        self.aspect_angles = {
            "conjunction": 0,
            "opposition": 180,
            "trine": 120,
            "square": 90,
            "sextile": 60
        }
        
        # Aspect orbs (allowable deviation) in degrees
        self.aspect_orbs = {
            "conjunction": 10,
            "opposition": 10,
            "trine": 8,
            "square": 8,
            "sextile": 6
        }
    
    def construct_graph(
        self,
        planetary_positions: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct a graph from planetary positions.
        
        Args:
            planetary_positions: Tensor of shape [batch_size, num_planets, features]
            batch_size: Batch size
            
        Returns:
            Tuple of (node_features, edge_index, edge_attr)
        """
        num_planets = planetary_positions.shape[1]
        device = planetary_positions.device
        
        # Reshape to [batch_size * num_planets, features]
        node_features = planetary_positions.reshape(-1, planetary_positions.shape[2])
        
        # Initialize edge lists
        edge_index_list = []
        edge_attr_list = []
        
        for b in range(batch_size):
            # Get planetary positions for this batch
            positions = planetary_positions[b]
            
            # Extract longitudes (assuming first feature is longitude)
            longitudes = positions[:, 0]
            
            if self.construction_type == "distance" or self.use_aspects:
                # Calculate pairwise angular distances
                distances = torch.zeros((num_planets, num_planets), device=device)
                for i in range(num_planets):
                    for j in range(num_planets):
                        if i != j:
                            # Calculate shortest angular distance
                            diff = abs(longitudes[i] - longitudes[j]) % 360
                            distances[i, j] = min(diff, 360 - diff)
            
            # Create edges based on construction type
            if self.construction_type == "distance":
                # Create edges for planets within threshold distance
                edges = (distances < self.threshold).nonzero()
                
                # Create edge attributes (distances)
                edge_attr = distances[edges[:, 0], edges[:, 1]].unsqueeze(1)
                
            elif self.construction_type == "threshold":
                # Create edges for all planets (fully connected)
                edges = torch.ones((num_planets, num_planets), device=device).nonzero()
                
                # Remove self-loops
                mask = edges[:, 0] != edges[:, 1]
                edges = edges[mask]
                
                # Create edge attributes (distances)
                edge_attr = distances[edges[:, 0], edges[:, 1]].unsqueeze(1)
                
            elif self.construction_type == "knn":
                # Create k nearest neighbors for each planet
                edges = []
                edge_attr_list_batch = []
                
                for i in range(num_planets):
                    # Get distances from planet i to all others
                    dist = distances[i]
                    
                    # Set self-distance to infinity
                    dist[i] = float('inf')
                    
                    # Get k nearest neighbors
                    _, indices = torch.topk(dist, min(self.k, num_planets - 1), largest=False)
                    
                    # Add edges
                    for j in indices:
                        edges.append([i, j])
                        edge_attr_list_batch.append([dist[j]])
                
                edges = torch.tensor(edges, device=device)
                edge_attr = torch.tensor(edge_attr_list_batch, device=device)
            
            # If using aspects, add aspect-based edges
            if self.use_aspects and self.construction_type != "distance":
                aspect_edges = []
                aspect_attr = []
                
                for aspect_type in self.aspect_types:
                    angle = self.aspect_angles[aspect_type]
                    orb = self.aspect_orbs[aspect_type]
                    
                    # Find planets forming this aspect
                    for i in range(num_planets):
                        for j in range(i + 1, num_planets):
                            # Check if planets form the aspect
                            if abs(distances[i, j] - angle) < orb:
                                # Add bidirectional edges
                                aspect_edges.extend([[i, j], [j, i]])
                                
                                # Edge attribute: [distance, aspect_type_idx]
                                aspect_type_idx = self.aspect_types.index(aspect_type)
                                attr_value = [distances[i, j], aspect_type_idx]
                                aspect_attr.extend([attr_value, attr_value])
                
                if aspect_edges:
                    aspect_edges = torch.tensor(aspect_edges, device=device)
                    aspect_attr = torch.tensor(aspect_attr, device=device)
                    
                    # Combine with existing edges
                    edges = torch.cat([edges, aspect_edges], dim=0)
                    edge_attr = torch.cat([
                        edge_attr,
                        torch.zeros((aspect_attr.shape[0], edge_attr.shape[1] - aspect_attr.shape[1]), device=device),
                        aspect_attr
                    ], dim=1)
            
            # Adjust node indices for batch
            edges = edges + b * num_planets
            
            edge_index_list.append(edges)
            edge_attr_list.append(edge_attr)
        
        # Combine edges from all batches
        edge_index = torch.cat(edge_index_list, dim=0).t()
        edge_attr = torch.cat(edge_attr_list, dim=0)
        
        return node_features, edge_index, edge_attr


class GNNModel(nn.Module):
    """
    Graph Neural Network model for planetary relationship modeling.
    
    Uses graph convolutions to capture relationships between planets
    and their aspects for market prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        gnn_type: str = "gat",
        aggregation: str = "mean",
        use_edge_features: bool = True,
        graph_construction: str = "distance",
    ):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Number of input features per node
            output_dim: Number of output features
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
            gnn_type: Type of GNN layer (gcn, gat, graphsage)
            aggregation: Aggregation method (mean, sum, max)
            use_edge_features: Whether to use edge features
            graph_construction: Type of graph construction
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.aggregation = aggregation
        self.use_edge_features = use_edge_features
        
        # Graph construction
        self.graph_constructor = PlanetaryGraphConstruction(
            construction_type=graph_construction
        )
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # First layer
        if gnn_type == "gcn":
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn_type == "gat":
            self.gnn_layers.append(GATConv(hidden_dim, hidden_dim))
        elif gnn_type == "graphsage":
            self.gnn_layers.append(GraphSAGE(hidden_dim, hidden_dim))
        
        # Additional layers
        for _ in range(num_layers - 1):
            if gnn_type == "gcn":
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "gat":
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim))
            elif gnn_type == "graphsage":
                self.gnn_layers.append(GraphSAGE(hidden_dim, hidden_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN model.
        
        Args:
            x: Input tensor of shape [batch_size, num_planets, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # Construct graph
        node_features, edge_index, edge_attr = self.graph_constructor.construct_graph(
            x, batch_size
        )
        
        # Project input
        node_features = self.input_projection(node_features)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            if self.use_edge_features and hasattr(gnn_layer, "edge_attr"):
                node_features = gnn_layer(node_features, edge_index, edge_attr)
            else:
                node_features = gnn_layer(node_features, edge_index)
            
            node_features = F.relu(node_features)
            node_features = self.dropout(node_features)
        
        # Aggregate node features per batch
        node_features = node_features.view(batch_size, -1, self.hidden_dim)
        
        if self.aggregation == "mean":
            graph_features = torch.mean(node_features, dim=1)
        elif self.aggregation == "sum":
            graph_features = torch.sum(node_features, dim=1)
        elif self.aggregation == "max":
            graph_features, _ = torch.max(node_features, dim=1)
        
        # Project to output
        output = self.output_projection(graph_features)
        
        return output


class HybridCNNTransformerModel(nn.Module):
    """
    Hybrid CNN-Transformer model for time series prediction.
    
    Combines convolutional layers for local pattern extraction with
    transformer layers for capturing long-range dependencies.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cnn_filters: int = 64,
        cnn_kernel_size: int = 3,
        transformer_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        """
        Initialize the hybrid CNN-Transformer model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            cnn_filters: Number of CNN filters
            cnn_kernel_size: CNN kernel size
            transformer_dim: Transformer embedding dimension
            transformer_heads: Number of transformer attention heads
            transformer_layers: Number of transformer layers
            dropout: Dropout probability
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.use_residual = use_residual
        
        # CNN layers
        self.conv1 = nn.Conv1d(
            input_dim, cnn_filters, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            cnn_filters, cnn_filters, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2
        )
        
        # Projection to transformer dimension
        self.projection = nn.Linear(cnn_filters, transformer_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(transformer_dim)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=transformer_layers
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid CNN-Transformer model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Transpose for CNN [batch_size, input_dim, seq_len]
        x_cnn = x.transpose(1, 2)
        
        # Apply CNN layers
        cnn_out = F.relu(self.conv1(x_cnn))
        cnn_out = F.relu(self.conv2(cnn_out))
        
        # Transpose back [batch_size, seq_len, cnn_filters]
        cnn_out = cnn_out.transpose(1, 2)
        
        # Project to transformer dimension
        transformer_in = self.projection(cnn_out)
        
        # Add positional encoding
        transformer_in = self.pos_encoder(transformer_in)
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(transformer_in)
        
        # Residual connection
        if self.use_residual:
            transformer_out = transformer_out + transformer_in
        
        # Use the last sequence element for prediction
        transformer_out = transformer_out[:, -1, :]
        
        # Project to output dimension
        output = self.output_projection(transformer_out)
        
        return output


class NeuralODEModel(nn.Module):
    """
    Neural ODE model for continuous-time dynamics.
    
    Models the continuous evolution of planetary positions and their
    influence on market movements.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        """
        Initialize the Neural ODE model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            hidden_dim: Hidden dimension
            num_layers: Number of layers in ODE function
            dropout: Dropout probability
            solver: ODE solver (dopri5, euler, rk4)
            rtol: Relative tolerance for adaptive solvers
            atol: Absolute tolerance for adaptive solvers
        """
        super().__init__()
        
        try:
            from torchdiffeq import odeint
            self.odeint = odeint
        except ImportError:
            raise ImportError(
                "torchdiffeq is required for Neural ODE models. "
                "Install it with: pip install torchdiffeq"
            )
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # ODE function
        self.ode_func = ODEFunc(hidden_dim, num_layers, dropout)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Neural ODE model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        h = self.input_projection(x[:, 0, :])
        
        # Integration times
        t = torch.linspace(0, 1, seq_len, device=x.device)
        
        # Solve ODE
        h_history = self.odeint(
            self.ode_func,
            h,
            t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )
        
        # Get final state
        h_final = h_history[-1]
        
        # Project to output dimension
        output = self.output_projection(h_final)
        
        return output


class ODEFunc(nn.Module):
    """
    ODE function for Neural ODE model.
    
    Defines the dynamics of the system.
    """
    
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        """
        Initialize the ODE function.
        
        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            dropout: Dropout probability
        """
        super().__init__()
        
        # Create layers
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ODE function.
        
        Args:
            t: Time tensor
            h: Hidden state tensor
            
        Returns:
            Derivative of hidden state
        """
        return self.net(h)


class EnsembleModel(nn.Module):
    """
    [DEPRECATED] This class is deprecated in favor of the more specialized ensemble 
    implementations in src.models.ensemble module.
    
    This is a forwarding class that redirects to the EnsembleModel implementation in ensemble.py.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model_configs: Dict[str, Dict[str, Any]],
        aggregation: str = "weighted",
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize the ensemble model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            model_configs: Dictionary of model configurations
            aggregation: Aggregation method (mean, weighted, stacking)
            weights: Weights for weighted aggregation
        """
        super().__init__()
        
        # Import here to avoid circular imports
        from src.models.ensemble import EnsembleModel as NewEnsembleModel
        
        # Create an instance of the new implementation
        self.model = NewEnsembleModel(
            input_dim=input_dim,
            output_dim=output_dim,
            model_configs=model_configs,
            aggregation=aggregation,
            weights=weights
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ensemble model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        return self.model(x)
