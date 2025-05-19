#!/usr/bin/env python
# NOTE: This file was automatically updated to use the new modular hierarchical RL framework.
# Please review the changes and refer to docs/migration_guide_hierarchical_rl.md for more information.


# Cosmic Market Oracle - Modular Hierarchical RL Strategy

"""
Trading strategy implementation using the modular hierarchical RL framework.

This module provides a trading strategy that uses the modular hierarchical RL
framework for decision making, allowing for sophisticated trading strategies
with strategic planning and tactical execution.
"""

import numpy as np
import pandas as pd
import torch
from collections import deque
import os
import logging

from src.trading.strategy_framework import BaseStrategy
from src.trading.modular_hierarchical_rl import (
    ModularHierarchicalRLAgent,
    MCTSStrategicPlanner,
    PPOTacticalExecutor
)
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("modular_hierarchical_rl_strategy")


class ModularHierarchicalRLStrategy(BaseStrategy):
    """
    Trading strategy using the modular hierarchical RL framework.
    
    This strategy uses a trained modular hierarchical RL agent to make trading decisions,
    with strategic planning and tactical execution layers.
    """
    
    def __init__(self, 
                 agent_path: str,
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 window_size: int = 30,
                 use_features: bool = True,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the strategy.
        
        Args:
            agent_path: Path to the trained agent
            state_dim: Dimension of the state space
            goal_dim: Dimension of the goal space
            action_dim: Dimension of the action space
            window_size: Size of the observation window
            use_features: Whether to use engineered features
            device: Device to run the models on
        """
        super(ModularHierarchicalRLStrategy, self).__init__()
        
        self.agent_path = agent_path
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.use_features = use_features
        self.device = device
        
        # Create strategic planner
        strategic_planner = MCTSStrategicPlanner(
            state_dim=state_dim,
            goal_dim=goal_dim,
            device=device
        )
        
        # Create tactical executor
        tactical_executor = PPOTacticalExecutor(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            device=device
        )
        
        # Create agent
        self.agent = ModularHierarchicalRLAgent(
            strategic_planner=strategic_planner,
            tactical_executor=tactical_executor,
            device=device
        )
        
        # Load trained agent
        self.agent.load(agent_path)
        
        # Initialize state buffer
        self.state_buffer = deque(maxlen=window_size)
        
        # Initialize position
        self.position = 0.0
        
        logger.info(f"Initialized ModularHierarchicalRLStrategy with agent from {agent_path}")
    
    def initialize(self, market_data, planetary_data=None):
        """
        Initialize the strategy with data.
        
        Args:
            market_data: Market data
            planetary_data: Planetary data
        """
        self.market_data = market_data
        self.planetary_data = planetary_data
        
        # Initialize state buffer with zeros
        for _ in range(self.window_size):
            if self.use_features:
                # Use all features
                features = np.zeros(self.state_dim)
            else:
                # Use only OHLCV
                features = np.zeros(5)
            
            self.state_buffer.append(features)
        
        logger.info("Strategy initialized with market data")
        if planetary_data is not None:
            logger.info("Planetary data also provided")
    
    def get_state(self, idx):
        """
        Get the current state.
        
        Args:
            idx: Current index in the data
            
        Returns:
            Current state
        """
        if idx < self.window_size:
            # Not enough history
            return None
        
        if self.use_features:
            # Use all features
            if self.planetary_data is not None:
                # Combine market and planetary data
                market_features = self.market_data.iloc[idx].values
                planetary_features = self.planetary_data.iloc[idx].values
                features = np.concatenate([market_features, planetary_features])
            else:
                # Use only market data
                features = self.market_data.iloc[idx].values
        else:
            # Use only OHLCV
            features = self.market_data.iloc[idx][['Open', 'High', 'Low', 'Close', 'Volume']].values
        
        # Update state buffer
        self.state_buffer.append(features)
        
        # Flatten state buffer
        state = np.array(self.state_buffer).flatten()
        
        # Add position
        state = np.append(state, self.position)
        
        return state
    
    def generate_signal(self, idx):
        """
        Generate a trading signal.
        
        Args:
            idx: Current index in the data
            
        Returns:
            Trading signal (-1 to 1)
        """
        # Get current state
        state = self.get_state(idx)
        
        if state is None:
            # Not enough history
            return 0.0
        
        # Select action using agent
        action = self.agent.select_action(state, evaluate=True)
        
        # Update position
        self.position = action[0]  # Assuming 1D action
        
        # Log signal
        logger.debug(f"Generated signal at idx {idx}: {self.position}")
        
        return self.position
    
    def calculate_position_size(self, idx, signal, capital):
        """
        Calculate position size based on signal and capital.
        
        Args:
            idx: Current index in the data
            signal: Trading signal (-1 to 1)
            capital: Available capital
            
        Returns:
            Position size
        """
        # Scale signal to position size
        position_size = signal * capital
        
        # Apply risk management
        max_position = 0.2 * capital  # Max 20% of capital per position
        position_size = np.clip(position_size, -max_position, max_position)
        
        return position_size
    
    def get_confidence(self, idx):
        """
        Get confidence in the current prediction.
        
        Args:
            idx: Current index in the data
            
        Returns:
            Confidence score (0 to 1)
        """
        # For now, use absolute value of position as confidence
        return abs(self.position)


class EnsembleHierarchicalRLStrategy(BaseStrategy):
    """
    Ensemble strategy using multiple modular hierarchical RL agents.
    
    This strategy combines predictions from multiple agents to make more robust
    trading decisions.
    """
    
    def __init__(self, 
                 agent_paths: list,
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 window_size: int = 30,
                 use_features: bool = True,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the ensemble strategy.
        
        Args:
            agent_paths: List of paths to trained agents
            state_dim: Dimension of the state space
            goal_dim: Dimension of the goal space
            action_dim: Dimension of the action space
            window_size: Size of the observation window
            use_features: Whether to use engineered features
            device: Device to run the models on
        """
        super(EnsembleHierarchicalRLStrategy, self).__init__()
        
        self.agent_paths = agent_paths
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.use_features = use_features
        self.device = device
        
        # Create individual strategies
        self.strategies = []
        for path in agent_paths:
            strategy = ModularHierarchicalRLStrategy(
                agent_path=path,
                state_dim=state_dim,
                goal_dim=goal_dim,
                action_dim=action_dim,
                window_size=window_size,
                use_features=use_features,
                device=device
            )
            self.strategies.append(strategy)
        
        # Initialize position
        self.position = 0.0
        
        logger.info(f"Initialized EnsembleHierarchicalRLStrategy with {len(agent_paths)} agents")
    
    def initialize(self, market_data, planetary_data=None):
        """
        Initialize the strategy with data.
        
        Args:
            market_data: Market data
            planetary_data: Planetary data
        """
        self.market_data = market_data
        self.planetary_data = planetary_data
        
        # Initialize individual strategies
        for strategy in self.strategies:
            strategy.initialize(market_data, planetary_data)
        
        logger.info("Ensemble strategy initialized with market data")
    
    def generate_signal(self, idx):
        """
        Generate a trading signal.
        
        Args:
            idx: Current index in the data
            
        Returns:
            Trading signal (-1 to 1)
        """
        # Generate signals from individual strategies
        signals = [strategy.generate_signal(idx) for strategy in self.strategies]
        
        # Combine signals (simple average for now)
        combined_signal = np.mean(signals)
        
        # Update position
        self.position = combined_signal
        
        # Log signal
        logger.debug(f"Generated ensemble signal at idx {idx}: {self.position}")
        
        return self.position
    
    def calculate_position_size(self, idx, signal, capital):
        """
        Calculate position size based on signal and capital.
        
        Args:
            idx: Current index in the data
            signal: Trading signal (-1 to 1)
            capital: Available capital
            
        Returns:
            Position size
        """
        # Scale signal to position size
        position_size = signal * capital
        
        # Apply risk management
        max_position = 0.2 * capital  # Max 20% of capital per position
        position_size = np.clip(position_size, -max_position, max_position)
        
        return position_size
    
    def get_confidence(self, idx):
        """
        Get confidence in the current prediction.
        
        Args:
            idx: Current index in the data
            
        Returns:
            Confidence score (0 to 1)
        """
        # Get confidence from individual strategies
        confidences = [strategy.get_confidence(idx) for strategy in self.strategies]
        
        # Calculate agreement between strategies
        signals = [strategy.position for strategy in self.strategies]
        signal_std = np.std(signals)
        
        # Higher agreement (lower std) means higher confidence
        agreement_factor = np.exp(-signal_std)
        
        # Combine individual confidences with agreement factor
        confidence = np.mean(confidences) * agreement_factor
        
        return confidence
