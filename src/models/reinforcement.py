#!/usr/bin/env python
# Cosmic Market Oracle - Reinforcement Learning Models

"""
Reinforcement Learning models for the Cosmic Market Oracle.

This module implements advanced reinforcement learning techniques for trading strategies,
including hierarchical RL, PPO, Monte Carlo Tree Search, and specialized experience replay.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
# Update to use gymnasium instead of gym for Python 3.10 compatibility
import gymnasium as gym
from gymnasium import spaces
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import logging

from src.trading.strategy_framework import BaseStrategy
from src.trading.backtest import BacktestEngine
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("reinforcement_learning")

# Define experience tuple for replay buffer
Experience = namedtuple('Experience', 
                        ['state', 'action', 'reward', 'next_state', 'done'])


class MarketEnvironment(gym.Env):
    """
    Sophisticated market simulation environment with realistic friction.
    
    This environment simulates trading in financial markets with realistic
    features like transaction costs, slippage, and market impact.
    """
    
    def __init__(self, 
                 market_data: pd.DataFrame,
                 planetary_data: pd.DataFrame,
                 initial_balance: float = 100000.0,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 market_impact: float = 0.0001,
                 window_size: int = 30,
                 reward_scaling: float = 0.01,
                 max_position: float = 1.0,
                 use_features: bool = True):
        """
        Initialize the market environment.
        
        Args:
            market_data: DataFrame with market data
            planetary_data: DataFrame with planetary data
            initial_balance: Initial account balance
            transaction_cost: Transaction cost as a fraction of trade value
            slippage: Slippage as a fraction of price
            market_impact: Market impact as a fraction of price per unit of position
            window_size: Size of the observation window
            reward_scaling: Scaling factor for rewards
            max_position: Maximum allowed position size as a fraction of balance
            use_features: Whether to use engineered features in the state
        """
        super(MarketEnvironment, self).__init__()
        
        # Store parameters
        self.market_data = market_data
        self.planetary_data = planetary_data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.market_impact = market_impact
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.max_position = max_position
        self.use_features = use_features
        
        # Calculate feature dimension
        self.market_dim = len(market_data.columns)
        self.planetary_dim = len(planetary_data.columns) if planetary_data is not None else 0
        
        # Define action and observation spaces
        # Action: -1 to 1 (short to long)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation: market data, planetary data, position, balance
        if self.use_features:
            obs_dim = (self.market_dim + self.planetary_dim) * window_size + 2
        else:
            obs_dim = 5 * window_size + 2  # OHLCV + position + balance
            
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
    
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            Initial observation
        """
        # Reset position and balance
        self.balance = self.initial_balance
        self.position = 0.0
        self.current_step = self.window_size
        self.done = False
        self.history = []
        
        # Get initial observation
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (-1 to 1, representing short to long)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Ensure action is in the correct format
        action = np.clip(action, -1.0, 1.0)[0]
        
        # Get current price and target position
        current_price = self.market_data.iloc[self.current_step]['Close']
        target_position = action * self.max_position * self.balance / current_price
        position_change = target_position - self.position
        
        # Calculate transaction costs and slippage
        transaction_value = abs(position_change * current_price)
        transaction_cost = transaction_value * self.transaction_cost
        slippage_cost = transaction_value * self.slippage
        
        # Calculate market impact
        market_impact = current_price * self.market_impact * abs(position_change)
        execution_price = current_price + np.sign(position_change) * market_impact
        
        # Execute trade
        self.balance -= transaction_cost + slippage_cost
        self.balance -= position_change * execution_price
        self.position = target_position
        
        # Move to the next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.market_data) - 1:
            self.done = True
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get new observation
        observation = self._get_observation()
        
        # Record history
        self.history.append({
            'step': self.current_step,
            'price': current_price,
            'position': self.position,
            'balance': self.balance,
            'action': action,
            'reward': reward
        })
        
        # Return step information
        info = {
            'balance': self.balance,
            'position': self.position,
            'price': current_price,
            'transaction_cost': transaction_cost,
            'slippage_cost': slippage_cost,
            'market_impact': market_impact
        }
        
        return observation, reward, self.done, info
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            Numpy array with the observation
        """
        # Get market data window
        market_window = self.market_data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Get planetary data window if available
        if self.planetary_data is not None:
            planetary_window = self.planetary_data.iloc[self.current_step - self.window_size:self.current_step]
        else:
            planetary_window = None
        
        # Normalize market data
        market_window_norm = (market_window - market_window.mean()) / (market_window.std() + 1e-10)
        
        if self.use_features:
            # Use all features
            market_features = market_window_norm.values.flatten()
            
            if planetary_window is not None:
                # Normalize planetary data
                planetary_window_norm = (planetary_window - planetary_window.mean()) / (planetary_window.std() + 1e-10)
                planetary_features = planetary_window_norm.values.flatten()
                
                # Combine features
                features = np.concatenate([market_features, planetary_features])
            else:
                features = market_features
        else:
            # Use only OHLCV data
            ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in ohlcv_cols if col in market_window.columns]
            features = market_window_norm[available_cols].values.flatten()
        
        # Add position and balance
        position_norm = self.position / (self.max_position * self.initial_balance / market_window['Close'].iloc[-1])
        balance_norm = self.balance / self.initial_balance
        
        # Combine all features
        observation = np.concatenate([features, [position_norm, balance_norm]])
        
        return observation.astype(np.float32)
    
    def _calculate_reward(self):
        """
        Calculate the reward for the current step.
        
        Returns:
            Reward value
        """
        # Get current and previous prices
        current_price = self.market_data.iloc[self.current_step]['Close']
        prev_price = self.market_data.iloc[self.current_step - 1]['Close']
        
        # Calculate price change
        price_change = (current_price - prev_price) / prev_price
        
        # Calculate position value
        position_value = self.position * current_price
        
        # Calculate PnL
        pnl = self.position * (current_price - prev_price)
        
        # Calculate Sharpe-like reward
        if len(self.history) > 0:
            returns = [h['reward'] for h in self.history[-20:]]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) if len(returns) > 1 else 0
        else:
            sharpe = 0
        
        # Combine multiple reward components
        reward = pnl * self.reward_scaling + sharpe * 0.1
        
        # Penalize for excessive risk
        if abs(self.position) > 0.8 * self.max_position * self.balance / current_price:
            reward -= 0.01
        
        # Penalize for bankruptcy
        if self.balance <= 0:
            reward -= 10.0
            self.done = True
        
        return reward
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if len(self.history) == 0:
            return
        
        if mode == 'human':
            # Plot price and position
            plt.figure(figsize=(12, 8))
            
            # Plot price
            ax1 = plt.subplot(2, 1, 1)
            steps = [h['step'] for h in self.history]
            prices = [h['price'] for h in self.history]
            plt.plot(steps, prices, label='Price')
            plt.ylabel('Price')
            plt.legend()
            
            # Plot position
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            positions = [h['position'] for h in self.history]
            plt.plot(steps, positions, label='Position', color='orange')
            plt.ylabel('Position')
            plt.xlabel('Step')
            plt.legend()
            
            plt.tight_layout()
            plt.show()


class ReplayBuffer:
    """
    Experience replay buffer for reinforcement learning.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Size of the batch to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Get the current size of the buffer.
        
        Returns:
            Current size of the buffer
        """
        return len(self.buffer)