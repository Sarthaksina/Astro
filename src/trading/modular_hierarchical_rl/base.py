#!/usr/bin/env python
# Cosmic Market Oracle - Modular Hierarchical RL Base Components

"""
Base components for the modular hierarchical reinforcement learning framework.

This module defines the core interfaces and base classes for the modular
hierarchical reinforcement learning system, allowing for flexible integration
of different strategic planning and tactical execution components.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import os
import logging
from abc import ABC, abstractmethod

from src.models.reinforcement import MarketEnvironment, ReplayBuffer, Experience
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("modular_hierarchical_rl")


class StrategicPlannerInterface(ABC):
    """
    Interface for strategic planning components.
    
    Strategic planners are responsible for high-level decision making,
    setting goals for the tactical executor to achieve.
    """
    
    @abstractmethod
    def plan(self, state, evaluate=False):
        """
        Generate a strategic plan (goal) based on the current state.
        
        Args:
            state: Current state
            evaluate: Whether to evaluate (deterministic) or explore
            
        Returns:
            Strategic goal
        """
        pass
    
    @abstractmethod
    def update(self, states, goals, rewards, next_states, dones):
        """
        Update the strategic planner based on experience.
        
        Args:
            states: Batch of states
            goals: Batch of goals
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            
        Returns:
            Dictionary with update metrics
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """
        Save the strategic planner.
        
        Args:
            path: Path to save to
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """
        Load the strategic planner.
        
        Args:
            path: Path to load from
        """
        pass


class TacticalExecutorInterface(ABC):
    """
    Interface for tactical execution components.
    
    Tactical executors are responsible for low-level decision making,
    executing actions to achieve the goals set by the strategic planner.
    """
    
    @abstractmethod
    def execute(self, state, goal, evaluate=False):
        """
        Execute a tactical action based on the current state and goal.
        
        Args:
            state: Current state
            goal: Strategic goal
            evaluate: Whether to evaluate (deterministic) or explore
            
        Returns:
            Tactical action
        """
        pass
    
    @abstractmethod
    def update(self, states, goals, actions, rewards, next_states, next_goals, dones):
        """
        Update the tactical executor based on experience.
        
        Args:
            states: Batch of states
            goals: Batch of goals
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            next_goals: Batch of next goals
            dones: Batch of done flags
            
        Returns:
            Dictionary with update metrics
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """
        Save the tactical executor.
        
        Args:
            path: Path to save to
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """
        Load the tactical executor.
        
        Args:
            path: Path to load from
        """
        pass


class ModularHierarchicalRLAgent:
    """
    Modular Hierarchical Reinforcement Learning Agent.
    
    This agent combines a strategic planner and a tactical executor
    to create a hierarchical decision-making system.
    """
    
    def __init__(self, 
                 strategic_planner: StrategicPlannerInterface,
                 tactical_executor: TacticalExecutorInterface,
                 strategic_horizon: int = 20,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the modular hierarchical RL agent.
        
        Args:
            strategic_planner: Strategic planning component
            tactical_executor: Tactical execution component
            strategic_horizon: Horizon for strategic decisions
            device: Device to run on
        """
        self.strategic_planner = strategic_planner
        self.tactical_executor = tactical_executor
        self.strategic_horizon = strategic_horizon
        self.device = device
        
        # Initialize goal tracking
        self.current_goal = None
        self.goal_steps = 0
        
        # Initialize buffers
        self.strategic_buffer = ReplayBuffer(100000)
        self.tactical_buffer = ReplayBuffer(1000000)
        
        # Initialize metrics
        self.strategic_metrics = []
        self.tactical_metrics = []
    
    def select_action(self, state, evaluate=False):
        """
        Select an action using the hierarchical policy.
        
        Args:
            state: Current state
            evaluate: Whether to evaluate (deterministic) or explore
            
        Returns:
            Selected action
        """
        # Check if we need a new strategic goal
        if self.current_goal is None or self.goal_steps >= self.strategic_horizon:
            # Generate new goal using strategic planner
            self.current_goal = self.strategic_planner.plan(state, evaluate)
            
            # Reset goal steps
            self.goal_steps = 0
        
        # Increment goal steps
        self.goal_steps += 1
        
        # Execute tactical action based on state and goal
        action = self.tactical_executor.execute(state, self.current_goal, evaluate)
        
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffers.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Store in tactical buffer
        self.tactical_buffer.add(
            np.concatenate([state, self.current_goal]),
            action,
            reward,
            np.concatenate([next_state, self.current_goal]),
            done
        )
        
        # If strategic step, store in strategic buffer
        if self.goal_steps == self.strategic_horizon or done:
            self.strategic_buffer.add(
                state,
                self.current_goal,
                reward,  # Use immediate reward for simplicity
                next_state,
                done
            )
    
    def update(self, batch_size=256):
        """
        Update the agent's components.
        
        Args:
            batch_size: Batch size for updates
            
        Returns:
            Dictionary with update metrics
        """
        metrics = {}
        
        # Update tactical executor if enough samples
        if len(self.tactical_buffer) >= batch_size:
            # Sample from tactical buffer
            states_goals, actions, rewards, next_states_goals, dones = self.tactical_buffer.sample(batch_size)
            
            # Split states and goals
            states = states_goals[:, :-self.current_goal.shape[0]]
            goals = states_goals[:, -self.current_goal.shape[0]:]
            next_states = next_states_goals[:, :-self.current_goal.shape[0]]
            next_goals = next_states_goals[:, -self.current_goal.shape[0]:]
            
            # Update tactical executor
            tactical_metrics = self.tactical_executor.update(
                states, goals, actions, rewards, next_states, next_goals, dones
            )
            
            # Store metrics
            metrics.update({"tactical_" + k: v for k, v in tactical_metrics.items()})
            self.tactical_metrics.append(tactical_metrics)
        
        # Update strategic planner if enough samples
        if len(self.strategic_buffer) >= batch_size:
            # Sample from strategic buffer
            states, goals, rewards, next_states, dones = self.strategic_buffer.sample(batch_size)
            
            # Update strategic planner
            strategic_metrics = self.strategic_planner.update(
                states, goals, rewards, next_states, dones
            )
            
            # Store metrics
            metrics.update({"strategic_" + k: v for k, v in strategic_metrics.items()})
            self.strategic_metrics.append(strategic_metrics)
        
        return metrics
    
    def save(self, path):
        """
        Save the agent's components.
        
        Args:
            path: Path to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save components
        self.strategic_planner.save(path + "_strategic")
        self.tactical_executor.save(path + "_tactical")
    
    def load(self, path):
        """
        Load the agent's components.
        
        Args:
            path: Path to load from
        """
        # Load components
        self.strategic_planner.load(path + "_strategic")
        self.tactical_executor.load(path + "_tactical")
