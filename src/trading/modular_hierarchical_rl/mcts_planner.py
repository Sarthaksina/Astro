#!/usr/bin/env python
# Cosmic Market Oracle - MCTS Strategic Planner

"""
Monte Carlo Tree Search based strategic planner for the modular hierarchical RL framework.

This module implements a strategic planner that uses Monte Carlo Tree Search
for long-term planning in the hierarchical reinforcement learning system.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from src.trading.modular_hierarchical_rl.base import StrategicPlannerInterface
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("mcts_planner")

# Default configuration values
DEFAULT_CONFIG = {
    "hidden_dim": 256,
    "num_simulations": 50,
    "exploration_weight": 1.0,
    "learning_rate": 0.0003
}


class MCTSStrategicPlanner(StrategicPlannerInterface):
    """
    Strategic planner using Monte Carlo Tree Search.
    
    This planner uses MCTS to generate long-term strategic goals
    by exploring possible future states and their outcomes.
    """
    
    def __init__(self, 
                 state_dim: int,
                 goal_dim: int,
                 hidden_dim: int = 256,
                 num_simulations: int = 50,
                 exploration_weight: float = 1.0,
                 learning_rate: float = 0.0003,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the MCTS strategic planner.
        
        Args:
            state_dim: Dimension of the state space
            goal_dim: Dimension of the goal space
            hidden_dim: Dimension of hidden layers
            num_simulations: Number of MCTS simulations
            exploration_weight: Weight for exploration in MCTS
            learning_rate: Learning rate for optimization
            device: Device to run on
        """
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.hidden_dim = hidden_dim
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.learning_rate = learning_rate
        self.device = device
        
        # Create MCTS predictor (neural network)
        self.predictor = MCTSPredictor(
            state_dim=state_dim,
            action_dim=goal_dim,  # Goals are the "actions" for the strategic planner
            hidden_dim=hidden_dim,
            device=device
        )
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.predictor.parameters(), lr=learning_rate
        )
        
        # Create MCTS instance
        self.mcts = MCTS(
            model=self.predictor,
            num_simulations=num_simulations,
            exploration_weight=exploration_weight,
            device=device
        )
        
        # Initialize metrics
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []
    
    def plan(self, state, evaluate=False):
        """
        Generate a strategic plan (goal) based on the current state.
        
        Args:
            state: Current state
            evaluate: Whether to evaluate (deterministic) or explore
            
        Returns:
            Strategic goal
        """
        # Use MCTS to select a goal
        goal = self.mcts.select_action(state, deterministic=evaluate)
        
        return goal
    
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
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get predictions from model
        goal_probs, values = self.predictor.forward(states)
        
        # Calculate policy loss (cross-entropy with goals as targets)
        # We need to convert goals to probability distributions
        goal_probs_target = F.softmax(goals, dim=1)
        policy_loss = -torch.sum(goal_probs_target * torch.log(goal_probs + 1e-10)) / states.size(0)
        
        # Calculate value loss (MSE with rewards as targets)
        value_loss = F.mse_loss(values, rewards)
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Store metrics
        policy_loss_val = policy_loss.item()
        value_loss_val = value_loss.item()
        total_loss_val = total_loss.item()
        
        self.policy_losses.append(policy_loss_val)
        self.value_losses.append(value_loss_val)
        self.total_losses.append(total_loss_val)
        
        return {
            "policy_loss": policy_loss_val,
            "value_loss": value_loss_val,
            "total_loss": total_loss_val
        }
    
    def save(self, path):
        """
        Save the strategic planner.
        
        Args:
            path: Path to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and optimizer
        torch.save({
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "total_losses": self.total_losses
        }, path + ".pt")
    
    def load(self, path):
        """
        Load the strategic planner.
        
        Args:
            path: Path to load from
        """
        # Load model and optimizer
        checkpoint = torch.load(path + ".pt", map_location=self.device)
        
        self.predictor.load_state_dict(checkpoint["predictor"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.policy_losses = checkpoint.get("policy_losses", [])
        self.value_losses = checkpoint.get("value_losses", [])
        self.total_losses = checkpoint.get("total_losses", [])
