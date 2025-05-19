#!/usr/bin/env python
# Cosmic Market Oracle - PPO Tactical Executor

"""
Proximal Policy Optimization based tactical executor for the modular hierarchical RL framework.

This module implements a tactical executor that uses PPO for low-level action execution
in the hierarchical reinforcement learning system, optimized for financial market environments.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from src.trading.modular_hierarchical_rl.base import TacticalExecutorInterface
from src.trading.modular_hierarchical_rl.networks import SharedNetwork, BaseNetwork
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("ppo_executor")

# Default configuration values
DEFAULT_CONFIG = {
    "hidden_dim": 256,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "ppo_epochs": 10,
    "mini_batch_size": 64
}

# Configure logging
logger = setup_logger("ppo_executor")


class PPOActor(nn.Module):
    """
    Actor network for PPO algorithm.
    
    Uses shared network components for common processing.
    """
    
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim=256):
        """
        Initialize the actor network.
        
        Args:
            state_dim: Dimension of the state space
            goal_dim: Dimension of the goal space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        # Combined state and goal dimension
        combined_dim = state_dim + goal_dim
        
        # Shared network for common processing
        self.shared_net = SharedNetwork(combined_dim, hidden_dim)
        
        # Output layers for action distribution
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_output_weights(self.mean)
        self._init_output_weights(self.log_std)
    
    def _init_output_weights(self, layer):
        """Initialize output layer weights."""
        nn.init.orthogonal_(layer.weight, gain=0.01)
        nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state, goal):
        """
        Forward pass through the network.
        
        Args:
            state: Input state
            goal: Input goal
            
        Returns:
            Action distribution
        """
        # Combine state and goal
        x = torch.cat([state, goal], dim=1)
        
        # Forward pass through shared layers
        x = self.shared_net(x)
        
        # Get mean and log_std
        mean = self.mean(x)
        log_std = self.log_std(x)
        
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        # Create normal distribution
        dist = Normal(mean, std)
        
        return dist
    
    def get_action(self, state, goal, deterministic=False):
        """
        Get an action from the policy.
        
        Args:
            state: Input state
            goal: Input goal
            deterministic: Whether to use deterministic actions
            
        Returns:
            Action and log probability
        """
        # Get distribution
        dist = self.forward(state, goal)
        
        # Sample action
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        
        # Get log probability
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Apply tanh to bound actions between -1 and 1
        action = torch.tanh(action)
        
        return action, log_prob


class PPOCritic(nn.Module):
    """
    Critic network for PPO algorithm.
    """
    
    def __init__(self, state_dim, goal_dim, hidden_dim=256):
        """
        Initialize the critic network.
        
        Args:
            state_dim: Dimension of the state space
            goal_dim: Dimension of the goal space
            hidden_dim: Dimension of hidden layers
        """
        super(PPOCritic, self).__init__()
        
        # Combined state and goal dimension
        combined_dim = state_dim + goal_dim
        
        # Network layers
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Special initialization for output layer
        nn.init.orthogonal_(self.fc4.weight, gain=1.0)
        nn.init.constant_(self.fc4.bias, 0.0)
    
    def forward(self, state, goal):
        """
        Forward pass through the network.
        
        Args:
            state: Input state
            goal: Input goal
            
        Returns:
            Value estimate
        """
        # Combine state and goal
        x = torch.cat([state, goal], dim=1)
        
        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc4(x)
        
        return value


class PPOTacticalExecutor(TacticalExecutorInterface):
    """
    Tactical executor using Proximal Policy Optimization.
    
    This executor uses PPO to learn low-level actions that achieve
    the strategic goals set by the planner.
    """
    
    def __init__(self, 
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10,
                 mini_batch_size: int = 64,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the PPO tactical executor.
        
        Args:
            state_dim: Dimension of the state space
            goal_dim: Dimension of the goal space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio
            value_coef: Value loss coefficient
            entropy_coef: Entropy loss coefficient
            max_grad_norm: Maximum gradient norm
            ppo_epochs: Number of PPO epochs per update
            mini_batch_size: Mini-batch size for PPO updates
            device: Device to run on
        """
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.device = device
        
        # Create actor and critic networks
        self.actor = PPOActor(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        self.critic = PPOCritic(
            state_dim=state_dim,
            goal_dim=goal_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=learning_rate
        )
        
        # Initialize metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.total_losses = []
    
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
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        
        # Get action from actor
        with torch.no_grad():
            action, _ = self.actor.get_action(state_tensor, goal_tensor, deterministic=evaluate)
        
        # Convert to numpy
        action = action.cpu().numpy().squeeze()
        
        return action
    
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
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_goals = torch.FloatTensor(next_goals).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Get old action log probs and values
        with torch.no_grad():
            old_dist = self.actor(states, goals)
            old_log_probs = old_dist.log_prob(torch.atanh(torch.clamp(actions, -0.999, 0.999))).sum(dim=-1, keepdim=True)
            old_values = self.critic(states, goals)
            next_values = self.critic(next_states, next_goals)
        
        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = old_values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + old_values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        total_losses = []
        
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = np.random.permutation(len(states))
            batch_size = min(self.mini_batch_size, len(states))
            num_batches = len(states) // batch_size
            
            for i in range(num_batches):
                # Get mini-batch indices
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                
                # Get mini-batch data
                batch_states = states[batch_indices]
                batch_goals = goals[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Get new action distribution and values
                new_dist = self.actor(batch_states, batch_goals)
                new_log_probs = new_dist.log_prob(torch.atanh(torch.clamp(batch_actions, -0.999, 0.999))).sum(dim=-1, keepdim=True)
                new_values = self.critic(batch_states, batch_goals)
                entropy = new_dist.entropy().mean()
                
                # Compute ratio and clipped ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                
                # Compute actor loss
                surrogate1 = ratio * batch_advantages
                surrogate2 = clipped_ratio * batch_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Compute critic loss
                critic_loss = F.mse_loss(new_values, batch_returns)
                
                # Compute entropy loss
                entropy_loss = -entropy
                
                # Compute total loss
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Update actor
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                # Apply gradients
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Store losses
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(total_loss.item())
        
        # Store average losses
        actor_loss_val = np.mean(actor_losses)
        critic_loss_val = np.mean(critic_losses)
        entropy_loss_val = np.mean(entropy_losses)
        total_loss_val = np.mean(total_losses)
        
        self.actor_losses.append(actor_loss_val)
        self.critic_losses.append(critic_loss_val)
        self.entropy_losses.append(entropy_loss_val)
        self.total_losses.append(total_loss_val)
        
        return {
            "actor_loss": actor_loss_val,
            "critic_loss": critic_loss_val,
            "entropy_loss": entropy_loss_val,
            "total_loss": total_loss_val
        }
    
    def save(self, path):
        """
        Save the tactical executor.
        
        Args:
            path: Path to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save actor and critic
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "entropy_losses": self.entropy_losses,
            "total_losses": self.total_losses
        }, path + ".pt")
    
    def load(self, path):
        """
        Load the tactical executor.
        
        Args:
            path: Path to load from
        """
        # Load actor and critic
        checkpoint = torch.load(path + ".pt", map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.actor_losses = checkpoint.get("actor_losses", [])
        self.critic_losses = checkpoint.get("critic_losses", [])
        self.entropy_losses = checkpoint.get("entropy_losses", [])
        self.total_losses = checkpoint.get("total_losses", [])
