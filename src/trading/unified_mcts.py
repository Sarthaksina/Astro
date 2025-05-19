#!/usr/bin/env python
# Cosmic Market Oracle - Unified Monte Carlo Tree Search

"""
Unified Monte Carlo Tree Search implementation for the Cosmic Market Oracle.

This module provides a comprehensive implementation of Monte Carlo Tree Search
for strategic planning in trading, combining the best features of previous implementations
and ensuring compatibility with both standalone usage and integration with hierarchical RL.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import time
import copy
from collections import defaultdict

from src.models.reinforcement import MarketEnvironment
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("unified_mcts")


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.
    
    Each node represents a state in the decision process and tracks statistics
    about the expected rewards from actions taken from this state.
    """
    
    def __init__(self, 
                 state=None, 
                 parent=None, 
                 action=None, 
                 prior=0.0,
                 c_puct=1.0):
        """
        Initialize a MCTS node.
        
        Args:
            state: The state represented by this node
            parent: Parent node
            action: Action that led to this node from parent
            prior: Prior probability of selecting this node
            c_puct: Exploration constant for PUCT algorithm
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.c_puct = c_puct
        
        # Children nodes
        self.children = {}
        
        # Statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        
        # Expanded flag
        self.expanded = False
    
    def expand(self, actions, priors, next_states=None, rewards=None):
        """
        Expand the node with possible actions and their priors.
        
        Args:
            actions: List of possible actions
            priors: Prior probabilities for each action
            next_states: Optional list of next states for each action
            rewards: Optional list of rewards for each action
        """
        self.expanded = True
        
        # Default reward and next states if not provided
        if rewards is None:
            rewards = [0.0] * len(actions)
            
        if next_states is None:
            next_states = [None] * len(actions)
        
        for i, action in enumerate(actions):
            # Convert action to tuple if it's a numpy array for hashability
            if isinstance(action, np.ndarray):
                action_key = tuple(action.flatten())
            else:
                action_key = action
                
            # Create child node if it doesn't exist
            if action_key not in self.children:
                child = MCTSNode(
                    state=next_states[i],
                    parent=self,
                    action=action,
                    prior=priors[i],
                    c_puct=self.c_puct
                )
                
                # Set reward
                child.reward = rewards[i] if i < len(rewards) else 0.0
                
                # Add child to children
                self.children[action_key] = child
    
    def update(self, value):
        """
        Update node statistics.
        
        Args:
            value: Value to update with
        """
        self.visit_count += 1
        self.value_sum += value
    
    def get_value(self):
        """
        Get the average value of this node.
        
        Returns:
            Average value
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_ucb_score(self, parent_visit_count, exploration_weight=1.0):
        """
        Calculate the UCB score for this node.
        
        Args:
            parent_visit_count: Visit count of parent node
            exploration_weight: Weight for exploration term
            
        Returns:
            UCB score
        """
        # Avoid division by zero
        if self.visit_count == 0:
            return float('inf')
        
        # Exploitation term
        exploitation = self.get_value()
        
        # Exploration term (PUCT algorithm)
        exploration = exploration_weight * self.prior * math.sqrt(
            parent_visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def select_child(self, exploration_weight=1.0):
        """
        Select the child with the highest UCB score.
        
        Args:
            exploration_weight: Weight for exploration term
            
        Returns:
            Selected child node and action
        """
        # Find child with highest UCB score
        best_score = float('-inf')
        best_child = None
        best_action = None
        
        for action, child in self.children.items():
            ucb_score = child.get_ucb_score(self.visit_count, exploration_weight)
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                best_action = child.action
        
        return best_child, best_action
    
    def is_leaf(self):
        """
        Check if this node is a leaf node (no children).
        
        Returns:
            True if leaf node, False otherwise
        """
        return len(self.children) == 0
    
    def is_fully_expanded(self):
        """
        Check if this node is fully expanded.
        
        Returns:
            True if fully expanded, False otherwise
        """
        return self.expanded
    
    def backup(self, value):
        """
        Backup the value up the tree.
        
        Args:
            value: Value to backup
        """
        # Update this node
        self.update(value)
        
        # Recursively update parent
        if self.parent is not None:
            self.parent.backup(value)


class MCTSPredictor:
    """
    Neural network for predicting state values and action probabilities.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, device=None):
        """
        Initialize the predictor.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Shared network
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(self.device)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1)
        ).to(self.device)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Value between -1 and 1
        ).to(self.device)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def to(self, device):
        """
        Move networks to device.
        
        Args:
            device: Device to move to
            
        Returns:
            Self
        """
        self.device = device
        self.shared = self.shared.to(device)
        self.policy_head = self.policy_head.to(device)
        self.value_head = self.value_head.to(device)
        return self
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state
            
        Returns:
            Tuple of (policy, value)
        """
        # Convert state to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Shared features
        features = self.shared(state)
        
        # Policy and value
        policy = self.policy_head(features)
        value = self.value_head(features)
        
        return policy, value
    
    def predict(self, state):
        """
        Predict policy and value for a state.
        
        Args:
            state: Input state
            
        Returns:
            Tuple of (policy, value)
        """
        with torch.no_grad():
            policy, value = self.forward(state)
            
            # Convert to numpy
            policy = policy.cpu().numpy()
            value = value.cpu().numpy()
            
            return policy, value


class MCTS:
    """
    Monte Carlo Tree Search for strategic planning.
    
    This class implements the MCTS algorithm for finding optimal strategic decisions
    by simulating possible future states and actions.
    """
    
    def __init__(self, 
                 model=None,
                 environment=None,
                 num_simulations=50,
                 exploration_weight=1.0,
                 discount_factor=0.99,
                 dirichlet_alpha=0.3,
                 dirichlet_weight=0.25,
                 temperature=1.0,
                 num_actions=5,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the MCTS.
        
        Args:
            model: Model for predicting action priors and values
            environment: Environment for simulating actions
            num_simulations: Number of simulations to run
            exploration_weight: Weight for exploration term in UCB
            discount_factor: Discount factor for future rewards
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_weight: Weight for Dirichlet noise
            temperature: Temperature for action selection
            num_actions: Number of discrete actions per dimension
            device: Device to run on
        """
        self.model = model
        self.environment = environment
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.discount_factor = discount_factor
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.temperature = temperature
        self.num_actions = num_actions
        self.device = device
        
        # Root node
        self.root = None
    
    def search(self, state):
        """
        Run MCTS search from a state.
        
        Args:
            state: Root state
            
        Returns:
            Action probabilities
        """
        # Create root node
        self.root = MCTSNode(state=state)
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Select and expand
            node, search_path = self._select_node(self.root)
            
            # If node is not expanded, expand it
            if not node.is_fully_expanded():
                value = self._expand_node(node)
                
                # Backup value
                for n in reversed(search_path):
                    n.update(value)
                    value = n.reward + self.discount_factor * value
            else:
                # Simulate and backup
                value = self._simulate_and_backpropagate(node, search_path)
        
        # Get action probabilities
        action_probs = self._get_action_probs()
        
        return action_probs
    
    def _select_node(self, node):
        """
        Select a node to expand using UCB.
        
        Args:
            node: Starting node
            
        Returns:
            Selected node and search path
        """
        search_path = [node]
        
        while not node.is_leaf() and node.is_fully_expanded():
            # Select child with highest UCB score
            node, _ = node.select_child(self.exploration_weight)
            search_path.append(node)
        
        return node, search_path
    
    def _expand_node(self, node):
        """
        Expand a node with possible actions and their priors.
        
        Args:
            node: Node to expand
            
        Returns:
            Value estimate for the node
        """
        # Get state
        state = node.state
        
        # Get action priors and value from model
        if self.model is not None:
            # Use model to predict priors and value
            priors, value = self.model.predict(state)
            priors = priors.flatten()
            value = value.item()
            
            # Get possible actions
            possible_actions = self._get_possible_actions(state)
            
            # Add Dirichlet noise to root node for exploration
            if node == self.root:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(possible_actions))
                priors = (1 - self.dirichlet_weight) * priors + self.dirichlet_weight * noise
            
            # Expand node
            node.expand(possible_actions, priors)
        else:
            # Without a model, use environment to simulate
            if self.environment is not None:
                # Create environment copy
                env = copy.deepcopy(self.environment)
                
                # Set environment state
                env.reset()
                env.state = state
                
                # Get possible actions
                possible_actions = self._get_possible_actions(state)
                
                # Initialize priors, next states, and rewards
                priors = np.ones(len(possible_actions)) / len(possible_actions)
                next_states = []
                rewards = []
                
                # Simulate each action
                for action in possible_actions:
                    env_copy = copy.deepcopy(env)
                    next_state, reward, _, _ = env_copy.step(action)
                    next_states.append(next_state)
                    rewards.append(reward)
                
                # Expand node
                node.expand(possible_actions, priors, next_states, rewards)
                
                # Use average reward as value estimate
                value = np.mean(rewards) if rewards else 0.0
            else:
                # Without model or environment, use random expansion
                possible_actions = self._get_possible_actions(state)
                priors = np.ones(len(possible_actions)) / len(possible_actions)
                node.expand(possible_actions, priors)
                value = 0.0
        
        return value
    
    def _get_possible_actions(self, state):
        """
        Get possible actions from the environment.
        
        Args:
            state: Current state
            
        Returns:
            List of possible actions
        """
        # If environment is available, use its action space
        if self.environment is not None:
            action_space = self.environment.action_space
            
            # For discrete action spaces
            if hasattr(action_space, 'n'):
                return list(range(action_space.n))
            
            # For continuous action spaces, discretize
            elif hasattr(action_space, 'shape'):
                action_dim = action_space.shape[0]
                action_low = action_space.low
                action_high = action_space.high
                
                # Create a grid of actions
                if action_dim == 1:
                    # 1D action space
                    actions = np.linspace(action_low[0], action_high[0], self.num_actions)
                    return [np.array([a]) for a in actions]
                else:
                    # Multi-dimensional action space (simplified approach)
                    actions = []
                    for dim in range(action_dim):
                        dim_actions = np.linspace(action_low[dim], action_high[dim], self.num_actions)
                        for a in dim_actions:
                            action = np.zeros(action_dim)
                            action[dim] = a
                            actions.append(action)
                    
                    return actions
        
        # Default: return a single action
        return [0]
    
    def _simulate_and_backpropagate(self, node, search_path):
        """
        Simulate from a node and backpropagate the value.
        
        Args:
            node: Node to simulate from
            search_path: Path from root to node
            
        Returns:
            Simulated value
        """
        # If environment is available, use it to simulate
        if self.environment is not None and node.state is not None:
            # Create environment copy
            env = copy.deepcopy(self.environment)
            
            # Set environment state
            env.reset()
            env.state = node.state
            
            # Simulate random actions until done or max depth
            max_depth = 10
            depth = 0
            done = False
            total_reward = 0.0
            discount = 1.0
            
            while not done and depth < max_depth:
                # Select random action
                action = env.action_space.sample()
                
                # Take step
                _, reward, done, _ = env.step(action)
                
                # Update total reward
                total_reward += discount * reward
                discount *= self.discount_factor
                depth += 1
            
            value = total_reward
        else:
            # Without environment, use model to estimate value
            if self.model is not None and node.state is not None:
                _, value = self.model.predict(node.state)
                value = value.item()
            else:
                # Random value
                value = np.random.uniform(-1.0, 1.0)
        
        # Backpropagate
        for n in reversed(search_path):
            n.update(value)
            value = n.reward + self.discount_factor * value
        
        return value
    
    def _get_action_probs(self):
        """
        Get action probabilities based on visit counts.
        
        Returns:
            Dictionary mapping actions to probabilities
        """
        visits = {action: child.visit_count for action, child in self.root.children.items()}
        total_visits = sum(visits.values())
        
        # Apply temperature
        if self.temperature == 0:
            # Deterministic selection of best action
            best_action = max(visits.items(), key=lambda x: x[1])[0]
            action_probs = {action: 0.0 for action in visits}
            action_probs[best_action] = 1.0
        else:
            # Apply temperature to visit counts
            scaled_visits = {action: count ** (1 / self.temperature) for action, count in visits.items()}
            total_scaled = sum(scaled_visits.values())
            
            # Convert to probabilities
            action_probs = {action: count / total_scaled for action, count in scaled_visits.items()}
        
        return action_probs
    
    def select_action(self, state, deterministic=False):
        """
        Select an action using MCTS.
        
        Args:
            state: Current state
            deterministic: Whether to select deterministically
            
        Returns:
            Selected action
        """
        # Perform search
        action_probs = self.search(state)
        
        if deterministic:
            # Select action with highest probability
            action = max(action_probs.items(), key=lambda x: x[1])[0]
        else:
            # Sample action based on probabilities
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            action_idx = np.random.choice(len(actions), p=probs)
            action = actions[action_idx]
        
        # Convert tuple back to numpy array if needed
        if isinstance(action, tuple):
            action = np.array(action)
        
        return action


# For backward compatibility
MCTSModel = MCTSPredictor
