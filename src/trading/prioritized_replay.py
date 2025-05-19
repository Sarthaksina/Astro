#!/usr/bin/env python
# Cosmic Market Oracle - Prioritized Experience Replay

"""
Prioritized Experience Replay for the Cosmic Market Oracle.

This module implements specialized experience replay with priority based on rare events,
improving sample efficiency by focusing on important transitions.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import logging
from sum_tree import SumTree

from src.models.reinforcement import Experience
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("prioritized_replay")


class SumTree:
    """
    Sum Tree data structure for efficient sampling based on priorities.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the sum tree.
        
        Args:
            capacity: Maximum capacity of the tree
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0
    
    def _propagate(self, idx: int, change: float):
        """
        Propagate a change up through the tree.
        
        Args:
            idx: Index of the node
            change: Change in value
        """
        parent = (idx - 1) // 2
        
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """
        Retrieve sample index based on priority.
        
        Args:
            idx: Index of the node
            s: Priority value
            
        Returns:
            Index of the sample
        """
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """
        Get the total priority.
        
        Returns:
            Total priority
        """
        return self.tree[0]
    
    def add(self, p: float, data: Any):
        """
        Add a sample with priority.
        
        Args:
            p: Priority value
            data: Sample data
        """
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write = (self.write + 1) % self.capacity
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, p: float):
        """
        Update the priority of a sample.
        
        Args:
            idx: Index of the sample
            p: New priority value
        """
        change = p - self.tree[idx]
        
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Get a sample based on priority.
        
        Args:
            s: Priority value
            
        Returns:
            Tuple of (index, priority, data)
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for reinforcement learning.
    
    This buffer prioritizes experiences based on their TD error, focusing
    on transitions that are surprising or rare.
    """
    
    def __init__(self, 
                 capacity: int = 100000, 
                 alpha: float = 0.6, 
                 beta: float = 0.4, 
                 beta_increment: float = 0.001,
                 epsilon: float = 0.01,
                 max_priority: float = 1.0):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
            alpha: Priority exponent (0 = uniform, 1 = greedy)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Increment for beta over time
            epsilon: Small constant to add to priorities
            max_priority: Maximum priority value
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = max_priority
        self.transitions = 0
    
    def add(self, state, action, reward, next_state, done, error=None):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            error: TD error (if known)
        """
        experience = Experience(state, action, reward, next_state, done)
        
        # Use max priority for new experiences if error is not provided
        priority = self.max_priority if error is None else (abs(error) + self.epsilon) ** self.alpha
        
        # Add to sum tree
        self.tree.add(priority, experience)
        self.transitions += 1
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Size of the batch to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        batch = []
        indices = []
        weights = []
        
        # Get segment size
        segment = self.tree.total() / batch_size
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate max weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()
        max_weight = (p_min * batch_size) ** (-self.beta)
        
        for i in range(batch_size):
            # Get random value in segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            # Get sample
            idx, priority, experience = self.tree.get(s)
            
            # Calculate weight
            p_sample = priority / self.tree.total()
            weight = (p_sample * batch_size) ** (-self.beta)
            weight = weight / max_weight  # Normalize
            
            # Store sample
            batch.append(experience)
            indices.append(idx)
            weights.append(weight)
        
        # Convert to numpy arrays
        states = np.vstack([e.state for e in batch])
        actions = np.vstack([e.action for e in batch])
        rewards = np.vstack([e.reward for e in batch])
        next_states = np.vstack([e.next_state for e in batch])
        dones = np.vstack([e.done for e in batch]).astype(np.uint8)
        weights = np.array(weights)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: List[int], errors: List[float]):
        """
        Update priorities of sampled transitions.
        
        Args:
            indices: Indices of the samples
            errors: TD errors of the samples
        """
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
    
    def __len__(self):
        """
        Get the current size of the buffer.
        
        Returns:
            Current size of the buffer
        """
        return self.tree.n_entries


class RareEventDetector:
    """
    Detector for rare events in market data.
    
    This class identifies rare or significant events in market data,
    such as market crashes, regime changes, or unusual price movements.
    """
    
    def __init__(self, 
                 window_size: int = 20, 
                 threshold: float = 2.0,
                 volatility_window: int = 60,
                 regime_change_threshold: float = 0.05):
        """
        Initialize the rare event detector.
        
        Args:
            window_size: Size of the window for detecting events
            threshold: Threshold for detecting events (in standard deviations)
            volatility_window: Window size for calculating volatility
            regime_change_threshold: Threshold for detecting regime changes
        """
        self.window_size = window_size
        self.threshold = threshold
        self.volatility_window = volatility_window
        self.regime_change_threshold = regime_change_threshold
        
        # Initialize history
        self.price_history = deque(maxlen=max(window_size, volatility_window))
        self.return_history = deque(maxlen=max(window_size, volatility_window))
        self.volatility_history = deque(maxlen=window_size)
        self.detected_events = []
    
    def update(self, price: float) -> Dict:
        """
        Update the detector with a new price.
        
        Args:
            price: New price
            
        Returns:
            Dictionary with detection results
        """
        # Add price to history
        self.price_history.append(price)
        
        # Calculate return if possible
        if len(self.price_history) > 1:
            ret = (price / self.price_history[-2]) - 1
            self.return_history.append(ret)
        else:
            self.return_history.append(0)
        
        # Calculate volatility if possible
        if len(self.return_history) >= self.volatility_window:
            volatility = np.std(list(self.return_history)[-self.volatility_window:])
            self.volatility_history.append(volatility)
        else:
            self.volatility_history.append(0)
        
        # Check for rare events
        events = {}
        
        # Check for price jumps
        if len(self.return_history) >= self.window_size:
            returns = list(self.return_history)[-self.window_size:]
            mean_return = np.mean(returns[:-1])
            std_return = np.std(returns[:-1])
            
            if std_return > 0:
                z_score = (returns[-1] - mean_return) / std_return
                
                if abs(z_score) > self.threshold:
                    events["price_jump"] = {
                        "z_score": z_score,
                        "return": returns[-1],
                        "mean_return": mean_return,
                        "std_return": std_return
                    }
        
        # Check for volatility jumps
        if len(self.volatility_history) >= self.window_size:
            volatilities = list(self.volatility_history)[-self.window_size:]
            mean_vol = np.mean(volatilities[:-1])
            std_vol = np.std(volatilities[:-1])
            
            if std_vol > 0:
                z_score = (volatilities[-1] - mean_vol) / std_vol
                
                if z_score > self.threshold:
                    events["volatility_jump"] = {
                        "z_score": z_score,
                        "volatility": volatilities[-1],
                        "mean_volatility": mean_vol,
                        "std_volatility": std_vol
                    }
        
        # Check for regime changes
        if len(self.volatility_history) >= self.window_size:
            current_vol = self.volatility_history[-1]
            past_vol = np.mean(list(self.volatility_history)[:-1])
            
            if abs(current_vol - past_vol) / past_vol > self.regime_change_threshold:
                events["regime_change"] = {
                    "current_volatility": current_vol,
                    "past_volatility": past_vol,
                    "change_ratio": (current_vol - past_vol) / past_vol
                }
        
        # Store detected events
        if events:
            self.detected_events.append({
                "price": price,
                "events": events
            })
        
        return events
    
    def get_priority(self, events: Dict) -> float:
        """
        Calculate priority based on detected events.
        
        Args:
            events: Dictionary with detected events
            
        Returns:
            Priority value
        """
        priority = 1.0  # Base priority
        
        # Increase priority for price jumps
        if "price_jump" in events:
            priority += abs(events["price_jump"]["z_score"])
        
        # Increase priority for volatility jumps
        if "volatility_jump" in events:
            priority += events["volatility_jump"]["z_score"]
        
        # Increase priority for regime changes
        if "regime_change" in events:
            priority += abs(events["regime_change"]["change_ratio"]) * 10
        
        return priority


class RareEventPrioritizedReplay:
    """
    Specialized experience replay with priority based on rare market events.
    
    This class combines prioritized experience replay with rare event detection
    to focus on important market transitions.
    """
    
    def __init__(self, 
                 capacity: int = 100000, 
                 alpha: float = 0.6, 
                 beta: float = 0.4, 
                 beta_increment: float = 0.001,
                 epsilon: float = 0.01,
                 max_priority: float = 1.0,
                 window_size: int = 20,
                 threshold: float = 2.0,
                 event_priority_weight: float = 0.5):
        """
        Initialize the rare event prioritized replay.
        
        Args:
            capacity: Maximum capacity of the buffer
            alpha: Priority exponent (0 = uniform, 1 = greedy)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Increment for beta over time
            epsilon: Small constant to add to priorities
            max_priority: Maximum priority value
            window_size: Size of the window for detecting events
            threshold: Threshold for detecting events (in standard deviations)
            event_priority_weight: Weight for event-based priority
        """
        self.buffer = PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=alpha,
            beta=beta,
            beta_increment=beta_increment,
            epsilon=epsilon,
            max_priority=max_priority
        )
        
        self.detector = RareEventDetector(
            window_size=window_size,
            threshold=threshold
        )
        
        self.event_priority_weight = event_priority_weight
    
    def add(self, state, action, reward, next_state, done, error=None, price=None):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            error: TD error (if known)
            price: Current price (for event detection)
        """
        # Detect rare events if price is provided
        events = {}
        if price is not None:
            events = self.detector.update(price)
        
        # Calculate combined priority
        if error is not None:
            # Combine TD error and event priority
            event_priority = self.detector.get_priority(events) if events else 1.0
            combined_priority = (1 - self.event_priority_weight) * abs(error) + self.event_priority_weight * event_priority
        else:
            # Use max priority if error is not provided
            combined_priority = None
        
        # Add to buffer
        self.buffer.add(state, action, reward, next_state, done, combined_priority)
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Size of the batch to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        return self.buffer.sample(batch_size)
    
    def update_priorities(self, indices: List[int], errors: List[float]):
        """
        Update priorities of sampled transitions.
        
        Args:
            indices: Indices of the samples
            errors: TD errors of the samples
        """
        self.buffer.update_priorities(indices, errors)
    
    def __len__(self):
        """
        Get the current size of the buffer.
        
        Returns:
            Current size of the buffer
        """
        return len(self.buffer)


class RareEventVisualizer:
    """
    Visualizer for rare events and their impact on replay priorities.
    """
    
    def __init__(self, replay: RareEventPrioritizedReplay):
        """
        Initialize the visualizer.
        
        Args:
            replay: Rare event prioritized replay buffer
        """
        self.replay = replay
    
    def plot_events(self, prices: List[float], save_path: Optional[str] = None):
        """
        Plot detected rare events.
        
        Args:
            prices: List of prices
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot prices
        plt.subplot(2, 1, 1)
        plt.plot(prices, label="Price")
        
        # Mark detected events
        for event in self.replay.detector.detected_events:
            idx = prices.index(event["price"])
            
            if "price_jump" in event["events"]:
                plt.scatter(idx, event["price"], color="red", s=100, marker="o", label="Price Jump" if "price_jump" not in plt.gca().get_legend_handles_labels()[1] else "")
            
            if "volatility_jump" in event["events"]:
                plt.scatter(idx, event["price"], color="green", s=100, marker="s", label="Volatility Jump" if "volatility_jump" not in plt.gca().get_legend_handles_labels()[1] else "")
            
            if "regime_change" in event["events"]:
                plt.scatter(idx, event["price"], color="blue", s=100, marker="^", label="Regime Change" if "regime_change" not in plt.gca().get_legend_handles_labels()[1] else "")
        
        plt.title("Price with Detected Rare Events")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        
        # Plot priorities
        plt.subplot(2, 1, 2)
        priorities = [self.replay.detector.get_priority(event["events"]) for event in self.replay.detector.detected_events]
        
        if priorities:
            indices = [prices.index(event["price"]) for event in self.replay.detector.detected_events]
            plt.bar(indices, priorities, alpha=0.7)
            plt.title("Event Priorities")
            plt.xlabel("Time")
            plt.ylabel("Priority")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
