#!/usr/bin/env python
# Cosmic Market Oracle - Multi-Agent Consensus Engine

"""
Consensus Engine for the Multi-Agent Orchestration Network.

This module implements strategies for reaching consensus among multiple agents
with different specializations and confidence levels.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from src.trading.multi_agent_orchestration.base import AgentMessage, MessageType

# Configure logging
logger = logging.getLogger(__name__)


class ConsensusStrategy(ABC):
    """Base class for consensus strategies."""
    
    @abstractmethod
    def reach_consensus(self, messages: List[AgentMessage], 
                      agent_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Reach consensus from a set of messages.
        
        Args:
            messages: List of messages to consider
            agent_weights: Dictionary of agent weights
            
        Returns:
            Consensus result
        """
        pass


class WeightedVotingStrategy(ConsensusStrategy):
    """
    Weighted voting strategy for reaching consensus.
    
    This strategy uses weighted voting based on agent weights and message
    confidence to determine the consensus.
    """
    
    def __init__(self, min_confidence: float = 0.6, 
                recency_weight: float = 0.8,
                min_messages: int = 3):
        """
        Initialize the weighted voting strategy.
        
        Args:
            min_confidence: Minimum confidence for inclusion in consensus
            recency_weight: Weight given to recency (0-1)
            min_messages: Minimum number of messages for valid consensus
        """
        self.min_confidence = min_confidence
        self.recency_weight = recency_weight
        self.min_messages = min_messages
        
    def reach_consensus(self, messages: List[AgentMessage], 
                      agent_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Reach consensus using weighted voting.
        
        Args:
            messages: List of messages to consider
            agent_weights: Dictionary of agent weights
            
        Returns:
            Consensus result
        """
        if not messages or len(messages) < self.min_messages:
            return {
                "status": "insufficient_data",
                "consensus": None,
                "confidence": 0.0,
                "contributing_messages": len(messages),
                "timestamp": datetime.now().isoformat()
            }
            
        # Filter messages by confidence
        filtered_messages = [
            msg for msg in messages 
            if msg.content.get("confidence", 0) >= self.min_confidence
        ]
        
        if not filtered_messages:
            return {
                "status": "low_confidence",
                "consensus": None,
                "confidence": 0.0,
                "contributing_messages": 0,
                "timestamp": datetime.now().isoformat()
            }
            
        # Calculate recency weights
        now = datetime.now()
        max_age = timedelta(hours=24)
        
        recency_weights = []
        for msg in filtered_messages:
            msg_time = msg.timestamp
            age = now - msg_time
            age_factor = max(0, 1 - (age / max_age).total_seconds() / 86400)
            recency_weights.append(self.recency_weight * age_factor + (1 - self.recency_weight))
            
        # Calculate total weights
        total_weights = []
        for i, msg in enumerate(filtered_messages):
            agent_weight = agent_weights.get(msg.sender_id, 1.0)
            confidence = msg.content.get("confidence", 0.5)
            total_weight = agent_weight * confidence * recency_weights[i]
            total_weights.append(total_weight)
            
        # Normalize weights
        weight_sum = sum(total_weights)
        if weight_sum > 0:
            normalized_weights = [w / weight_sum for w in total_weights]
        else:
            normalized_weights = [1.0 / len(total_weights)] * len(total_weights)
            
        # Calculate weighted votes for different signal types
        signal_votes = {}
        
        for i, msg in enumerate(filtered_messages):
            signal_type = msg.content.get("signal_type", "neutral")
            if signal_type not in signal_votes:
                signal_votes[signal_type] = 0.0
                
            signal_votes[signal_type] += normalized_weights[i]
            
        # Find consensus signal type
        if signal_votes:
            consensus_type = max(signal_votes, key=signal_votes.get)
            consensus_confidence = signal_votes[consensus_type]
        else:
            consensus_type = "neutral"
            consensus_confidence = 0.0
            
        # Calculate consensus horizon
        horizon_votes = {}
        
        for i, msg in enumerate(filtered_messages):
            horizon = msg.content.get("horizon", "medium")
            if horizon not in horizon_votes:
                horizon_votes[horizon] = 0.0
                
            horizon_votes[horizon] += normalized_weights[i]
            
        consensus_horizon = max(horizon_votes, key=horizon_votes.get) if horizon_votes else "medium"
        
        # Prepare consensus result
        result = {
            "status": "success",
            "consensus": {
                "signal_type": consensus_type,
                "confidence": consensus_confidence,
                "horizon": consensus_horizon,
                "action": "buy" if consensus_type == "bullish" else 
                         ("sell" if consensus_type == "bearish" else "hold"),
                "signal_distribution": signal_votes
            },
            "contributing_messages": len(filtered_messages),
            "timestamp": datetime.now().isoformat()
        }
        
        return result


class BayesianConsensusStrategy(ConsensusStrategy):
    """
    Bayesian consensus strategy for reaching consensus.
    
    This strategy uses Bayesian updating to combine evidence from multiple
    agents, accounting for their historical accuracy and confidence.
    """
    
    def __init__(self, prior_distribution: Optional[Dict[str, float]] = None,
                min_messages: int = 2):
        """
        Initialize the Bayesian consensus strategy.
        
        Args:
            prior_distribution: Prior probability distribution for signal types
            min_messages: Minimum number of messages for valid consensus
        """
        # Default prior distribution (equal probabilities)
        self.prior_distribution = prior_distribution or {
            "bullish": 0.25,
            "bearish": 0.25,
            "neutral": 0.25,
            "volatile": 0.25
        }
        
        self.min_messages = min_messages
        
    def reach_consensus(self, messages: List[AgentMessage], 
                      agent_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Reach consensus using Bayesian updating.
        
        Args:
            messages: List of messages to consider
            agent_weights: Dictionary of agent weights
            
        Returns:
            Consensus result
        """
        if not messages or len(messages) < self.min_messages:
            return {
                "status": "insufficient_data",
                "consensus": None,
                "confidence": 0.0,
                "contributing_messages": len(messages),
                "timestamp": datetime.now().isoformat()
            }
            
        # Start with prior distribution
        posterior = self.prior_distribution.copy()
        
        # Update posterior with each message
        for msg in messages:
            self._update_posterior(posterior, msg, agent_weights.get(msg.sender_id, 1.0))
            
        # Normalize posterior
        total = sum(posterior.values())
        if total > 0:
            posterior = {k: v / total for k, v in posterior.items()}
            
        # Find consensus signal type
        consensus_type = max(posterior, key=posterior.get)
        consensus_confidence = posterior[consensus_type]
        
        # Determine consensus horizon
        horizon_weights = {"short": 0.0, "medium": 0.0, "long": 0.0}
        
        for msg in messages:
            horizon = msg.content.get("horizon", "medium")
            confidence = msg.content.get("confidence", 0.5)
            agent_weight = agent_weights.get(msg.sender_id, 1.0)
            
            if horizon in horizon_weights:
                horizon_weights[horizon] += confidence * agent_weight
                
        # Normalize horizon weights
        total_horizon_weight = sum(horizon_weights.values())
        if total_horizon_weight > 0:
            horizon_weights = {k: v / total_horizon_weight for k, v in horizon_weights.items()}
            
        consensus_horizon = max(horizon_weights, key=horizon_weights.get)
        
        # Prepare consensus result
        result = {
            "status": "success",
            "consensus": {
                "signal_type": consensus_type,
                "confidence": consensus_confidence,
                "horizon": consensus_horizon,
                "action": "buy" if consensus_type == "bullish" else 
                         ("sell" if consensus_type == "bearish" else "hold"),
                "probability_distribution": posterior
            },
            "contributing_messages": len(messages),
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    def _update_posterior(self, posterior: Dict[str, float], 
                        message: AgentMessage, agent_weight: float):
        """
        Update posterior distribution with a message.
        
        Args:
            posterior: Posterior distribution to update
            message: Message to incorporate
            agent_weight: Weight of the agent
        """
        signal_type = message.content.get("signal_type", "neutral")
        confidence = message.content.get("confidence", 0.5)
        
        # Adjust confidence based on agent weight
        adjusted_confidence = min(0.95, confidence * agent_weight)
        
        # Create likelihood for this message
        likelihood = {signal_type: adjusted_confidence}
        
        # Add small likelihood for other signal types
        remaining_prob = 1.0 - adjusted_confidence
        other_types = [t for t in posterior.keys() if t != signal_type]
        
        if other_types:
            prob_per_type = remaining_prob / len(other_types)
            for t in other_types:
                likelihood[t] = prob_per_type
                
        # Update posterior
        for signal_type, prior_prob in posterior.items():
            posterior[signal_type] = prior_prob * likelihood.get(signal_type, 0.0)


class ConsensusEngine:
    """
    Engine for reaching consensus among multiple agents.
    
    This engine collects messages from different agents and applies
    consensus strategies to reach a unified decision.
    """
    
    def __init__(self, strategy: Optional[ConsensusStrategy] = None):
        """
        Initialize the consensus engine.
        
        Args:
            strategy: Consensus strategy to use
        """
        self.strategy = strategy or WeightedVotingStrategy()
        self.message_buffer: List[AgentMessage] = []
        self.consensus_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        
    def add_message(self, message: AgentMessage):
        """
        Add a message to the consensus engine.
        
        Args:
            message: Message to add
        """
        self.message_buffer.append(message)
        
        # Limit buffer size
        if len(self.message_buffer) > 1000:
            self.message_buffer = self.message_buffer[-1000:]
            
    def reach_consensus(self, message_types: List[MessageType],
                      time_window: timedelta = timedelta(hours=24),
                      agent_weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Reach consensus from buffered messages.
        
        Args:
            message_types: Types of messages to consider
            time_window: Time window for messages
            agent_weights: Dictionary of agent weights
            
        Returns:
            Consensus result
        """
        # Default agent weights
        if agent_weights is None:
            agent_weights = {}
            
        # Filter messages by type and time
        now = datetime.now()
        cutoff_time = now - time_window
        
        filtered_messages = [
            msg for msg in self.message_buffer
            if msg.message_type in message_types and msg.timestamp >= cutoff_time
        ]
        
        # Reach consensus
        result = self.strategy.reach_consensus(filtered_messages, agent_weights)
        
        # Add to history
        self.consensus_history.append(result)
        if len(self.consensus_history) > self.max_history_size:
            self.consensus_history = self.consensus_history[-self.max_history_size:]
            
        return result
        
    def set_strategy(self, strategy: ConsensusStrategy):
        """
        Set the consensus strategy.
        
        Args:
            strategy: New consensus strategy
        """
        self.strategy = strategy
        
    def get_consensus_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get consensus history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of consensus results
        """
        return self.consensus_history[-limit:]
