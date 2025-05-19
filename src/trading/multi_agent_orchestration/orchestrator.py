#!/usr/bin/env python
# Cosmic Market Oracle - Multi-Agent Orchestration Network Orchestrator

"""
Orchestrator for the Multi-Agent Orchestration Network.

This module provides the central orchestration component that coordinates
the activities of specialized agents in the multi-agent system.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import threading
import queue

from src.trading.multi_agent_orchestration.base import (
    AgentInterface, AgentMessage, MessageBroker, AgentRegistry, MessageType
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    processing_interval: float = 1.0  # seconds
    max_queue_size: int = 1000
    agent_timeout: float = 5.0  # seconds
    enable_monitoring: bool = True
    monitoring_interval: float = 60.0  # seconds
    log_level: str = "INFO"
    consensus_threshold: float = 0.7  # Minimum consensus level to accept a prediction
    performance_tracking: bool = True
    adaptive_weighting: bool = True  # Adjust agent weights based on performance


class Orchestrator:
    """
    Orchestrator for coordinating multiple agents in the system.
    
    The orchestrator manages the lifecycle of agents, coordinates message passing,
    and ensures the system operates coherently to produce reliable predictions.
    """
    
    def __init__(self, config: OrchestratorConfig = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()
        
        # Set up logging
        logging_level = getattr(logging, self.config.log_level)
        logger.setLevel(logging_level)
        
        # Core components
        self.message_broker = MessageBroker()
        self.agent_registry = AgentRegistry()
        
        # Agent management
        self.agents: Dict[str, AgentInterface] = {}
        self.agent_weights: Dict[str, float] = {}  # For weighted consensus
        self.agent_performance: Dict[str, List[float]] = {}  # Historical performance
        
        # Processing state
        self.running = False
        self.message_queue = queue.PriorityQueue(maxsize=self.config.max_queue_size)
        self.processing_thread = None
        
        # Monitoring
        self.monitoring_thread = None
        self.last_activity_time: Dict[str, datetime] = {}
        self.system_metrics: Dict[str, List[float]] = {
            "message_rate": [],
            "response_time": [],
            "consensus_level": [],
            "prediction_accuracy": []
        }
        
        # Data cache
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.prediction_history: List[Dict[str, Any]] = []
        
    def register_agent(self, agent: AgentInterface, message_types: List[MessageType] = None,
                      initial_weight: float = 1.0):
        """
        Register an agent with the orchestrator.
        
        Args:
            agent: The agent to register
            message_types: Message types the agent should subscribe to
            initial_weight: Initial weight for the agent in consensus
        """
        # Register with broker
        self.message_broker.register_agent(agent)
        
        # Subscribe to message types
        if message_types:
            self.message_broker.subscribe(agent.agent_id, message_types)
            
        # Add to local registry
        self.agents[agent.agent_id] = agent
        self.agent_weights[agent.agent_id] = initial_weight
        self.agent_performance[agent.agent_id] = []
        self.last_activity_time[agent.agent_id] = datetime.now()
        
        logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type})")
        
    def create_and_register_agent(self, agent_type: str, agent_id: str = None, 
                                message_types: List[MessageType] = None,
                                initial_weight: float = 1.0, **kwargs) -> AgentInterface:
        """
        Create and register an agent.
        
        Args:
            agent_type: Type of agent to create
            agent_id: ID for the new agent (auto-generated if None)
            message_types: Message types the agent should subscribe to
            initial_weight: Initial weight for the agent in consensus
            **kwargs: Additional arguments for agent initialization
            
        Returns:
            The created agent
        """
        # Generate ID if not provided
        if agent_id is None:
            agent_id = f"{agent_type}_{len(self.agents) + 1}"
            
        # Create agent
        agent = self.agent_registry.create_agent(agent_type, agent_id, **kwargs)
        
        # Register agent
        self.register_agent(agent, message_types, initial_weight)
        
        return agent
        
    def start(self):
        """Start the orchestrator."""
        if self.running:
            logger.warning("Orchestrator is already running")
            return
            
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        # Start monitoring thread if enabled
        if self.config.enable_monitoring:
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            
        logger.info("Orchestrator started")
        
    def stop(self):
        """Stop the orchestrator."""
        if not self.running:
            logger.warning("Orchestrator is not running")
            return
            
        self.running = False
        
        # Wait for threads to terminate
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        logger.info("Orchestrator stopped")
        
    def submit_data(self, data_type: str, data: pd.DataFrame):
        """
        Submit data to the system.
        
        Args:
            data_type: Type of data (e.g., 'market_data', 'astrological_data')
            data: Data frame containing the data
        """
        # Update data cache
        self.data_cache[data_type] = data
        
        # Create message
        message_type = MessageType.MARKET_DATA if data_type == 'market_data' else MessageType.ASTROLOGICAL_DATA
        
        message = AgentMessage(
            sender_id="orchestrator",
            message_type=message_type,
            content={"data_type": data_type, "timestamp": datetime.now().isoformat()}
        )
        
        # Queue message with high priority
        self.message_queue.put((1, message))
        
        logger.info(f"Submitted {data_type} data: {len(data)} rows")
        
    def process_data(self):
        """
        Process all available data with registered agents.
        
        This method triggers all agents to process the current data cache
        and generate messages.
        """
        if not self.data_cache:
            logger.warning("No data available for processing")
            return
            
        # Process data with each agent
        for agent_id, agent in self.agents.items():
            try:
                messages = agent.process(self.data_cache)
                
                # Queue messages
                for message in messages:
                    self.message_queue.put((message.priority, message))
                    
                # Update activity time
                self.last_activity_time[agent_id] = datetime.now()
                
            except Exception as e:
                logger.error(f"Error processing data with agent {agent_id}: {e}")
                
        logger.info(f"Processed data with {len(self.agents)} agents")
        
    def get_predictions(self) -> Dict[str, Any]:
        """
        Get the current consensus predictions.
        
        Returns:
            Dictionary containing consensus predictions
        """
        # Filter prediction messages from history
        prediction_messages = [
            msg for msg in self.message_broker.message_history
            if msg.message_type == MessageType.PREDICTION
        ]
        
        if not prediction_messages:
            return {"status": "no_predictions", "predictions": {}}
            
        # Group by prediction target
        predictions_by_target = {}
        
        for msg in prediction_messages:
            target = msg.content.get("target", "unknown")
            if target not in predictions_by_target:
                predictions_by_target[target] = []
                
            predictions_by_target[target].append({
                "agent_id": msg.sender_id,
                "value": msg.content.get("value"),
                "confidence": msg.content.get("confidence", 0.5),
                "timestamp": msg.timestamp,
                "weight": self.agent_weights.get(msg.sender_id, 1.0)
            })
            
        # Calculate consensus for each target
        consensus = {}
        
        for target, preds in predictions_by_target.items():
            # Sort by timestamp (newest first)
            preds.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Get recent predictions (last hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_preds = [p for p in preds if p["timestamp"] > cutoff_time]
            
            if not recent_preds:
                recent_preds = preds[:5]  # Take at most 5 if no recent ones
                
            # Calculate weighted average
            total_weight = sum(p["weight"] * p["confidence"] for p in recent_preds)
            
            if total_weight > 0:
                weighted_sum = sum(p["value"] * p["weight"] * p["confidence"] 
                                for p in recent_preds)
                consensus_value = weighted_sum / total_weight
                
                # Calculate consensus level (agreement among agents)
                values = [p["value"] for p in recent_preds]
                mean_value = sum(values) / len(values)
                variance = sum((v - mean_value) ** 2 for v in values) / len(values)
                consensus_level = 1.0 / (1.0 + variance)  # Normalize to 0-1
                
                consensus[target] = {
                    "value": consensus_value,
                    "consensus_level": consensus_level,
                    "contributing_agents": len(recent_preds),
                    "timestamp": datetime.now()
                }
                
        result = {
            "status": "success",
            "predictions": consensus,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in prediction history
        self.prediction_history.append(result)
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
            
        return result
        
    def update_agent_performance(self, agent_id: str, performance_score: float):
        """
        Update an agent's performance score.
        
        Args:
            agent_id: ID of the agent
            performance_score: Performance score (0-1)
        """
        if agent_id not in self.agent_performance:
            logger.warning(f"Unknown agent: {agent_id}")
            return
            
        self.agent_performance[agent_id].append(performance_score)
        
        # Keep only recent performance (last 100 scores)
        if len(self.agent_performance[agent_id]) > 100:
            self.agent_performance[agent_id] = self.agent_performance[agent_id][-100:]
            
        # Update weight if adaptive weighting is enabled
        if self.config.adaptive_weighting:
            avg_performance = sum(self.agent_performance[agent_id]) / len(self.agent_performance[agent_id])
            self.agent_weights[agent_id] = max(0.1, min(5.0, avg_performance * 5.0))
            
        logger.debug(f"Updated performance for agent {agent_id}: {performance_score:.4f}")
        
    def export_system_state(self, file_path: str):
        """
        Export the current system state to a file.
        
        Args:
            file_path: Path to the output file
        """
        state = {
            "timestamp": datetime.now().isoformat(),
            "agents": {
                agent_id: {
                    "type": agent.agent_type,
                    "weight": self.agent_weights.get(agent_id, 1.0),
                    "avg_performance": (
                        sum(self.agent_performance.get(agent_id, [])) / 
                        len(self.agent_performance.get(agent_id, [1.0]))
                    ),
                    "last_activity": self.last_activity_time.get(
                        agent_id, datetime.now()
                    ).isoformat()
                }
                for agent_id, agent in self.agents.items()
            },
            "metrics": {
                metric: values[-10:] if values else []
                for metric, values in self.system_metrics.items()
            },
            "predictions": self.prediction_history[-10:] if self.prediction_history else []
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Exported system state to {file_path}")
        
    def _processing_loop(self):
        """Main processing loop for handling messages."""
        while self.running:
            try:
                # Get next message from queue (with timeout)
                try:
                    _, message = self.message_queue.get(timeout=self.config.processing_interval)
                except queue.Empty:
                    continue
                    
                # Process message
                start_time = time.time()
                responses = self.message_broker.publish(message)
                
                # Handle responses
                for response in responses:
                    self.message_queue.put((response.priority, response))
                    
                # Track response time
                response_time = time.time() - start_time
                self.system_metrics["response_time"].append(response_time)
                
                # Limit metrics history
                if len(self.system_metrics["response_time"]) > 1000:
                    self.system_metrics["response_time"] = self.system_metrics["response_time"][-1000:]
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                
    def _monitoring_loop(self):
        """Monitoring loop for system health and performance."""
        last_message_count = 0
        
        while self.running:
            try:
                # Sleep for monitoring interval
                time.sleep(self.config.monitoring_interval)
                
                # Calculate message rate
                current_count = len(self.message_broker.message_history)
                message_rate = (current_count - last_message_count) / self.config.monitoring_interval
                last_message_count = current_count
                
                self.system_metrics["message_rate"].append(message_rate)
                
                # Limit metrics history
                if len(self.system_metrics["message_rate"]) > 100:
                    self.system_metrics["message_rate"] = self.system_metrics["message_rate"][-100:]
                    
                # Check for inactive agents
                now = datetime.now()
                for agent_id, last_time in self.last_activity_time.items():
                    if (now - last_time).total_seconds() > 300:  # 5 minutes
                        logger.warning(f"Agent {agent_id} has been inactive for "
                                    f"{(now - last_time).total_seconds():.1f} seconds")
                        
                # Log system status
                logger.info(f"System status: {len(self.agents)} agents, "
                          f"{message_rate:.1f} msg/s, "
                          f"{self.message_queue.qsize()} queued messages")
                          
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
    def __del__(self):
        """Clean up resources when the orchestrator is deleted."""
        if self.running:
            self.stop()
