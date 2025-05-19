#!/usr/bin/env python
# Tests for Multi-Agent Orchestration Network

"""
Unit tests for the Multi-Agent Orchestration Network components.

This module tests the functionality of the multi-agent system, including
agent communication, message passing, and consensus building.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import logging
from unittest.mock import MagicMock, patch

# Ensure src directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.trading.multi_agent_orchestration.base import (
    AgentInterface, AgentMessage, MessageBroker, AgentRegistry, MessageType
)
from src.trading.multi_agent_orchestration.orchestrator import (
    Orchestrator, OrchestratorConfig
)
from src.trading.multi_agent_orchestration.specialized_agents import MarketRegimeAgent
from src.trading.multi_agent_orchestration.signal_generator_agent import SignalGeneratorAgent
from src.trading.multi_agent_orchestration.risk_management_agent import RiskManagementAgent
from src.trading.multi_agent_orchestration.macro_environment_agent import MacroEnvironmentAgent
from src.trading.multi_agent_orchestration.consensus import (
    ConsensusEngine, WeightedVotingStrategy, BayesianConsensusStrategy
)

# Configure logging
logging.basicConfig(level=logging.ERROR)


class TestAgentMessage(unittest.TestCase):
    """Tests for the AgentMessage class."""
    
    def test_message_creation(self):
        """Test creating a message."""
        message = AgentMessage(
            sender_id="test_agent",
            recipient_id="target_agent",
            message_type=MessageType.MARKET_DATA,
            content={"data": "test_data"}
        )
        
        self.assertEqual(message.sender_id, "test_agent")
        self.assertEqual(message.recipient_id, "target_agent")
        self.assertEqual(message.message_type, MessageType.MARKET_DATA)
        self.assertEqual(message.content, {"data": "test_data"})
        
    def test_message_serialization(self):
        """Test message serialization and deserialization."""
        original = AgentMessage(
            sender_id="test_agent",
            message_type=MessageType.PREDICTION,
            content={"value": 0.5, "confidence": 0.8}
        )
        
        # Convert to dict and back
        message_dict = original.to_dict()
        reconstructed = AgentMessage.from_dict(message_dict)
        
        self.assertEqual(reconstructed.sender_id, original.sender_id)
        self.assertEqual(reconstructed.message_type, original.message_type)
        self.assertEqual(reconstructed.content, original.content)


class TestAgentInterface(unittest.TestCase):
    """Tests for the AgentInterface class."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        # Create a concrete implementation of AgentInterface for testing
        class TestAgent(AgentInterface):
            def process(self, data):
                return []
                
        agent = TestAgent("test_id", "test_type")
        
        self.assertEqual(agent.agent_id, "test_id")
        self.assertEqual(agent.agent_type, "test_type")
        
    def test_message_handling(self):
        """Test message handling."""
        # Create a concrete implementation with custom handler
        class TestAgent(AgentInterface):
            def __init__(self, agent_id, agent_type):
                super().__init__(agent_id, agent_type)
                self.message_handlers[MessageType.QUERY] = self._handle_query
                
            def _handle_query(self, message):
                return self.create_message(
                    MessageType.RESPONSE,
                    {"response": "test_response"},
                    message.sender_id,
                    message.message_id
                )
                
            def process(self, data):
                return []
                
        agent = TestAgent("test_id", "test_type")
        
        # Create a query message
        query = AgentMessage(
            sender_id="sender",
            recipient_id="test_id",
            message_type=MessageType.QUERY,
            content={"query": "test_query"}
        )
        
        # Handle message
        response = agent.handle_message(query)
        
        self.assertIsNotNone(response)
        self.assertEqual(response.message_type, MessageType.RESPONSE)
        self.assertEqual(response.content, {"response": "test_response"})
        self.assertEqual(response.recipient_id, "sender")


class TestMessageBroker(unittest.TestCase):
    """Tests for the MessageBroker class."""
    
    def setUp(self):
        """Set up test environment."""
        self.broker = MessageBroker()
        
        # Create mock agents
        self.agent1 = MagicMock()
        self.agent1.agent_id = "agent1"
        self.agent1.handle_message.return_value = None
        
        self.agent2 = MagicMock()
        self.agent2.agent_id = "agent2"
        self.agent2.handle_message.return_value = AgentMessage(
            sender_id="agent2",
            recipient_id="agent1",
            message_type=MessageType.RESPONSE,
            content={"response": "test_response"}
        )
        
        # Register agents
        self.broker.register_agent(self.agent1)
        self.broker.register_agent(self.agent2)
        
    def test_message_routing(self):
        """Test message routing to specific agent."""
        # Create message for agent1
        message = AgentMessage(
            sender_id="test",
            recipient_id="agent1",
            message_type=MessageType.COMMAND,
            content={"command": "test_command"}
        )
        
        # Publish message
        self.broker.publish(message)
        
        # Verify agent1 received the message
        self.agent1.handle_message.assert_called_once()
        self.agent2.handle_message.assert_not_called()
        
    def test_message_broadcast(self):
        """Test message broadcasting."""
        # Subscribe agents to message type
        self.broker.subscribe("agent1", [MessageType.MARKET_DATA])
        self.broker.subscribe("agent2", [MessageType.MARKET_DATA])
        
        # Create broadcast message
        message = AgentMessage(
            sender_id="test",
            recipient_id=None,  # Broadcast
            message_type=MessageType.MARKET_DATA,
            content={"data": "test_data"}
        )
        
        # Publish message
        self.broker.publish(message)
        
        # Verify both agents received the message
        self.agent1.handle_message.assert_called_once()
        self.agent2.handle_message.assert_called_once()
        
    def test_message_response(self):
        """Test handling message responses."""
        # Subscribe agent1 to message type
        self.broker.subscribe("agent1", [MessageType.QUERY])
        
        # Create message that will generate a response
        message = AgentMessage(
            sender_id="test",
            recipient_id=None,  # Broadcast
            message_type=MessageType.QUERY,
            content={"query": "test_query"}
        )
        
        # Publish message
        responses = self.broker.publish(message)
        
        # Verify response was collected
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].sender_id, "agent2")
        self.assertEqual(responses[0].recipient_id, "agent1")


class TestMarketRegimeAgent(unittest.TestCase):
    """Tests for the MarketRegimeAgent class."""
    
    def setUp(self):
        """Set up test environment."""
        self.agent = MarketRegimeAgent("regime_agent")
        
    def test_regime_detection(self):
        """Test market regime detection."""
        # Create mock market data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        prices = np.linspace(100, 150, 100)  # Uptrend
        
        # Add some volatility
        prices = prices + np.random.normal(0, 2, 100)
        
        market_data = pd.DataFrame({
            "date": dates,
            "close": prices,
            "volume": np.random.randint(1000, 5000, 100)
        })
        
        # Process data
        messages = self.agent.process({"market_data": market_data})
        
        # Verify agent detected a regime
        self.assertIsNotNone(self.agent.current_regime)
        self.assertNotEqual(self.agent.current_regime, "unknown")
        
    def test_query_handling(self):
        """Test handling of query messages."""
        # Create query message
        query = AgentMessage(
            sender_id="test",
            recipient_id="regime_agent",
            message_type=MessageType.QUERY,
            content={"query_type": "current_regime"}
        )
        
        # Handle query
        response = self.agent.handle_message(query)
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertEqual(response.message_type, MessageType.RESPONSE)
        self.assertEqual(response.content["query_type"], "current_regime")
        self.assertIn("current_regime", response.content)


class TestConsensusEngine(unittest.TestCase):
    """Tests for the ConsensusEngine class."""
    
    def setUp(self):
        """Set up test environment."""
        self.engine = ConsensusEngine()
        
        # Create test messages
        self.messages = [
            AgentMessage(
                sender_id=f"agent{i}",
                message_type=MessageType.PREDICTION,
                content={
                    "signal_type": "bullish" if i % 3 == 0 else 
                                  ("bearish" if i % 3 == 1 else "neutral"),
                    "confidence": 0.7 + (i % 3) * 0.1,
                    "horizon": "medium"
                },
                timestamp=datetime.now() - timedelta(hours=i)
            )
            for i in range(10)
        ]
        
        # Add messages to engine
        for msg in self.messages:
            self.engine.add_message(msg)
            
    def test_weighted_voting_consensus(self):
        """Test weighted voting consensus strategy."""
        # Set strategy
        self.engine.set_strategy(WeightedVotingStrategy())
        
        # Define agent weights
        agent_weights = {f"agent{i}": 1.0 + i * 0.1 for i in range(10)}
        
        # Reach consensus
        result = self.engine.reach_consensus(
            message_types=[MessageType.PREDICTION],
            time_window=timedelta(days=1),
            agent_weights=agent_weights
        )
        
        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertIn("consensus", result)
        self.assertIn("signal_type", result["consensus"])
        self.assertIn("confidence", result["consensus"])
        self.assertIn("horizon", result["consensus"])
        
    def test_bayesian_consensus(self):
        """Test Bayesian consensus strategy."""
        # Set strategy
        self.engine.set_strategy(BayesianConsensusStrategy())
        
        # Reach consensus
        result = self.engine.reach_consensus(
            message_types=[MessageType.PREDICTION],
            time_window=timedelta(days=1)
        )
        
        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertIn("consensus", result)
        self.assertIn("signal_type", result["consensus"])
        self.assertIn("confidence", result["consensus"])
        self.assertIn("probability_distribution", result["consensus"])


class TestOrchestrator(unittest.TestCase):
    """Tests for the Orchestrator class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create orchestrator with test configuration
        config = OrchestratorConfig(
            processing_interval=0.1,
            max_queue_size=100,
            enable_monitoring=False
        )
        self.orchestrator = Orchestrator(config)
        
        # Register agent types
        self.orchestrator.agent_registry.register_agent_type(
            "market_regime", MarketRegimeAgent
        )
        
        # Create and register agents
        self.regime_agent = self.orchestrator.create_and_register_agent(
            "market_regime", "regime_agent",
            message_types=[MessageType.MARKET_DATA, MessageType.QUERY],
            initial_weight=1.0
        )
        
    def test_agent_registration(self):
        """Test agent registration."""
        self.assertIn("regime_agent", self.orchestrator.agents)
        self.assertEqual(self.orchestrator.agent_weights["regime_agent"], 1.0)
        
    def test_data_submission(self):
        """Test data submission."""
        # Create mock market data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        prices = np.linspace(100, 150, 100)
        
        market_data = pd.DataFrame({
            "date": dates,
            "close": prices,
            "volume": np.random.randint(1000, 5000, 100)
        })
        
        # Submit data
        self.orchestrator.submit_data("market_data", market_data)
        
        # Verify data was cached
        self.assertIn("market_data", self.orchestrator.data_cache)
        self.assertTrue(
            self.orchestrator.data_cache["market_data"].equals(market_data)
        )
        
    def test_process_data(self):
        """Test data processing."""
        # Create mock market data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        prices = np.linspace(100, 150, 100)
        
        market_data = pd.DataFrame({
            "date": dates,
            "close": prices,
            "volume": np.random.randint(1000, 5000, 100)
        })
        
        # Submit data
        self.orchestrator.submit_data("market_data", market_data)
        
        # Process data
        with patch.object(self.regime_agent, 'process') as mock_process:
            mock_process.return_value = []
            self.orchestrator.process_data()
            
            # Verify agent process was called
            mock_process.assert_called_once()
            
    def test_start_stop(self):
        """Test orchestrator start and stop."""
        # Start orchestrator
        self.orchestrator.start()
        self.assertTrue(self.orchestrator.running)
        
        # Stop orchestrator
        self.orchestrator.stop()
        self.assertFalse(self.orchestrator.running)


if __name__ == '__main__':
    unittest.main()
