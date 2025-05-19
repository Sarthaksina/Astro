#!/usr/bin/env python
# Cosmic Market Oracle - Multi-Agent Orchestration Network

"""
Multi-Agent Orchestration Network for the Cosmic Market Oracle.

This package provides a framework for coordinating multiple specialized agents
to create a robust and adaptable market prediction system.
"""

from src.trading.multi_agent_orchestration.base import (
    AgentInterface,
    AgentMessage,
    AgentRegistry,
    MessageBroker
)

from src.trading.multi_agent_orchestration.orchestrator import (
    Orchestrator,
    OrchestratorConfig
)

from src.trading.multi_agent_orchestration.specialized_agents import (
    MarketRegimeAgent,
    SignalGeneratorAgent,
    RiskManagementAgent,
    MacroEnvironmentAgent
)

from src.trading.multi_agent_orchestration.consensus import (
    ConsensusEngine,
    WeightedVotingStrategy,
    BayesianConsensusStrategy
)

__all__ = [
    'AgentInterface',
    'AgentMessage',
    'AgentRegistry',
    'MessageBroker',
    'Orchestrator',
    'OrchestratorConfig',
    'MarketRegimeAgent',
    'SignalGeneratorAgent',
    'RiskManagementAgent',
    'MacroEnvironmentAgent',
    'ConsensusEngine',
    'WeightedVotingStrategy',
    'BayesianConsensusStrategy'
]
