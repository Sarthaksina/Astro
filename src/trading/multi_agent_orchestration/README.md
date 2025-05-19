# Multi-Agent Orchestration Network

## Overview

The Multi-Agent Orchestration Network is a sophisticated framework for coordinating multiple specialized agents to create a robust and adaptable market prediction system. This system leverages the strengths of different agent types, each focusing on specific aspects of market analysis, to generate more reliable and comprehensive predictions.

## Architecture

The Multi-Agent Orchestration Network consists of the following key components:

### Core Components

1. **Base Framework** (`base.py`)
   - `AgentInterface`: Abstract base class that all agents must implement
   - `AgentMessage`: Data structure for messages exchanged between agents
   - `MessageBroker`: Handles message routing between agents
   - `AgentRegistry`: Manages agent creation and configuration

2. **Orchestrator** (`orchestrator.py`)
   - Central coordination component that manages the lifecycle of agents
   - Handles data distribution to agents
   - Collects and processes agent outputs
   - Manages system state and monitoring

3. **Consensus Engine** (`consensus.py`)
   - Implements strategies for reaching consensus among multiple agents
   - `WeightedVotingStrategy`: Uses weighted voting based on agent weights and confidence
   - `BayesianConsensusStrategy`: Uses Bayesian updating to combine evidence from multiple agents

### Specialized Agents

1. **Market Regime Agent** (`specialized_agents.py`)
   - Detects and tracks market regimes (bull, bear, sideways, volatile)
   - Notifies other agents when regime changes occur

2. **Signal Generator Agent** (`signal_generator_agent.py`)
   - Generates trading signals based on astrological data and market conditions
   - Adjusts signal confidence based on current market regime

3. **Risk Management Agent** (`risk_management_agent.py`)
   - Monitors risk factors and generates risk alerts
   - Provides mitigation strategies for identified risks

4. **Macro Environment Agent** (`macro_environment_agent.py`)
   - Analyzes macroeconomic factors and their astrological correlations
   - Provides broader market context for predictions

## Usage

### Basic Setup

```python
from src.trading.multi_agent_orchestration import (
    Orchestrator, OrchestratorConfig,
    MarketRegimeAgent, SignalGeneratorAgent, RiskManagementAgent, MacroEnvironmentAgent,
    ConsensusEngine, WeightedVotingStrategy
)

# Create orchestrator with configuration
config = OrchestratorConfig(
    processing_interval=1.0,  # seconds
    enable_monitoring=True,
    consensus_threshold=0.7
)
orchestrator = Orchestrator(config)

# Register agent types
orchestrator.agent_registry.register_agent_type("market_regime", MarketRegimeAgent)
orchestrator.agent_registry.register_agent_type("signal_generator", SignalGeneratorAgent)
orchestrator.agent_registry.register_agent_type("risk_management", RiskManagementAgent)
orchestrator.agent_registry.register_agent_type("macro_environment", MacroEnvironmentAgent)

# Create and register agents
regime_agent = orchestrator.create_and_register_agent(
    "market_regime", "regime_agent",
    message_types=[MessageType.MARKET_DATA, MessageType.QUERY],
    initial_weight=1.0
)

signal_agent = orchestrator.create_and_register_agent(
    "signal_generator", "signal_agent",
    message_types=[MessageType.MARKET_DATA, MessageType.ASTROLOGICAL_DATA, 
                  MessageType.REGIME_CHANGE, MessageType.QUERY],
    initial_weight=1.2
)

risk_agent = orchestrator.create_and_register_agent(
    "risk_management", "risk_agent",
    message_types=[MessageType.MARKET_DATA, MessageType.REGIME_CHANGE, MessageType.QUERY],
    initial_weight=1.5
)

macro_agent = orchestrator.create_and_register_agent(
    "macro_environment", "macro_agent",
    message_types=[MessageType.MARKET_DATA, MessageType.ASTROLOGICAL_DATA, MessageType.QUERY],
    initial_weight=0.8
)

# Start orchestrator
orchestrator.start()
```

### Submitting Data

```python
# Submit market data
orchestrator.submit_data("market_data", market_data_df)

# Submit astrological data
orchestrator.submit_data("astrological_data", astro_data_df)

# Process data with all agents
orchestrator.process_data()
```

### Getting Predictions

```python
# Get consensus predictions
predictions = orchestrator.get_predictions()

# Access specific prediction
if predictions["status"] == "success":
    for target, prediction in predictions["predictions"].items():
        print(f"Prediction for {target}: {prediction['value']} "
              f"(Consensus level: {prediction['consensus_level']:.2f})")
```

### Monitoring and Management

```python
# Update agent performance based on accuracy
orchestrator.update_agent_performance("signal_agent", 0.85)

# Export system state for analysis
orchestrator.export_system_state("system_state.json")

# Stop orchestrator when done
orchestrator.stop()
```

## Integration with Trading Strategies

The Multi-Agent Orchestration Network is designed to integrate seamlessly with the existing trading strategy framework:

```python
from src.trading.strategy_framework import BaseStrategy
from src.trading.multi_agent_orchestration import Orchestrator

class MultiAgentStrategy(BaseStrategy):
    def __init__(self, name, orchestrator, **kwargs):
        super().__init__(name, **kwargs)
        self.orchestrator = orchestrator
        
    def generate_signals(self, market_data, planetary_data):
        # Submit data to orchestrator
        self.orchestrator.submit_data("market_data", market_data)
        self.orchestrator.submit_data("astrological_data", planetary_data)
        
        # Process data
        self.orchestrator.process_data()
        
        # Get predictions
        predictions = self.orchestrator.get_predictions()
        
        # Convert predictions to strategy signals
        signals = []
        if predictions["status"] == "success":
            for target, prediction in predictions["predictions"].items():
                if prediction["consensus_level"] >= 0.7:
                    signal_type = 1 if prediction["value"] > 0 else -1
                    signals.append({
                        "type": signal_type,
                        "strength": prediction["consensus_level"],
                        "source": "multi_agent_network"
                    })
                    
        return signals
```

## Testing

Unit tests for the Multi-Agent Orchestration Network are available in `tests/trading/test_multi_agent_orchestration.py`. Run the tests using pytest:

```bash
pytest tests/trading/test_multi_agent_orchestration.py
```
