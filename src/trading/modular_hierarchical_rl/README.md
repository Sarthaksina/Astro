# Modular Hierarchical Reinforcement Learning Framework

This directory contains a modular framework for hierarchical reinforcement learning in the Cosmic Market Oracle project. The framework consolidates redundant implementations and provides a clean, extensible architecture for strategic planning and tactical execution in trading systems.

## Architecture Overview

The modular hierarchical RL framework is built around a clear separation of concerns:

1. **Strategic Planning**: High-level decision making that determines long-term goals and strategies
2. **Tactical Execution**: Low-level action selection that implements the strategic decisions

This separation allows for flexible combinations of different planning and execution algorithms while maintaining a consistent interface.

## Components

### Base Interfaces and Agent

- `base.py`: Contains the core interfaces (`StrategicPlannerInterface`, `TacticalExecutorInterface`) and the main agent class (`ModularHierarchicalRLAgent`).

### Strategic Planners

- `mcts_planner.py`: MCTS-based strategic planner that uses the unified MCTS implementation for long-term planning.

### Tactical Executors

- `ppo_executor.py`: PPO-based tactical executor for low-level action execution optimized for financial markets.

### Trading Strategies

- `strategy.py`: Trading strategy implementations that leverage the modular hierarchical RL framework.

## API Documentation

### ModularHierarchicalRLAgent

The main agent class that combines a strategic planner and tactical executor.

```python
ModularHierarchicalRLAgent(
    strategic_planner: StrategicPlannerInterface,
    tactical_executor: TacticalExecutorInterface,
    replay_buffer_size: int = 10000,
    strategic_interval: int = 5
)
```

**Parameters:**
- `strategic_planner`: An instance implementing the StrategicPlannerInterface
- `tactical_executor`: An instance implementing the TacticalExecutorInterface
- `replay_buffer_size`: Size of the experience replay buffer
- `strategic_interval`: How often to make strategic decisions (in steps)

**Methods:**
- `select_action(state)`: Select an action for the given state
- `store_experience(state, action, reward, next_state, done)`: Store a transition in the replay buffer
- `train(batch_size=64)`: Train both the strategic planner and tactical executor

### MCTSStrategicPlanner

A strategic planner that uses Monte Carlo Tree Search for long-term planning.

```python
MCTSStrategicPlanner(
    state_dim: int,
    num_simulations: int = 50,
    max_depth: int = 5,
    exploration_weight: float = 1.0
)
```

**Parameters:**
- `state_dim`: Dimension of the state space
- `num_simulations`: Number of MCTS simulations to run
- `max_depth`: Maximum depth of the search tree
- `exploration_weight`: Weight for exploration in UCB formula

### PPOTacticalExecutor

A tactical executor that uses Proximal Policy Optimization for action selection.

```python
PPOTacticalExecutor(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 64,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    clip_ratio: float = 0.2
)
```

**Parameters:**
- `state_dim`: Dimension of the state space
- `action_dim`: Dimension of the action space
- `hidden_dim`: Dimension of the hidden layers
- `learning_rate`: Learning rate for the optimizer
- `gamma`: Discount factor for future rewards
- `clip_ratio`: PPO clipping parameter

## Usage Examples

### Basic Usage

```python
from src.trading.modular_hierarchical_rl import (
    ModularHierarchicalRLAgent,
    MCTSStrategicPlanner,
    PPOTacticalExecutor
)

# Create components
strategic_planner = MCTSStrategicPlanner(
    state_dim=env.observation_space.shape[0],
    num_simulations=50
)

tactical_executor = PPOTacticalExecutor(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dim=64
)

# Create agent
agent = ModularHierarchicalRLAgent(
    strategic_planner=strategic_planner,
    tactical_executor=tactical_executor
)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        state = next_state
        
        # Train periodically
        if env.current_step % 10 == 0:
            agent.train(batch_size=32)
```

### Creating Custom Components

You can create custom strategic planners or tactical executors by implementing the respective interfaces:

```python
from src.trading.modular_hierarchical_rl.base import StrategicPlannerInterface

class CustomStrategicPlanner(StrategicPlannerInterface):
    def __init__(self, state_dim):
        super().__init__(state_dim)
        # Custom initialization
    
    def plan_strategy(self, state):
        # Custom strategy planning logic
        return strategy
    
    def train(self, experiences):
        # Custom training logic
        return loss
```

## Redundancy Resolution

This framework resolves the following redundancies in the codebase:

1. **MCTS Implementations**: Consolidates the redundant implementations in `mcts.py` and `monte_carlo_search.py` into a single unified implementation in `unified_mcts.py`.

2. **Hierarchical RL Implementations**: Consolidates the overlapping code in `hierarchical_rl.py` and `hierarchical_mcts_rl.py` into a modular framework that can be used with or without MCTS.

3. **Policy Implementations**: Standardizes the policy implementations across different components.

## Performance Considerations

- The MCTS strategic planner can be computationally intensive. Consider adjusting `num_simulations` based on your hardware capabilities.
- For faster training, you can use smaller batch sizes initially and increase them as training progresses.
- The `strategic_interval` parameter controls how often strategic decisions are made. Larger values reduce computational overhead but may affect performance.

## See Also

- `scripts/modular_hierarchical_rl_demo.py`: Complete demonstration of the framework
- `docs/migration_guide_hierarchical_rl.md`: Guide for migrating from the deprecated implementations
- `tests/trading/test_modular_hierarchical_rl.py`: Unit tests showing usage patterns

## Migration Guide

To migrate from the old implementations to the new modular framework:

1. Replace imports from `hierarchical_rl` or `hierarchical_mcts_rl` with imports from `modular_hierarchical_rl`.
2. Use the appropriate strategic planner and tactical executor for your use case.
3. Create a `ModularHierarchicalRLAgent` with the chosen components.

Example:

```python
from src.trading.modular_hierarchical_rl import (
    ModularHierarchicalRLAgent,
    MCTSStrategicPlanner,
    PPOTacticalExecutor
)

# Create strategic planner
strategic_planner = MCTSStrategicPlanner(
    state_dim=state_dim,
    goal_dim=goal_dim
)

# Create tactical executor
tactical_executor = PPOTacticalExecutor(
    state_dim=state_dim,
    goal_dim=goal_dim,
    action_dim=action_dim
)

# Create agent
agent = ModularHierarchicalRLAgent(
    strategic_planner=strategic_planner,
    tactical_executor=tactical_executor
)
```
