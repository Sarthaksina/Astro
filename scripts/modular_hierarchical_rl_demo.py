#!/usr/bin/env python
# NOTE: This file was automatically updated to use the new modular hierarchical RL framework.
# Please review the changes and refer to docs/migration_guide_hierarchical_rl.md for more information.


# Cosmic Market Oracle - Modular Hierarchical RL Demo

"""
Demonstration script for the new modular hierarchical RL framework.

This script shows how to use the new modular hierarchical RL framework,
which replaces the redundant implementations in:
- hierarchical_rl.py
- hierarchical_mcts_rl.py
- mcts.py
- monte_carlo_search.py

The new implementation provides a more flexible and maintainable architecture
with clear separation between strategic planning and tactical execution.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
# import logging # Removed

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
# logging.basicConfig( # Removed
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__) # Removed
from src.utils.logger import get_logger # Added
logger = get_logger(__name__) # Added

# Import the modular hierarchical RL components
from src.trading.modular_hierarchical_rl import (
    ModularHierarchicalRLAgent,
    MCTSStrategicPlanner,
    PPOTacticalExecutor
)
from src.models.reinforcement import MarketEnvironment, ReplayBuffer

def run_demo():
    """Run a demonstration of the modular hierarchical RL framework."""
    logger.info("Starting modular hierarchical RL demonstration")
    
    # Create a simple market environment
    env = MarketEnvironment(
        data_path="data/market/sample_data.csv",
        window_size=10,
        max_steps=100
    )
    
    # Create the strategic planner using MCTS
    strategic_planner = MCTSStrategicPlanner(
        state_dim=env.observation_space.shape[0],
        num_simulations=50,
        max_depth=5,
        exploration_weight=1.0
    )
    
    # Create the tactical executor using PPO
    tactical_executor = PPOTacticalExecutor(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=64,
        learning_rate=3e-4
    )
    
    # Create the modular hierarchical RL agent
    agent = ModularModularHierarchicalRLAgent(
        strategic_planner=strategic_planner,
        tactical_executor=tactical_executor,
        replay_buffer_size=10000,
        strategic_interval=5
    )
    
    # Train the agent
    logger.info("Training the agent for a few episodes...")
    rewards = []
    for episode in range(5):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            # Train the agent periodically
            if env.current_step % 10 == 0:
                agent.train(batch_size=32)
        
        rewards.append(episode_reward)
        logger.info(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Modular Hierarchical RL Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('modular_hierarchical_rl_results.png')
    
    logger.info("Demonstration completed. Results saved to modular_hierarchical_rl_results.png")
    
    # Show how to access components individually
    logger.info("\nAccessing individual components:")
    logger.info(f"Strategic planner type: {type(agent.strategic_planner).__name__}")
    logger.info(f"Tactical executor type: {type(agent.tactical_executor).__name__}")
    
    # Show the advantage of the modular approach
    logger.info("\nAdvantage of modular approach:")
    logger.info("You can easily swap components without changing the overall architecture.")
    logger.info("For example, you could replace MCTS with another strategic planner or PPO with another RL algorithm.")

if __name__ == "__main__":
    run_demo()
