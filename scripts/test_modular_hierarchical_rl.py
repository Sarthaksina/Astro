#!/usr/bin/env python
# NOTE: This file was automatically updated to use the new modular hierarchical RL framework.
# Please review the changes and refer to docs/migration_guide_hierarchical_rl.md for more information.


# Cosmic Market Oracle - Test Modular Hierarchical RL

"""
Script to test the modular hierarchical reinforcement learning system.

This script runs a simple test of the modular hierarchical RL framework
to ensure that all components work correctly together.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import logging

# Ensure Python 3.10 compatibility
if sys.version_info.major != 3 or sys.version_info.minor != 10:
    print("Warning: This script is optimized for Python 3.10")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.modular_hierarchical_rl import (
    ModularHierarchicalRLAgent,
    MCTSStrategicPlanner,
    PPOTacticalExecutor
)
from src.models.reinforcement import MarketEnvironment
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("test_modular_hierarchical_rl")


def test_agent_creation():
    """Test agent creation."""
    logger.info("Testing agent creation...")
    
    # Create environment
    env = MarketEnvironment()
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = 8  # Dimension of strategic goals
    
    # Create strategic planner
    strategic_planner = MCTSStrategicPlanner(
        state_dim=state_dim,
        goal_dim=goal_dim,
        hidden_dim=64,
        num_simulations=10,
        device="cpu"
    )
    
    # Create tactical executor
    tactical_executor = PPOTacticalExecutor(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        hidden_dim=64,
        device="cpu"
    )
    
    # Create agent
    agent = ModularModularHierarchicalRLAgent(
        strategic_planner=strategic_planner,
        tactical_executor=tactical_executor,
        strategic_horizon=5,
        device="cpu"
    )
    
    logger.info("Agent created successfully.")
    return agent, env


def test_action_selection(agent, env):
    """Test action selection."""
    logger.info("Testing action selection...")
    
    # Reset environment
    state, _ = env.reset()
    
    # Select action
    action = agent.select_action(state)
    
    logger.info(f"Action shape: {action.shape}")
    logger.info(f"Action: {action}")
    
    # Check action shape
    assert action.shape == env.action_space.shape, f"Expected action shape {env.action_space.shape}, got {action.shape}"
    
    logger.info("Action selection test passed.")


def test_experience_storage(agent, env):
    """Test experience storage."""
    logger.info("Testing experience storage...")
    
    # Reset environment
    state, _ = env.reset()
    
    # Select action
    action = agent.select_action(state)
    
    # Take step in environment
    next_state, reward, done, _, _ = env.step(action)
    
    # Store experience
    agent.store_experience(state, action, reward, next_state, done)
    
    # Check buffer sizes
    assert len(agent.tactical_buffer) > 0, "Tactical buffer should not be empty"
    
    logger.info("Experience storage test passed.")


def test_agent_update(agent):
    """Test agent update."""
    logger.info("Testing agent update...")
    
    # Fill buffers with random data
    for _ in range(300):
        state = np.random.randn(agent.strategic_planner.state_dim)
        action = np.random.randn(agent.tactical_executor.action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(agent.strategic_planner.state_dim)
        done = False
        
        # Store experience
        agent.store_experience(state, action, reward, next_state, done)
    
    # Update agent
    metrics = agent.update(batch_size=64)
    
    logger.info(f"Update metrics: {metrics}")
    
    # Check metrics
    assert metrics, "Update metrics should not be empty"
    
    logger.info("Agent update test passed.")


def test_agent_save_load(agent, tmp_dir="tmp"):
    """Test agent save and load."""
    logger.info("Testing agent save and load...")
    
    # Create temporary directory
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Save agent
    save_path = os.path.join(tmp_dir, "test_agent")
    agent.save(save_path)
    
    # Check that files were created
    assert os.path.exists(save_path + "_strategic.pt"), "Strategic model file should exist"
    assert os.path.exists(save_path + "_tactical.pt"), "Tactical model file should exist"
    
    # Create new agent
    new_agent = ModularModularHierarchicalRLAgent(
        strategic_planner=MCTSStrategicPlanner(
            state_dim=agent.strategic_planner.state_dim,
            goal_dim=agent.strategic_planner.goal_dim,
            hidden_dim=64,
            device="cpu"
        ),
        tactical_executor=PPOTacticalExecutor(
            state_dim=agent.strategic_planner.state_dim,
            goal_dim=agent.strategic_planner.goal_dim,
            action_dim=agent.tactical_executor.action_dim,
            hidden_dim=64,
            device="cpu"
        ),
        strategic_horizon=agent.strategic_horizon,
        device="cpu"
    )
    
    # Load agent
    new_agent.load(save_path)
    
    logger.info("Agent save and load test passed.")


def test_full_episode(agent, env):
    """Test a full episode."""
    logger.info("Testing full episode...")
    
    # Reset environment
    state, _ = env.reset()
    episode_return = 0
    
    # Run episode
    for step in range(100):
        # Select action
        action = agent.select_action(state)
        
        # Take step in environment
        next_state, reward, done, _, _ = env.step(action)
        
        # Store experience
        agent.store_experience(state, action, reward, next_state, done)
        
        # Update state and return
        state = next_state
        episode_return += reward
        
        # Update agent
        if step % 10 == 0:
            agent.update(batch_size=64)
        
        # Check if done
        if done:
            break
    
    logger.info(f"Episode return: {episode_return}")
    logger.info(f"Episode steps: {step + 1}")
    
    logger.info("Full episode test passed.")


def main():
    """Run all tests."""
    logger.info("Starting modular hierarchical RL tests...")
    
    # Test agent creation
    agent, env = test_agent_creation()
    
    # Test action selection
    test_action_selection(agent, env)
    
    # Test experience storage
    test_experience_storage(agent, env)
    
    # Test agent update
    test_agent_update(agent)
    
    # Test agent save and load
    test_agent_save_load(agent)
    
    # Test full episode
    test_full_episode(agent, env)
    
    logger.info("All tests passed!")


if __name__ == "__main__":
    main()
