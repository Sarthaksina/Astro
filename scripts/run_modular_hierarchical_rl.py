#!/usr/bin/env python
# NOTE: This file was automatically updated to use the new modular hierarchical RL framework.
# Please review the changes and refer to docs/migration_guide_hierarchical_rl.md for more information.


# Cosmic Market Oracle - Run Modular Hierarchical RL

"""
Script to run the modular hierarchical reinforcement learning system.

This script demonstrates how to use the modular hierarchical RL framework
with different strategic planners and tactical executors.
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
from src.utils.logger import get_logger # Changed
from src.utils.visualization import plot_training_metrics, plot_episode_returns

# Configure logging
logger = get_logger("run_modular_hierarchical_rl") # Changed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Modular Hierarchical RL")
    
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--strategic-horizon", type=int, default=20,
                        help="Horizon for strategic decisions")
    parser.add_argument("--mcts-simulations", type=int, default=50,
                        help="Number of MCTS simulations")
    parser.add_argument("--exploration-weight", type=float, default=1.0,
                        help="Exploration weight for MCTS")
    parser.add_argument("--learning-rate", type=float, default=0.0003,
                        help="Learning rate for optimization")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for updates")
    parser.add_argument("--update-interval", type=int, default=10,
                        help="Interval between updates")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Interval between evaluations")
    parser.add_argument("--save-interval", type=int, default=20,
                        help="Interval between model saves")
    parser.add_argument("--output-dir", type=str, default="../outputs/modular_hierarchical_rl",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    
    return parser.parse_args()


def main():
    """Run the modular hierarchical RL system."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
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
        hidden_dim=256,
        num_simulations=args.mcts_simulations,
        exploration_weight=args.exploration_weight,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Create tactical executor
    tactical_executor = PPOTacticalExecutor(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        hidden_dim=256,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        device=args.device
    )
    
    # Create agent
    agent = ModularModularHierarchicalRLAgent(
        strategic_planner=strategic_planner,
        tactical_executor=tactical_executor,
        strategic_horizon=args.strategic_horizon,
        device=args.device
    )
    
    # Training loop
    episode_returns = []
    update_metrics = []
    
    for episode in range(args.episodes):
        # Reset environment
        state, _ = env.reset()
        episode_return = 0
        
        for step in range(args.max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, _, info = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Update state and return
            state = next_state
            episode_return += reward
            
            # Update agent
            if step % args.update_interval == 0:
                metrics = agent.update(batch_size=args.batch_size)
                update_metrics.append(metrics)
                
                # Log metrics
                if metrics:
                    log_str = f"Episode {episode}, Step {step}: "
                    log_str += ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                    logger.info(log_str)
            
            # Check if done
            if done:
                break
        
        # Store episode return
        episode_returns.append(episode_return)
        
        # Log episode
        logger.info(f"Episode {episode}: Return = {episode_return:.4f}, Steps = {step + 1}")
        
        # Evaluate agent
        if episode % args.eval_interval == 0:
            eval_return = evaluate_agent(agent, env, num_episodes=5, max_steps=args.max_steps)
            logger.info(f"Evaluation: Episode {episode}, Return = {eval_return:.4f}")
        
        # Save agent
        if episode % args.save_interval == 0:
            agent.save(os.path.join(output_dir, f"agent_episode_{episode}"))
            logger.info(f"Saved agent at episode {episode}")
    
    # Save final agent
    agent.save(os.path.join(output_dir, "agent_final"))
    logger.info("Saved final agent")
    
    # Plot training metrics
    plot_training_metrics(update_metrics, output_dir)
    
    # Plot episode returns
    plot_episode_returns(episode_returns, output_dir)
    
    logger.info(f"Training completed. Results saved to {output_dir}")


def evaluate_agent(agent, env, num_episodes=5, max_steps=1000):
    """
    Evaluate the agent.
    
    Args:
        agent: Agent to evaluate
        env: Environment to evaluate in
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        
    Returns:
        Average episode return
    """
    episode_returns = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0
        
        for step in range(max_steps):
            # Select action (deterministic)
            action = agent.select_action(state, evaluate=True)
            
            # Take step in environment
            next_state, reward, done, _, info = env.step(action)
            
            # Update state and return
            state = next_state
            episode_return += reward
            
            # Check if done
            if done:
                break
        
        # Store episode return
        episode_returns.append(episode_return)
    
    # Return average episode return
    return np.mean(episode_returns)


if __name__ == "__main__":
    main()
