#!/usr/bin/env python
# NOTE: This file was automatically updated to use the new modular hierarchical RL framework.
# Please review the changes and refer to docs/migration_guide_hierarchical_rl.md for more information.


# Cosmic Market Oracle - Tests for Modular Hierarchical RL

"""
Unit tests for the modular hierarchical reinforcement learning framework.

This module contains tests for the base components, strategic planner,
and tactical executor of the modular hierarchical RL framework.
"""

import os
import sys
import numpy as np
import torch
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.trading.modular_hierarchical_rl import (
    StrategicPlannerInterface,
    TacticalExecutorInterface,
    ModularHierarchicalRLAgent,
    MCTSStrategicPlanner,
    PPOTacticalExecutor
)


class MockStrategicPlanner(StrategicPlannerInterface):
    """Mock strategic planner for testing."""
    
    def __init__(self, goal_dim=8):
        self.goal_dim = goal_dim
    
    def plan(self, state, evaluate=False):
        return np.random.randn(self.goal_dim)
    
    def update(self, states, goals, rewards, next_states, dones):
        return {"mock_loss": 0.5}
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass


class MockTacticalExecutor(TacticalExecutorInterface):
    """Mock tactical executor for testing."""
    
    def __init__(self, action_dim=4):
        self.action_dim = action_dim
    
    def execute(self, state, goal, evaluate=False):
        return np.random.randn(self.action_dim)
    
    def update(self, states, goals, actions, rewards, next_states, next_goals, dones):
        return {"mock_loss": 0.5}
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass


class TestModularHierarchicalRLAgent:
    """Tests for ModularHierarchicalRLAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state_dim = 10
        self.goal_dim = 8
        self.action_dim = 4
        self.strategic_horizon = 5
        
        self.strategic_planner = MockStrategicPlanner(goal_dim=self.goal_dim)
        self.tactical_executor = MockTacticalExecutor(action_dim=self.action_dim)
        
        self.agent = ModularModularHierarchicalRLAgent(
            strategic_planner=self.strategic_planner,
            tactical_executor=self.tactical_executor,
            strategic_horizon=self.strategic_horizon
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.strategic_planner is self.strategic_planner
        assert self.agent.tactical_executor is self.tactical_executor
        assert self.agent.strategic_horizon == self.strategic_horizon
        assert self.agent.current_goal is None
        assert self.agent.goal_steps == 0
    
    def test_select_action_new_goal(self):
        """Test action selection with new goal."""
        # Mock strategic planner
        self.strategic_planner.plan = MagicMock(return_value=np.zeros(self.goal_dim))
        
        # Mock tactical executor
        self.tactical_executor.execute = MagicMock(return_value=np.zeros(self.action_dim))
        
        # Select action
        state = np.random.randn(self.state_dim)
        action = self.agent.select_action(state)
        
        # Check that strategic planner was called
        self.strategic_planner.plan.assert_called_once_with(state, False)
        
        # Check that tactical executor was called
        self.tactical_executor.execute.assert_called_once()
        
        # Check that goal steps was incremented
        assert self.agent.goal_steps == 1
        
        # Check that action has correct shape
        assert action.shape == (self.action_dim,)
    
    def test_select_action_existing_goal(self):
        """Test action selection with existing goal."""
        # Set up existing goal
        self.agent.current_goal = np.zeros(self.goal_dim)
        self.agent.goal_steps = 2
        
        # Mock strategic planner
        self.strategic_planner.plan = MagicMock()
        
        # Mock tactical executor
        self.tactical_executor.execute = MagicMock(return_value=np.zeros(self.action_dim))
        
        # Select action
        state = np.random.randn(self.state_dim)
        action = self.agent.select_action(state)
        
        # Check that strategic planner was not called
        self.strategic_planner.plan.assert_not_called()
        
        # Check that tactical executor was called
        self.tactical_executor.execute.assert_called_once()
        
        # Check that goal steps was incremented
        assert self.agent.goal_steps == 3
        
        # Check that action has correct shape
        assert action.shape == (self.action_dim,)
    
    def test_select_action_horizon_reached(self):
        """Test action selection when horizon is reached."""
        # Set up existing goal at horizon
        self.agent.current_goal = np.zeros(self.goal_dim)
        self.agent.goal_steps = self.strategic_horizon
        
        # Mock strategic planner
        self.strategic_planner.plan = MagicMock(return_value=np.ones(self.goal_dim))
        
        # Mock tactical executor
        self.tactical_executor.execute = MagicMock(return_value=np.zeros(self.action_dim))
        
        # Select action
        state = np.random.randn(self.state_dim)
        action = self.agent.select_action(state)
        
        # Check that strategic planner was called
        self.strategic_planner.plan.assert_called_once_with(state, False)
        
        # Check that tactical executor was called
        self.tactical_executor.execute.assert_called_once()
        
        # Check that goal steps was reset and incremented
        assert self.agent.goal_steps == 1
        
        # Check that goal was updated
        assert np.array_equal(self.agent.current_goal, np.ones(self.goal_dim))
        
        # Check that action has correct shape
        assert action.shape == (self.action_dim,)
    
    def test_store_experience(self):
        """Test storing experience."""
        # Set up current goal
        self.agent.current_goal = np.zeros(self.goal_dim)
        
        # Create mock buffers
        self.agent.tactical_buffer = MagicMock()
        self.agent.strategic_buffer = MagicMock()
        
        # Store experience
        state = np.random.randn(self.state_dim)
        action = np.random.randn(self.action_dim)
        reward = 1.0
        next_state = np.random.randn(self.state_dim)
        done = False
        
        self.agent.store_experience(state, action, reward, next_state, done)
        
        # Check that tactical buffer was called
        self.agent.tactical_buffer.add.assert_called_once()
        
        # Check that strategic buffer was not called (not at horizon)
        self.agent.strategic_buffer.add.assert_not_called()
        
        # Store experience at horizon
        self.agent.goal_steps = self.strategic_horizon
        
        self.agent.store_experience(state, action, reward, next_state, done)
        
        # Check that strategic buffer was called
        self.agent.strategic_buffer.add.assert_called_once()
    
    def test_update(self):
        """Test agent update."""
        # Create mock buffers with enough samples
        self.agent.tactical_buffer = MagicMock()
        self.agent.tactical_buffer.__len__ = MagicMock(return_value=1000)
        self.agent.tactical_buffer.sample = MagicMock(return_value=(
            np.random.randn(256, self.state_dim + self.goal_dim),
            np.random.randn(256, self.action_dim),
            np.random.randn(256),
            np.random.randn(256, self.state_dim + self.goal_dim),
            np.random.randn(256)
        ))
        
        self.agent.strategic_buffer = MagicMock()
        self.agent.strategic_buffer.__len__ = MagicMock(return_value=1000)
        self.agent.strategic_buffer.sample = MagicMock(return_value=(
            np.random.randn(256, self.state_dim),
            np.random.randn(256, self.goal_dim),
            np.random.randn(256),
            np.random.randn(256, self.state_dim),
            np.random.randn(256)
        ))
        
        # Set up current goal
        self.agent.current_goal = np.zeros(self.goal_dim)
        
        # Mock component updates
        self.strategic_planner.update = MagicMock(return_value={"strategic_loss": 0.5})
        self.tactical_executor.update = MagicMock(return_value={"tactical_loss": 0.3})
        
        # Update agent
        metrics = self.agent.update(batch_size=256)
        
        # Check that components were updated
        self.strategic_planner.update.assert_called_once()
        self.tactical_executor.update.assert_called_once()
        
        # Check metrics
        assert "strategic_strategic_loss" in metrics
        assert "tactical_tactical_loss" in metrics
        assert metrics["strategic_strategic_loss"] == 0.5
        assert metrics["tactical_tactical_loss"] == 0.3
    
    def test_save_load(self):
        """Test saving and loading."""
        # Mock component save/load
        self.strategic_planner.save = MagicMock()
        self.tactical_executor.save = MagicMock()
        self.strategic_planner.load = MagicMock()
        self.tactical_executor.load = MagicMock()
        
        # Save agent
        self.agent.save("test_path")
        
        # Check that components were saved
        self.strategic_planner.save.assert_called_once_with("test_path_strategic")
        self.tactical_executor.save.assert_called_once_with("test_path_tactical")
        
        # Load agent
        self.agent.load("test_path")
        
        # Check that components were loaded
        self.strategic_planner.load.assert_called_once_with("test_path_strategic")
        self.tactical_executor.load.assert_called_once_with("test_path_tactical")


class TestMCTSStrategicPlanner:
    """Tests for MCTSStrategicPlanner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state_dim = 10
        self.goal_dim = 8
        self.hidden_dim = 64
        self.num_simulations = 10
        
        # Use CPU for tests
        self.device = "cpu"
        
        self.planner = MCTSStrategicPlanner(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            hidden_dim=self.hidden_dim,
            num_simulations=self.num_simulations,
            device=self.device
        )
    
    def test_initialization(self):
        """Test planner initialization."""
        assert self.planner.state_dim == self.state_dim
        assert self.planner.goal_dim == self.goal_dim
        assert self.planner.hidden_dim == self.hidden_dim
        assert self.planner.num_simulations == self.num_simulations
        assert self.planner.device == self.device
        
        # Check that predictor and MCTS were created
        assert self.planner.predictor is not None
        assert self.planner.mcts is not None
    
    def test_plan(self):
        """Test planning."""
        # Mock MCTS
        self.planner.mcts.select_action = MagicMock(return_value=np.zeros(self.goal_dim))
        
        # Plan
        state = np.random.randn(self.state_dim)
        goal = self.planner.plan(state)
        
        # Check that MCTS was called
        self.planner.mcts.select_action.assert_called_once()
        
        # Check that goal has correct shape
        assert goal.shape == (self.goal_dim,)
    
    def test_update(self):
        """Test updating."""
        # Create batch data
        batch_size = 8
        states = np.random.randn(batch_size, self.state_dim)
        goals = np.random.randn(batch_size, self.goal_dim)
        rewards = np.random.randn(batch_size)
        next_states = np.random.randn(batch_size, self.state_dim)
        dones = np.zeros(batch_size)
        
        # Update
        metrics = self.planner.update(states, goals, rewards, next_states, dones)
        
        # Check metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "total_loss" in metrics
    
    def test_save_load(self, tmp_path):
        """Test saving and loading."""
        # Create temporary path
        path = os.path.join(tmp_path, "test_planner")
        
        # Save planner
        self.planner.save(path)
        
        # Check that file was created
        assert os.path.exists(path + ".pt")
        
        # Create new planner
        new_planner = MCTSStrategicPlanner(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            hidden_dim=self.hidden_dim,
            num_simulations=self.num_simulations,
            device=self.device
        )
        
        # Load planner
        new_planner.load(path)
        
        # Check that parameters were loaded
        for p1, p2 in zip(self.planner.predictor.parameters(), new_planner.predictor.parameters()):
            assert torch.allclose(p1, p2)


class TestPPOTacticalExecutor:
    """Tests for PPOTacticalExecutor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state_dim = 10
        self.goal_dim = 8
        self.action_dim = 4
        self.hidden_dim = 64
        
        # Use CPU for tests
        self.device = "cpu"
        
        self.executor = PPOTacticalExecutor(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            device=self.device
        )
    
    def test_initialization(self):
        """Test executor initialization."""
        assert self.executor.state_dim == self.state_dim
        assert self.executor.goal_dim == self.goal_dim
        assert self.executor.action_dim == self.action_dim
        assert self.executor.hidden_dim == self.hidden_dim
        assert self.executor.device == self.device
        
        # Check that actor and critic were created
        assert self.executor.actor is not None
        assert self.executor.critic is not None
    
    def test_execute(self):
        """Test execution."""
        # Execute
        state = np.random.randn(self.state_dim)
        goal = np.random.randn(self.goal_dim)
        action = self.executor.execute(state, goal)
        
        # Check that action has correct shape
        assert action.shape == (self.action_dim,)
    
    def test_update(self):
        """Test updating."""
        # Create batch data
        batch_size = 8
        states = np.random.randn(batch_size, self.state_dim)
        goals = np.random.randn(batch_size, self.goal_dim)
        actions = np.random.randn(batch_size, self.action_dim)
        rewards = np.random.randn(batch_size)
        next_states = np.random.randn(batch_size, self.state_dim)
        next_goals = np.random.randn(batch_size, self.goal_dim)
        dones = np.zeros(batch_size)
        
        # Mock actor and critic for faster testing
        self.executor.actor = MagicMock()
        self.executor.critic = MagicMock()
        
        # Mock distributions
        mock_dist = MagicMock()
        mock_dist.log_prob = MagicMock(return_value=torch.zeros(batch_size, self.action_dim))
        mock_dist.entropy = MagicMock(return_value=torch.zeros(batch_size))
        
        self.executor.actor.return_value = mock_dist
        self.executor.critic.return_value = torch.zeros(batch_size, 1)
        
        # Update
        metrics = self.executor.update(states, goals, actions, rewards, next_states, next_goals, dones)
        
        # Check metrics
        assert "actor_loss" in metrics
        assert "critic_loss" in metrics
        assert "entropy_loss" in metrics
        assert "total_loss" in metrics
    
    def test_save_load(self, tmp_path):
        """Test saving and loading."""
        # Create temporary path
        path = os.path.join(tmp_path, "test_executor")
        
        # Save executor
        self.executor.save(path)
        
        # Check that file was created
        assert os.path.exists(path + ".pt")
        
        # Create new executor
        new_executor = PPOTacticalExecutor(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            device=self.device
        )
        
        # Load executor
        new_executor.load(path)
        
        # Check that parameters were loaded
        for p1, p2 in zip(self.executor.actor.parameters(), new_executor.actor.parameters()):
            assert torch.allclose(p1, p2)
        
        for p1, p2 in zip(self.executor.critic.parameters(), new_executor.critic.parameters()):
            assert torch.allclose(p1, p2)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
