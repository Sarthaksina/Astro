#!/usr/bin/env python3
"""
Reinforcement Learning Benchmark

This script benchmarks reinforcement learning performance on GPUs.
It measures the time taken to train a simple RL agent in a simulated environment
and outputs performance metrics in a standardized JSON format.
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Configure benchmark parameters
ENV_SIZE = 64  # Environment grid size
BATCH_SIZE = 128
EPISODES = 100
MAX_STEPS = 200
WARMUP_EPISODES = 10


class SimpleEnvironment:
    """A simple grid world environment for reinforcement learning benchmarking."""
    
    def __init__(self, size=ENV_SIZE, device=torch.device("cuda")):
        self.size = size
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        # Agent position (x, y)
        self.agent_pos = torch.tensor([self.size // 2, self.size // 2], device=self.device)
        # Goal position (x, y)
        self.goal_pos = torch.randint(0, self.size, (2,), device=self.device)
        # Obstacles (random binary grid)
        self.obstacles = torch.rand(self.size, self.size, device=self.device) > 0.8
        # Clear agent and goal positions
        self.obstacles[self.agent_pos[0], self.agent_pos[1]] = False
        self.obstacles[self.goal_pos[0], self.goal_pos[1]] = False
        return self._get_state()
    
    def _get_state(self):
        """Get the current state representation."""
        # Create a 3-channel state: agent position, goal position, obstacles
        state = torch.zeros(3, self.size, self.size, device=self.device)
        state[0, self.agent_pos[0], self.agent_pos[1]] = 1.0  # Agent channel
        state[1, self.goal_pos[0], self.goal_pos[1]] = 1.0  # Goal channel
        state[2] = self.obstacles  # Obstacles channel
        return state
    
    def step(self, action):
        """Take a step in the environment.
        
        Args:
            action: Integer action (0: up, 1: right, 2: down, 3: left)
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        # Action deltas (up, right, down, left)
        deltas = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]], device=self.device)
        
        # Calculate new position
        new_pos = self.agent_pos + deltas[action]
        
        # Check bounds and obstacles
        if (new_pos[0] >= 0 and new_pos[0] < self.size and 
            new_pos[1] >= 0 and new_pos[1] < self.size and 
            not self.obstacles[new_pos[0], new_pos[1]]):
            self.agent_pos = new_pos
        
        # Check if goal reached
        goal_reached = torch.all(self.agent_pos == self.goal_pos)
        
        # Calculate reward
        if goal_reached:
            reward = 10.0
        else:
            # Negative reward based on distance to goal
            distance = torch.sum(torch.abs(self.agent_pos - self.goal_pos))
            reward = -0.1 * distance / self.size
        
        return self._get_state(), reward, goal_reached


class DQNAgent(nn.Module):
    """A simple DQN agent for reinforcement learning benchmarking."""
    
    def __init__(self, state_shape, num_actions):
        super(DQNAgent, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate size after convolutions and pooling
        conv_size = state_shape[1] // 4  # After two 2x2 max pools
        self.fc1 = nn.Linear(64 * conv_size * conv_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReplayBuffer:
    """A simple replay buffer for reinforcement learning."""
    
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer."""
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert to tensors
        state = torch.stack(state)
        action = torch.tensor(action, device=self.device).unsqueeze(1)
        reward = torch.tensor(reward, device=self.device).unsqueeze(1)
        next_state = torch.stack(next_state)
        done = torch.tensor(done, device=self.device, dtype=torch.float).unsqueeze(1)
        
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


def benchmark_reinforcement_learning():
    """Run reinforcement learning benchmark on the current GPU.
    
    Returns:
        Dictionary containing benchmark results
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires a GPU.")
        return {"error": "CUDA not available"}
    
    # Get device information
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    print(f"Running benchmark on: {device_name}")
    
    # Create environment
    env = SimpleEnvironment(size=ENV_SIZE, device=device)
    state_shape = env.reset().shape
    num_actions = 4  # up, right, down, left
    
    # Create agent and optimizer
    agent = DQNAgent(state_shape, num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(10000, device)
    
    # Hyperparameters
    gamma = 0.99  # Discount factor
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    epsilon = epsilon_start
    
    # Warmup
    print(f"Performing warmup for {WARMUP_EPISODES} episodes...")
    for episode in range(WARMUP_EPISODES):
        state = env.reset()
        for step in range(MAX_STEPS):
            # Select action
            if torch.rand(1).item() < epsilon:
                action = torch.randint(0, num_actions, (1,)).item()
            else:
                with torch.no_grad():
                    action = agent(state.unsqueeze(0)).argmax().item()
            
            # Take step in environment
            next_state, reward, done = env.step(action)
            
            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    # Benchmark training
    print(f"\nBenchmarking RL training for {EPISODES} episodes...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    total_steps = 0
    total_rewards = 0
    episode_lengths = []
    
    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            # Select action
            if torch.rand(1).item() < epsilon:
                action = torch.randint(0, num_actions, (1,)).item()
            else:
                with torch.no_grad():
                    action = agent(state.unsqueeze(0)).argmax().item()
            
            # Take step in environment
            next_state, reward, done = env.step(action)
            episode_reward += reward.item()
            
            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            total_steps += 1
            
            # Train agent if enough samples in replay buffer
            if len(replay_buffer) >= BATCH_SIZE:
                # Sample batch from replay buffer
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                
                # Compute Q values
                q_values = agent(states).gather(1, actions)
                
                # Compute target Q values
                with torch.no_grad():
                    next_q_values = agent(next_states).max(1, keepdim=True)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                # Compute loss and update
                loss = F.smooth_l1_loss(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Record statistics
        episode_lengths.append(step + 1)
        total_rewards += episode_reward
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}: Steps: {step+1}, Reward: {episode_reward:.2f}")
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    # Calculate performance metrics
    steps_per_second = total_steps / total_time
    episodes_per_second = EPISODES / total_time
    avg_episode_length = sum(episode_lengths) / len(episode_lengths)
    avg_reward = total_rewards / EPISODES
    
    # Prepare final results
    benchmark_results = {
        "device": device_name,
        "env_size": ENV_SIZE,
        "episodes": EPISODES,
        "total_steps": total_steps,
        "total_time": total_time,
        "steps_per_second": steps_per_second,
        "episodes_per_second": episodes_per_second,
        "avg_episode_length": avg_episode_length,
        "avg_reward": avg_reward,
        "performance": steps_per_second,  # Steps/second as the main performance metric
        "unit": "steps/second",
        "higher_is_better": True
    }
    
    return benchmark_results


def main():
    """Run the benchmark and output results as JSON."""
    results = benchmark_reinforcement_learning()
    print(json.dumps(results))


if __name__ == "__main__":
    main()