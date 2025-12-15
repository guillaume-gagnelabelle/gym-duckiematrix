"""
REINFORCE (Policy Gradient) implementation for Duckiematrix environment.

This is a simple on-policy algorithm that:
1. Collects full episodes
2. Computes discounted returns
3. Updates the policy using the policy gradient theorem
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from time import sleep
import math


class PolicyNetwork(nn.Module):
    """Simple MLP policy network that outputs mean and std for a Gaussian action distribution."""
    
    def __init__(self, obs_dim=2, action_dim=2, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.xavier_uniform_(self.fc_std.weight)
        
    def forward(self, obs, min_std=0.15):
        """Forward pass through the network."""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc_mean(x))  # Tanh to keep actions in [-1, 1]
        std = F.softplus(self.fc_std(x)) + min_std  # Softplus ensures std > 0, add minimum for exploration
        return mean, std


class REINFORCEAgent:
    """REINFORCE agent implementation."""
    
    def __init__(self, obs_dim=2, action_dim=2, lr=3e-4, gamma=0.99, device='cpu',
                 exploration_noise=0.1, epsilon_start=0.3, epsilon_end=0.05, epsilon_decay=0.995):
        self.gamma = gamma
        self.device = device
        
        # Policy network
        self.policy = PolicyNetwork(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Exploration parameters
        self.exploration_noise = exploration_noise
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.min_std = 0.15  # Minimum std for exploration
        
        # Episode storage
        self.reset_episode()
        
    def reset_episode(self):
        """Reset episode buffers."""
        self.episode_obs = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
    def select_action(self, obs, apply_exploration=True):
        """Select an action using the current policy with exploration."""
        # Check for NaN in observation
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"Warning: Invalid observation detected: {obs}, using zeros")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Get action distribution (always compute for proper log_prob)
        mean, std = self.policy(obs_tensor, min_std=self.min_std)
        
        # Check for NaN in network output
        if torch.any(torch.isnan(mean)) or torch.any(torch.isnan(std)):
            print(f"Warning: NaN detected in policy output, using default actions")
            mean = torch.zeros_like(mean)
            std = torch.ones_like(std) * 0.1
        
        # Ensure std is positive and reasonable (with minimum for exploration)
        std = torch.clamp(std, min=self.min_std, max=1.0)
        
        dist = torch.distributions.Normal(mean, std)
        
        # Epsilon-greedy exploration: random action with probability epsilon
        if apply_exploration and np.random.random() < self.epsilon:
            # Random action, but bias toward forward motion
            action_np = np.random.uniform(-1.0, 1.0, size=2)
            if np.random.random() < 0.7:  # 70% chance of forward-biased action
                action_np = np.clip(action_np + 0.3, -1.0, 1.0)
            action = torch.FloatTensor(action_np).unsqueeze(0).to(self.device)
            # Compute log_prob for the random action under current policy
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            # Sample action from policy
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Add exploration noise
            if apply_exploration:
                noise = torch.randn_like(action) * self.exploration_noise
                action = action + noise
                # Recompute log_prob after adding noise (approximate)
                log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clip action to valid range
        action = torch.clamp(action, -1.0, 1.0)
        
        # Store for training
        self.episode_obs.append(obs)
        self.episode_actions.append(action.cpu().numpy().flatten())
        self.episode_log_probs.append(log_prob)
        
        return action.cpu().numpy().flatten()
    
    def decay_epsilon(self):
        """Decay epsilon for epsilon-greedy exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_reward(self, reward):
        """Store reward for the current step."""
        self.episode_rewards.append(reward)
    
    def update(self):
        """Update policy using REINFORCE algorithm."""
        if len(self.episode_rewards) == 0:
            return 0.0
        
        # Compute discounted returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns (reduces variance) - only if we have enough samples
        if len(returns) > 1:
            returns_mean = returns.mean()
            returns_std = returns.std()
            if returns_std > 1e-4:  # Only normalize if std is significant
                returns = (returns - returns_mean) / (returns_std + 1e-8)
            else:
                # If all returns are very similar, DON'T zero them out
                # Instead, use raw returns (they still provide a learning signal)
                # The mean subtraction would zero them, so we skip that
                # Just ensure they're not all exactly zero
                if torch.allclose(returns, returns[0], atol=1e-6):
                    # All returns are essentially the same - use raw returns
                    # This ensures we still have a gradient signal
                    pass  # Keep returns as-is
                else:
                    # Small variance - center but don't divide
                    returns = returns - returns_mean
                    # Add small scale to ensure meaningful gradients
                    returns = returns * 0.1  # Scale down but don't zero
        else:
            # Single return - can't normalize, but still use it
            pass
        
        # Convert to tensors
        log_probs = torch.stack(self.episode_log_probs)
        
        # Check for NaN in log_probs
        if torch.any(torch.isnan(log_probs)):
            print("Warning: NaN detected in log_probs, skipping update")
            self.reset_episode()
            return 0.0
        
        # Compute policy gradient loss (negative because we want to maximize)
        loss = -(log_probs * returns).mean()
        
        # Check for NaN in loss
        if torch.isnan(loss):
            print("Warning: NaN loss detected, skipping update")
            self.reset_episode()
            return 0.0
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Reset episode
        self.reset_episode()
        
        return loss.item()


def train_reinforce(num_episodes=1000, max_steps_per_episode=1000, save_freq=100):
    """Train REINFORCE agent on Duckiematrix environment."""
    
    # Create environment
    env = DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0")
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    agent = REINFORCEAgent(obs_dim=2, action_dim=2, lr=3e-4, gamma=0.99, device=device)
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    
    print("Starting REINFORCE training...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    for episode in range(num_episodes):
        # Always use random reset: 60% curved tiles, 40% straight tiles
        obs, info = env.reset(curve_prob=0.6)
        
        agent.reset_episode()
        
        episode_reward = 0
        episode_length = 0
        last_pose = None
        
        for step in range(max_steps_per_episode):
            # Select action with exploration
            action = agent.select_action(obs, apply_exploration=True)
            # Decay epsilon periodically
            if step % 50 == 0:
                agent.decay_epsilon()
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store reward
            agent.store_reward(reward)
            episode_reward += reward
            episode_length += 1
            
            # Track last pose for reset
            if not terminated:
                last_pose = info.get("pose")
            
            # Check if episode is done
            done = terminated or truncated
            
            if done:
                break
            
            obs = next_obs
            sleep(0.01)  # Small delay to prevent overwhelming the simulator
        
        # Update policy after episode
        loss = agent.update()
        
        # Store statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            # Debug: show if loss was skipped
            loss_str = f"{loss:.4f}" if loss != 0.0 else "0.0000 (skipped)"
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Loss: {loss_str}")
        
        # Save model periodically
        if (episode + 1) % save_freq == 0:
            torch.save(agent.policy.state_dict(), f"reinforce_policy_ep{episode + 1}.pth")
            print(f"Saved model at episode {episode + 1}")
    
    # Final save
    torch.save(agent.policy.state_dict(), "reinforce_policy_final.pth")
    print("Training complete! Model saved to reinforce_policy_final.pth")
    
    # Cleanup
    try:
        env.robot.camera.stop()
    except:
        pass  # Camera may not be started
    env.robot.motors.stop()
    
    return agent, episode_rewards, episode_lengths


if __name__ == "__main__":
    # Train the agent
    agent, rewards, lengths = train_reinforce(
        num_episodes=500,
        max_steps_per_episode=2000,
        save_freq=50
    )

