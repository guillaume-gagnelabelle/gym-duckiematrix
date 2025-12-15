"""
PPO (Proximal Policy Optimization) implementation for Duckiematrix environment.

PPO is an on-policy algorithm that:
1. Collects batches of experience
2. Uses clipped objective to prevent large policy updates
3. Performs multiple epochs of updates on the same batch
4. Can use a value function (actor-critic) for variance reduction
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
import argparse


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
        
    def forward(self, obs, min_std=0.1):
        """Forward pass through the network."""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc_mean(x))  # Tanh to keep actions in [-1, 1]
        std = F.softplus(self.fc_std(x)) + min_std  # Softplus ensures std > 0, add minimum for exploration
        return mean, std


class ValueNetwork(nn.Module):
    """Value network for estimating state values (reduces variance in PPO)."""
    
    def __init__(self, obs_dim=2, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc_value.weight)
        
    def forward(self, obs):
        """Forward pass through the network."""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        value = self.fc_value(x)
        return value


class PPOAgent:
    """PPO agent implementation."""
    
    def __init__(self, obs_dim=2, action_dim=2, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=4, use_value=True, device='cpu',
                 exploration_noise=0.1, epsilon_start=0.3, epsilon_end=0.05, epsilon_decay=0.995):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter (typically 0.1-0.3)
            k_epochs: Number of epochs to update on same batch
            use_value: Whether to use value function (actor-critic)
            device: Device to run on
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.use_value = use_value
        self.device = device
        
        # Policy network (actor)
        self.policy = PolicyNetwork(obs_dim, action_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Value network (critic) - optional
        if use_value:
            self.value = ValueNetwork(obs_dim).to(device)
            self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        else:
            self.value = None
        
        # Exploration parameters
        self.exploration_noise = exploration_noise
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.min_std = 0.15  # Minimum std for exploration (prevents premature convergence)
        
        # Batch storage
        self.reset_batch()
    
    def load_checkpoint(self, policy_path, value_path=None):
        """
        Load checkpoint from saved model files.
        
        Args:
            policy_path: Path to saved policy network state dict
            value_path: Path to saved value network state dict (optional, only needed if use_value=True)
        """
        try:
            self.policy.load_state_dict(torch.load(policy_path, map_location=self.device))
            print(f"Loaded policy checkpoint from {policy_path}")
            
            if self.use_value and value_path is not None:
                self.value.load_state_dict(torch.load(value_path, map_location=self.device))
                print(f"Loaded value checkpoint from {value_path}")
            elif self.use_value and value_path is None:
                print("Warning: use_value=True but no value_path provided. Value network not loaded.")
        except FileNotFoundError as e:
            print(f"Error loading checkpoint: {e}")
            raise
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
        
    def reset_batch(self):
        """Reset batch buffers."""
        self.batch_obs = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_log_probs = []
        self.batch_dones = []
        self.batch_values = []  # Only used if use_value=True
        
    def select_action(self, obs, apply_exploration=True):
        """Select an action using the current policy with exploration."""
        # Check for NaN in observation
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"Warning: Invalid observation detected: {obs}, using zeros")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy exploration: random action with probability epsilon
        if apply_exploration and np.random.random() < self.epsilon:
            # Random action, but bias toward forward motion
            action = np.random.uniform(-1.0, 1.0, size=2)
            if np.random.random() < 0.7:  # 70% chance of forward-biased action
                action = np.clip(action + 0.3, -1.0, 1.0)
            action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            # Create a dummy log_prob for storage (will be recomputed during update)
            # Must have same shape and device as policy log_prob: [1] not scalar, on correct device
            log_prob = torch.tensor([0.0], device=self.device)
        else:
            # Get action distribution
            mean, std = self.policy(obs_tensor, min_std=self.min_std)
            
            # Check for NaN in network output
            if torch.any(torch.isnan(mean)) or torch.any(torch.isnan(std)):
                print(f"Warning: NaN detected in policy output, using default actions")
                mean = torch.zeros_like(mean)
                std = torch.ones_like(std) * 0.1
            
            # Ensure std is positive and reasonable (with minimum for exploration)
            std = torch.clamp(std, min=self.min_std, max=1.0)
            
            dist = torch.distributions.Normal(mean, std)
            
            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Add exploration noise
            if apply_exploration:
                noise = torch.randn_like(action) * self.exploration_noise
                action = action + noise
        
        # Clip action to valid range
        action = torch.clamp(action, -1.0, 1.0)
        
        # Get value estimate if using value function
        value = None
        if self.use_value:
            with torch.no_grad():
                value = self.value(obs_tensor)
        
        # Store for training
        self.batch_obs.append(obs)
        self.batch_actions.append(action.cpu().numpy().flatten())
        self.batch_log_probs.append(log_prob)
        if value is not None:
            self.batch_values.append(value.cpu().item())
        
        return action.cpu().numpy().flatten()
    
    def decay_epsilon(self):
        """Decay epsilon for epsilon-greedy exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, reward, done):
        """Store reward and done flag for the current step."""
        self.batch_rewards.append(reward)
        self.batch_dones.append(done)
    
    def compute_returns_and_advantages(self):
        """Compute discounted returns and advantages."""
        returns = []
        advantages = []
        
        # Compute discounted returns
        G = 0
        for reward, done in zip(reversed(self.batch_rewards), reversed(self.batch_dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        if self.use_value and len(self.batch_values) > 0:
            # Compute advantages using value function
            values = torch.FloatTensor(self.batch_values).to(self.device)
            advantages = returns - values
            
            # Normalize advantages
            if len(advantages) > 1:
                advantages_std = advantages.std()
                if advantages_std > 1e-4:
                    advantages = (advantages - advantages.mean()) / (advantages_std + 1e-8)
                else:
                    # If all advantages are similar, keep raw but ensure non-zero
                    if torch.allclose(advantages, advantages[0], atol=1e-6):
                        pass  # Keep as-is
                    else:
                        advantages = (advantages - advantages.mean()) * 0.1
        else:
            # Use returns as advantages (no value function)
            advantages = returns
            # Normalize advantages
            if len(advantages) > 1:
                advantages_mean = advantages.mean()
                advantages_std = advantages.std()
                if advantages_std > 1e-4:
                    advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
                else:
                    if torch.allclose(advantages, advantages[0], atol=1e-6):
                        pass  # Keep as-is
                    else:
                        advantages = (advantages - advantages_mean) * 0.1
        
        return returns, advantages
    
    def update(self):
        """Update policy using PPO algorithm."""
        if len(self.batch_rewards) == 0:
            return 0.0, 0.0
        
        # Convert batch to tensors
        obs_tensor = torch.FloatTensor(np.array(self.batch_obs)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.batch_actions)).to(self.device)
        old_log_probs = torch.stack(self.batch_log_probs).to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages()
        
        # Check for NaN
        if torch.any(torch.isnan(advantages)) or torch.any(torch.isnan(returns)):
            print("Warning: NaN detected in advantages/returns, skipping update")
            self.reset_batch()
            return 0.0, 0.0
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        # Multiple epochs of updates on the same batch
        for epoch in range(self.k_epochs):
            # Get current policy distribution
            mean, std = self.policy(obs_tensor, min_std=self.min_std)
            std = torch.clamp(std, min=self.min_std, max=1.0)
            dist = torch.distributions.Normal(mean, std)
            
            # Compute new log probabilities
            new_log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
            
            # Compute ratio (importance sampling)
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            
            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Check for NaN in policy loss
            if torch.isnan(policy_loss):
                print(f"Warning: NaN in policy loss at epoch {epoch}, skipping")
                break
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.policy_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            
            # Update value function if using it
            if self.use_value:
                values = self.value(obs_tensor).squeeze()
                value_loss = F.mse_loss(values, returns)
                
                # Check for NaN in value loss
                if torch.isnan(value_loss):
                    print(f"Warning: NaN in value loss at epoch {epoch}, skipping")
                    break
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
                self.value_optimizer.step()
                
                total_value_loss += value_loss.item()
        
        # Reset batch
        self.reset_batch()
        
        avg_policy_loss = total_policy_loss / self.k_epochs if self.k_epochs > 0 else 0.0
        avg_value_loss = total_value_loss / self.k_epochs if self.k_epochs > 0 else 0.0
        
        return avg_policy_loss, avg_value_loss


def train_ppo(num_episodes=1000, max_steps_per_episode=1000, 
              batch_size=2048, save_freq=100, use_value=True,
              policy_checkpoint=None, value_checkpoint=None, start_episode=0):
    """
    Train PPO agent on Duckiematrix environment.
    
    Args:
        num_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        batch_size: Number of steps to collect before updating (PPO batch size)
        save_freq: Frequency to save model
        use_value: Whether to use value function
        policy_checkpoint: Path to policy checkpoint to load (for resuming training)
        value_checkpoint: Path to value checkpoint to load (for resuming training)
        start_episode: Starting episode number (for resuming training, affects save naming)
    """
    
    # Create environment
    env = DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0")
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    agent = PPOAgent(obs_dim=2, action_dim=2, lr=3e-4, gamma=0.99, 
                     eps_clip=0.2, k_epochs=4, use_value=use_value, device=device)
    
    # Load checkpoint if provided
    if policy_checkpoint is not None:
        agent.load_checkpoint(policy_checkpoint, value_checkpoint)
        print(f"Resuming training from episode {start_episode}")
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    
    print("Starting PPO training...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Batch size: {batch_size}, Epochs per update: {agent.k_epochs}")
    print(f"Using value function: {use_value}")
    
    total_steps = 0
    
    for episode in range(start_episode, start_episode + num_episodes):
        # Always use random reset: 60% curved tiles, 40% straight tiles
        obs, info = env.reset(curve_prob=0.6)
        
        episode_reward = 0
        episode_length = 0
        last_pose = None
        
        for step in range(max_steps_per_episode):
            # Select action with exploration
            action = agent.select_action(obs, apply_exploration=True)
            # Decay epsilon periodically
            if total_steps % 100 == 0:
                agent.decay_epsilon()
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store transition
            done = terminated or truncated
            agent.store_transition(reward, done)
            
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Track last pose for reset
            if not terminated:
                last_pose = info.get("pose")
            
            # Check if episode is done
            if done:
                break
            
            obs = next_obs
            sleep(0.01)  # Small delay to prevent overwhelming the simulator
            
            # Update if we've collected enough steps
            if total_steps % batch_size == 0:
                policy_loss, value_loss = agent.update()
                if policy_loss > 0:  # Only print if update was successful
                    print(f"  Batch update | Policy Loss: {policy_loss:.4f} | "
                          f"Value Loss: {value_loss:.4f}")
        
        # Store statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Update at end of episode if we have collected some data
        if len(agent.batch_obs) > 0:
            policy_loss, value_loss = agent.update()
        else:
            policy_loss, value_loss = 0.0, 0.0
        
        # Print progress
        episode_num = episode + 1
        if episode_num % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            loss_str = f"{policy_loss:.4f}" if policy_loss != 0.0 else "0.0000"
            value_str = f"{value_loss:.4f}" if value_loss != 0.0 else "0.0000"
            print(f"Episode {episode_num}/{start_episode + num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Policy Loss: {loss_str} | "
                  f"Value Loss: {value_str}")
        
        # Save model periodically
        if episode_num % save_freq == 0:
            torch.save(agent.policy.state_dict(), f"ppo_policy_ep{episode_num}.pth")
            if agent.use_value:
                torch.save(agent.value.state_dict(), f"ppo_value_ep{episode_num}.pth")
            print(f"Saved model at episode {episode_num}")
    
    # Final save
    torch.save(agent.policy.state_dict(), "ppo_policy_final.pth")
    if agent.use_value:
        torch.save(agent.value.state_dict(), "ppo_value_final.pth")
    print("Training complete! Model saved to ppo_policy_final.pth")
    
    # Cleanup
    try:
        env.robot.camera.stop()
    except:
        pass  # Camera may not be started
    env.robot.motors.stop()
    
    return agent, episode_rewards, episode_lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent on Duckiematrix environment')
    parser.add_argument('--num_episodes', type=int, default=500,
                        help='Number of episodes to train (default: 500)')
    parser.add_argument('--max_steps_per_episode', type=int, default=2000,
                        help='Maximum steps per episode (default: 2000)')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Number of steps to collect before updating (default: 2048)')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Frequency to save model (default: 50)')
    parser.add_argument('--use_value', action='store_true', default=True,
                        help='Use value function (actor-critic) (default: True)')
    parser.add_argument('--no_value', dest='use_value', action='store_false',
                        help='Disable value function')
    parser.add_argument('--policy_checkpoint', type=str, default=None,
                        help='Path to policy checkpoint to load (for resuming training)')
    parser.add_argument('--value_checkpoint', type=str, default=None,
                        help='Path to value checkpoint to load (for resuming training)')
    parser.add_argument('--start_episode', type=int, default=0,
                        help='Starting episode number (for resuming training, affects save naming)')
    
    args = parser.parse_args()
    
    # Train the agent
    agent, rewards, lengths = train_ppo(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        batch_size=args.batch_size,
        save_freq=args.save_freq,
        use_value=args.use_value,
        policy_checkpoint=args.policy_checkpoint,
        value_checkpoint=args.value_checkpoint,
        start_episode=args.start_episode
    )

