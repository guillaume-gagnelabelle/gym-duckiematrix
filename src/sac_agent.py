"""
SAC (Soft Actor-Critic) implementation for Duckiematrix environment.

SAC is an off-policy algorithm that:
1. Uses experience replay buffer
2. Learns Q-functions (critics) and policy (actor)
3. Uses entropy regularization for exploration
4. Updates target networks using soft updates (polyak averaging)
5. Uses reparameterization trick for continuous actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from duckietown.sdk.utils.loop_lane_position import get_closest_tile
from time import sleep
import math
import argparse
import random


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-network (critic) that estimates Q(s, a)."""
    
    def __init__(self, obs_dim=2, action_dim=2, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, state, action):
        """Forward pass: Q(s, a)."""
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class PolicyNetwork(nn.Module):
    """Policy network (actor) that outputs a squashed Gaussian distribution."""
    
    def __init__(self, obs_dim=2, action_dim=2, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.xavier_uniform_(self.fc_log_std.weight)
    
    def forward(self, state):
        """Forward pass: returns mean and log_std for action distribution."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state, epsilon=1e-6):
        """Sample action using reparameterization trick."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Sample from standard normal
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Squash to [-1, 1] using tanh, then map to [0, 1] to prevent backward motion
        action = torch.tanh(x_t)
        action = 0.5 * (action + 1.0)
        
        # Compute log probability (with tanh correction)
        log_prob = normal.log_prob(x_t)
        # Tanh correction: log(1 - tanh^2(x))
        tanh_action = 2.0 * action - 1.0  # recover tanh(action) for correction
        log_prob -= torch.log(1 - tanh_action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # Adjust for scaling from [-1,1] to [0,1] (Jacobian |0.5| per dim)
        log_prob -= math.log(2.0) * action.shape[-1]
        
        return action, log_prob
    
    def deterministic_action(self, state):
        """Get deterministic action (mean of distribution, squashed)."""
        mean, _ = self.forward(state)
        return 0.5 * (torch.tanh(mean) + 1.0)


class SACAgent:
    """SAC (Soft Actor-Critic) agent implementation."""
    
    def __init__(self, obs_dim=2, action_dim=2, lr=3e-4, gamma=0.99, 
                 tau=0.005, alpha=0.2, auto_alpha=True, device='cpu',
                 hidden_dim=256, exploration_noise=0.1, epsilon_start=0.5, epsilon_end=0.05, epsilon_decay=0.995):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient for target network
            alpha: Entropy regularization coefficient (if auto_alpha=False)
            auto_alpha: Whether to automatically tune alpha
            device: Device to run on
            hidden_dim: Hidden layer dimension
        """
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.auto_alpha = auto_alpha
        
        # Policy network (actor)
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Two Q-networks (critics) for double Q-learning
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        # Target Q-networks
        self.q1_target = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.q2_target = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        
        # Initialize target networks with same weights as main networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Entropy coefficient (alpha)
        if auto_alpha:
            # Learnable alpha - start with higher value for more exploration
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            # Initialize log_alpha to give higher initial alpha (more exploration)
            self.log_alpha = torch.tensor([math.log(alpha * 2.0)], requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            # Fixed alpha - use higher value for more exploration
            self.alpha = torch.tensor(alpha * 1.5, device=device)
            self.log_alpha = None
            self.alpha_optimizer = None
        
        # Exploration parameters
        self.exploration_noise = exploration_noise  # Action noise for exploration
        self.epsilon = epsilon_start  # Epsilon-greedy exploration
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
    
    def load_checkpoint(self, policy_path, q1_path=None, q2_path=None):
        """
        Load checkpoint from saved model files.
        
        Args:
            policy_path: Path to saved policy network state dict
            q1_path: Path to saved Q1 network state dict
            q2_path: Path to saved Q2 network state dict
        """
        try:
            self.policy.load_state_dict(torch.load(policy_path, map_location=self.device))
            print(f"Loaded policy checkpoint from {policy_path}")
            
            if q1_path is not None:
                self.q1.load_state_dict(torch.load(q1_path, map_location=self.device))
                self.q1_target.load_state_dict(self.q1.state_dict())
                print(f"Loaded Q1 checkpoint from {q1_path}")
            
            if q2_path is not None:
                self.q2.load_state_dict(torch.load(q2_path, map_location=self.device))
                self.q2_target.load_state_dict(self.q2.state_dict())
                print(f"Loaded Q2 checkpoint from {q2_path}")
        except FileNotFoundError as e:
            print(f"Error loading checkpoint: {e}")
            raise
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
    
    def select_action(self, state, deterministic=False, apply_exploration=True):
        """Select an action using the current policy with exploration."""
        # Check for NaN in observation
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"Warning: Invalid observation detected: {state}, using zeros")
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy exploration: random action with probability epsilon
        if apply_exploration and not deterministic and np.random.random() < self.epsilon:
            # Random forward-only action in [0, 1]
            action = np.random.uniform(0.0, 1.0, size=2)
            action = torch.FloatTensor(action).unsqueeze(0)
        else:
            if deterministic:
                with torch.no_grad():
                    action = self.policy.deterministic_action(state_tensor)
            else:
                with torch.no_grad():
                    action, _ = self.policy.sample(state_tensor)
            
            # Add exploration noise (even when using policy) then clip to [0, 1]
            if apply_exploration and not deterministic:
                noise = torch.randn_like(action) * self.exploration_noise
                action = action + noise
                action = torch.clamp(action, 0.0, 1.0)
        
        # Clip action to valid forward-only range
        action = torch.clamp(action, 0.0, 1.0)
        
        return action.cpu().numpy().flatten()
    
    def decay_epsilon(self):
        """Decay epsilon for epsilon-greedy exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, batch_size=256):
        """Update networks using SAC algorithm."""
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0, 0.0, 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update Q-networks
        with torch.no_grad():
            # Sample next actions from policy (already mapped to [0, 1])
            next_actions, next_log_probs = self.policy.sample(next_states)
            
            # Compute target Q-values using target networks
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            
            # Compute target
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        # Update Q1
        q1_pred = self.q1(states, actions)
        q1_loss = F.mse_loss(q1_pred, q_target)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=1.0)
        self.q1_optimizer.step()
        
        # Update Q2
        q2_pred = self.q2(states, actions)
        q2_loss = F.mse_loss(q2_pred, q_target)
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=1.0)
        self.q2_optimizer.step()
        
        # Update policy
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        # Update alpha (if auto-tuning)
        alpha_loss = None
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        self._soft_update(self.q1_target, self.q1, self.tau)
        self._soft_update(self.q2_target, self.q2, self.tau)
        
        # Check for NaN
        if torch.isnan(q1_loss) or torch.isnan(q2_loss) or torch.isnan(policy_loss):
            print("Warning: NaN detected in losses")
            return 0.0, 0.0, 0.0, 0.0
        
        alpha_val = self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
        alpha_loss_val = alpha_loss.item() if alpha_loss is not None else 0.0
        
        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss_val
    
    def _soft_update(self, target, source, tau):
        """Soft update target network using polyak averaging."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def train_sac(num_episodes=1500, max_steps_per_episode=1000, 
              batch_size=256, update_freq=1, save_freq=100,
              policy_checkpoint=None, q1_checkpoint=None, q2_checkpoint=None, start_episode=0):
    """
    Train SAC agent on Duckiematrix environment.
    
    Args:
        num_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        batch_size: Batch size for updates
        update_freq: Update frequency (update every N steps)
        save_freq: Frequency to save model
        policy_checkpoint: Path to policy checkpoint to load (for resuming training)
        q1_checkpoint: Path to Q1 checkpoint to load (for resuming training)
        q2_checkpoint: Path to Q2 checkpoint to load (for resuming training)
        start_episode: Starting episode number (for resuming training, affects save naming)
    """
    
    # Create environment
    env = DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0", include_curve_flag=True)
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        device=device,
    )
    
    # Load checkpoint if provided
    if policy_checkpoint is not None:
        agent.load_checkpoint(policy_checkpoint, q1_checkpoint, q2_checkpoint)
        print(f"Resuming training from episode {start_episode}")
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    
    print("Starting SAC training...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Batch size: {batch_size}, Update frequency: {update_freq}")
    print(f"Auto-tuning alpha: {agent.auto_alpha}")
    
    reset_tile = None  # Track tile for reset
    total_steps = 0
    warmup_steps = 1000  # Collect some experience before updating
    
    for episode in range(start_episode, start_episode + num_episodes):
        # Reset environment (to closest tile if previous episode terminated)
        if reset_tile is not None:
            obs, info = env.reset(tile=reset_tile)
            reset_tile = None
        else:
            obs, info = env.reset()
        
        episode_reward = 0
        episode_length = 0
        last_pose = None
        
        for step in range(max_steps_per_episode):
            # Select action
            if total_steps < warmup_steps:
                # Random forward-only action during warmup
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, deterministic=False, apply_exploration=True)
                # Decay epsilon periodically
                if total_steps % 100 == 0:
                    agent.decay_epsilon()
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store transition in replay buffer
            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Track last pose for reset
            if not terminated:
                last_pose = info.get("pose")
            
            # Update networks
            if total_steps >= warmup_steps and total_steps % update_freq == 0:
                q1_loss, q2_loss, policy_loss, alpha_loss = agent.update(batch_size)
            
            # Check if episode is done
            if done:
                # Determine reset tile for next episode if terminated
                if terminated:
                    terminated_pos = info.get("terminated_position")
                    if terminated_pos is None and last_pose is not None:
                        terminated_pos = (last_pose["position"]["x"], last_pose["position"]["y"], 0.0)
                    
                    if terminated_pos is not None:
                        x, y, _ = terminated_pos
                        reset_tile = get_closest_tile(x, y)
                
                break
            
            obs = next_obs
            sleep(0.01)  # Small delay to prevent overwhelming the simulator
        
        # Store statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Print progress
        episode_num = episode + 1
        if episode_num % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            buffer_size = len(agent.replay_buffer)
            alpha_val = agent.alpha.item() if isinstance(agent.alpha, torch.Tensor) else agent.alpha
            print(f"Episode {episode_num}/{start_episode + num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Buffer Size: {buffer_size} | "
                  f"Alpha: {alpha_val:.4f}")
        
        # Save model periodically
        if episode_num % save_freq == 0:
            torch.save(agent.policy.state_dict(), f"sac_policy_ep{episode_num}.pth")
            torch.save(agent.q1.state_dict(), f"sac_q1_ep{episode_num}.pth")
            torch.save(agent.q2.state_dict(), f"sac_q2_ep{episode_num}.pth")
            print(f"Saved model at episode {episode_num}")
    
    # Final save
    torch.save(agent.policy.state_dict(), "sac_policy_final.pth")
    torch.save(agent.q1.state_dict(), "sac_q1_final.pth")
    torch.save(agent.q2.state_dict(), "sac_q2_final.pth")
    print("Training complete! Model saved to sac_*_final.pth")
    
    # Cleanup
    env.robot.camera.stop()
    env.robot.motors.stop()
    
    return agent, episode_rewards, episode_lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAC agent on Duckiematrix environment')
    parser.add_argument('--num_episodes', type=int, default=1500,
                        help='Number of episodes to train (default: 1500)')
    parser.add_argument('--max_steps_per_episode', type=int, default=2000,
                        help='Maximum steps per episode (default: 2000)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for updates (default: 256)')
    parser.add_argument('--update_freq', type=int, default=1,
                        help='Update frequency (update every N steps) (default: 1)')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Frequency to save model (default: 50)')
    parser.add_argument('--policy_checkpoint', type=str, default=None,
                        help='Path to policy checkpoint to load (for resuming training)')
    parser.add_argument('--q1_checkpoint', type=str, default=None,
                        help='Path to Q1 checkpoint to load (for resuming training)')
    parser.add_argument('--q2_checkpoint', type=str, default=None,
                        help='Path to Q2 checkpoint to load (for resuming training)')
    parser.add_argument('--start_episode', type=int, default=0,
                        help='Starting episode number (for resuming training, affects save naming)')
    
    args = parser.parse_args()
    
    # Train the agent
    agent, rewards, lengths = train_sac(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        batch_size=args.batch_size,
        update_freq=args.update_freq,
        save_freq=args.save_freq,
        policy_checkpoint=args.policy_checkpoint,
        q1_checkpoint=args.q1_checkpoint,
        q2_checkpoint=args.q2_checkpoint,
        start_episode=args.start_episode
    )

