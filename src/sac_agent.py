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
from gym_duckiematrix.DB21J_gym import DuckiematrixDB21JEnvGym
from training_metrics import TrainingMetrics
from time import sleep
import math
import argparse
import random
import time


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
        import os
        
        # Handle relative paths - check current directory, src/ directory, and checkpoints/ directory
        def find_checkpoint(path):
            if os.path.exists(path):
                return path
            # Try in src/ directory if running from root
            src_path = os.path.join("src", path)
            if os.path.exists(src_path):
                return src_path
            # Try in checkpoints/ directory
            checkpoint_path = os.path.join("checkpoints", path)
            if os.path.exists(checkpoint_path):
                return checkpoint_path
            # Try just the filename in checkpoints/ (in case full path was provided)
            filename = os.path.basename(path)
            checkpoint_path = os.path.join("checkpoints", filename)
            if os.path.exists(checkpoint_path):
                return checkpoint_path
            return path  # Return original path to get proper error message
        
        try:
            policy_path = find_checkpoint(policy_path)
            self.policy.load_state_dict(torch.load(policy_path, map_location=self.device))
            print(f"Loaded policy checkpoint from {policy_path}")
            
            if q1_path is not None:
                q1_path = find_checkpoint(q1_path)
                self.q1.load_state_dict(torch.load(q1_path, map_location=self.device))
                self.q1_target.load_state_dict(self.q1.state_dict())
                print(f"Loaded Q1 checkpoint from {q1_path}")
            
            if q2_path is not None:
                q2_path = find_checkpoint(q2_path)
                self.q2.load_state_dict(torch.load(q2_path, map_location=self.device))
                self.q2_target.load_state_dict(self.q2.state_dict())
                print(f"Loaded Q2 checkpoint from {q2_path}")
        except FileNotFoundError as e:
            print(f"Error loading checkpoint: {e}")
            print(f"Current working directory: {os.getcwd()}")
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
              policy_checkpoint=None, q1_checkpoint=None, q2_checkpoint=None, start_episode=0,
              checkpoint_dir="checkpoints", use_gym_mode=False, step_duration=0.1,
              metrics_dir="training_logs", save_metrics=True, hyperparams=None, hyperparams_file=None):
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
        checkpoint_dir: Directory to save checkpoints (default: "checkpoints")
        use_gym_mode: Whether to use gym mode (faster, non-real-time) (default: False)
        step_duration: Step duration for gym mode in seconds (default: 0.1)
        metrics_dir: Directory to save training metrics (default: "training_logs")
        save_metrics: Whether to track and save training metrics (default: True)
        hyperparams: Dictionary of hyperparameters (overrides defaults)
        hyperparams_file: Path to JSON file containing hyperparameters (overrides defaults)
    """
    import os
    import json
    
    # Load hyperparameters from file if provided
    if hyperparams_file is not None:
        with open(hyperparams_file, 'r') as f:
            file_hyperparams = json.load(f)
            if hyperparams is None:
                hyperparams = file_hyperparams
            else:
                hyperparams.update(file_hyperparams)
    
    # Default hyperparameters
    default_hyperparams = {
        'lr': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'auto_alpha': True,
        'hidden_dim': 256,
        'exploration_noise': 0.1,
        'epsilon_start': 0.5,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.995,
    }
    
    # Merge with provided hyperparameters
    if hyperparams is not None:
        default_hyperparams.update(hyperparams)
    hyperparams = default_hyperparams
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}/")
    
    # Initialize metrics tracker
    metrics = None
    if save_metrics:
        metrics = TrainingMetrics(save_dir=metrics_dir)
        config = {
            "num_episodes": num_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "batch_size": batch_size,
            "update_freq": update_freq,
            "save_freq": save_freq,
            "start_episode": start_episode,
            "use_gym_mode": use_gym_mode,
            "step_duration": step_duration,
            "checkpoint_dir": checkpoint_dir,
            "hyperparams": hyperparams,
        }
        metrics.start_training(config)
    
    # Create environment (gym mode or regular mode)
    if use_gym_mode:
        print(f"Using GYM MODE (step_duration={step_duration}s)")
        env = DuckiematrixDB21JEnvGym(
            entity_name="map_0/vehicle_0", 
            include_curve_flag=True,
            step_duration=step_duration
        )
    else:
        print("Using REGULAR MODE (real-time)")
        env = DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0", include_curve_flag=True)
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    
    print(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")
    
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=hyperparams['lr'],
        gamma=hyperparams['gamma'],
        tau=hyperparams['tau'],
        alpha=hyperparams['alpha'],
        auto_alpha=hyperparams['auto_alpha'],
        device=device,
        hidden_dim=hyperparams['hidden_dim'],
        exploration_noise=hyperparams['exploration_noise'],
        epsilon_start=hyperparams['epsilon_start'],
        epsilon_end=hyperparams['epsilon_end'],
        epsilon_decay=hyperparams['epsilon_decay'],
    )
    
    # Load checkpoint if provided
    if policy_checkpoint is not None:
        print(f"Loading checkpoint: {policy_checkpoint}")
        agent.load_checkpoint(policy_checkpoint, q1_checkpoint, q2_checkpoint)
        print(f"Resuming training from episode {start_episode}")
        
        # Verify observation space compatibility
        loaded_policy_obs_dim = agent.policy.fc1.in_features
        if loaded_policy_obs_dim != obs_dim:
            print(f"WARNING: Observation dimension mismatch!")
            print(f"  Checkpoint expects obs_dim={loaded_policy_obs_dim}")
            print(f"  Environment provides obs_dim={obs_dim}")
            print(f"  This may cause errors. Make sure checkpoint matches environment configuration.")
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    
    print("Starting SAC training...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Batch size: {batch_size}, Update frequency: {update_freq}")
    print(f"Auto-tuning alpha: {agent.auto_alpha}")
    
    total_steps = 0
    warmup_steps = 1000  # Collect some experience before updating
    
    # Track losses for metrics
    last_q1_loss = None
    last_q2_loss = None
    last_policy_loss = None
    last_alpha_loss = None
    
    for episode in range(start_episode, start_episode + num_episodes):
        # Start episode tracking
        if metrics:
            metrics.start_episode(episode)
        
        # Always use random reset: 60% curved tiles, 40% straight tiles
        obs, info = env.reset(curve_prob=0.6)
        
        episode_reward = 0
        episode_length = 0
        last_pose = None
        
        for step in range(max_steps_per_episode):
            step_start_time = time.perf_counter()
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
            
            # Record step metrics (for every step)
            step_time = time.perf_counter() - step_start_time
            if metrics:
                alpha_val = agent.alpha.item() if isinstance(agent.alpha, torch.Tensor) else agent.alpha
                metrics.record_step_metrics(
                    reward=reward,
                    q1_loss=last_q1_loss,
                    q2_loss=last_q2_loss,
                    policy_loss=last_policy_loss,
                    alpha_loss=last_alpha_loss,
                    alpha_value=alpha_val,
                    buffer_size=len(agent.replay_buffer),
                    step_time=step_time
                )
            
            # Update networks
            if total_steps >= warmup_steps and total_steps % update_freq == 0:
                q1_loss, q2_loss, policy_loss, alpha_loss = agent.update(batch_size)
                last_q1_loss = q1_loss
                last_q2_loss = q2_loss
                last_policy_loss = policy_loss
                last_alpha_loss = alpha_loss
            
            # Check if episode is done
            if done:
                break
            
            obs = next_obs
            # Only sleep in regular mode (gym mode handles timing internally)
            if not use_gym_mode:
                sleep(0.01)  # Small delay to prevent overwhelming the simulator
        
        # Store statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Record episode metrics
        episode_num = episode + 1
        if metrics:
            losses = {}
            if last_q1_loss is not None:
                losses['q1_loss'] = last_q1_loss
            if last_q2_loss is not None:
                losses['q2_loss'] = last_q2_loss
            if last_policy_loss is not None:
                losses['policy_loss'] = last_policy_loss
            if last_alpha_loss is not None:
                losses['alpha_loss'] = last_alpha_loss
            
            metrics.end_episode(episode_num, episode_reward, episode_length, losses)
        
        # Print progress every 10 episodes
        if episode_num % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-10:]) if len(episode_lengths) >= 10 else np.mean(episode_lengths)
            buffer_size = len(agent.replay_buffer)
            alpha_val = agent.alpha.item() if isinstance(agent.alpha, torch.Tensor) else agent.alpha
            print(f"Episode {episode_num}/{start_episode + num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward (last 10): {avg_reward:.2f} | "
                  f"Length: {episode_length} | "
                  f"Avg Length (last 10): {avg_length:.1f} | "
                  f"Buffer Size: {buffer_size} | "
                  f"Alpha: {alpha_val:.4f}")
            import sys
            sys.stdout.flush()  # Ensure output is printed immediately
        
        # Save model periodically
        if episode_num % save_freq == 0:
            policy_path = os.path.join(checkpoint_dir, f"sac_policy_ep{episode_num}.pth")
            q1_path = os.path.join(checkpoint_dir, f"sac_q1_ep{episode_num}.pth")
            q2_path = os.path.join(checkpoint_dir, f"sac_q2_ep{episode_num}.pth")
            torch.save(agent.policy.state_dict(), policy_path)
            torch.save(agent.q1.state_dict(), q1_path)
            torch.save(agent.q2.state_dict(), q2_path)
            print(f"Saved model at episode {episode_num} to {checkpoint_dir}/")
    
    # Final save
    policy_path = os.path.join(checkpoint_dir, "sac_policy_final.pth")
    q1_path = os.path.join(checkpoint_dir, "sac_q1_final.pth")
    q2_path = os.path.join(checkpoint_dir, "sac_q2_final.pth")
    torch.save(agent.policy.state_dict(), policy_path)
    torch.save(agent.q1.state_dict(), q1_path)
    torch.save(agent.q2.state_dict(), q2_path)
    print(f"Training complete! Model saved to {checkpoint_dir}/sac_*_final.pth")
    
    # Save and plot metrics
    if metrics:
        metrics.end_training()
        metrics.save_metrics()
        print("Generating training plots...")
        metrics.plot_all(save_plots=True, show_plots=False)
        print(f"Training metrics and plots saved to: {metrics_dir}/")
    
    # Cleanup
    try:
        env.robot.camera.stop()
    except:
        pass  # Camera may not be started
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
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--gym_mode', action='store_true',
                        help='Use gym mode (faster, non-real-time simulation)')
    parser.add_argument('--step_duration', type=float, default=0.1,
                        help='Step duration for gym mode in seconds (default: 0.1)')
    parser.add_argument('--metrics_dir', type=str, default='training_logs',
                        help='Directory to save training metrics (default: training_logs)')
    parser.add_argument('--no_metrics', action='store_true',
                        help='Disable metrics tracking and plotting')
    parser.add_argument('--hyperparams_file', type=str, default=None,
                        help='Path to JSON file containing hyperparameters (default: None)')
    
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
        start_episode=args.start_episode,
        checkpoint_dir=args.checkpoint_dir,
        use_gym_mode=args.gym_mode,
        step_duration=args.step_duration,
        metrics_dir=args.metrics_dir,
        save_metrics=not args.no_metrics,
        hyperparams_file=args.hyperparams_file
    )
