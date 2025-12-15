"""
Training metrics tracking and plotting utilities.
Tracks and saves training metrics for later analysis and visualization.
"""

import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingMetrics:
    """Track and save training metrics during training."""
    
    def __init__(self, save_dir: str = "training_logs"):
        """
        Initialize metrics tracker.
        
        Args:
            save_dir: Directory to save metrics and plots
        """
        self.save_dir = Path(save_dir)
        self.metrics_dir = self.save_dir / "metrics"
        self.plots_dir = self.save_dir / "plots"
        self.checkpoints_dir = self.save_dir / "checkpoints"
        
        # Create directories
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Training time tracking
        self.training_start_time = None
        self.training_end_time = None
        self.episode_start_times = []
        self.episode_end_times = []
        self.step_times = []
        
        # Episode metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = defaultdict(list)  # q1_loss, q2_loss, policy_loss, alpha_loss
        
        # Step-level metrics
        self.step_rewards = []
        self.step_q1_losses = []
        self.step_q2_losses = []
        self.step_policy_losses = []
        self.step_alpha_losses = []
        self.step_alpha_values = []
        self.step_buffer_sizes = []
        
        # Windowed averages (for smoothing)
        self.window_size = 100
        self.avg_rewards_windowed = []
        self.avg_lengths_windowed = []
        
        # Training configuration (to be saved)
        self.config = {}
        
    def start_training(self, config: Dict):
        """Mark the start of training and save configuration."""
        self.training_start_time = time.time()
        self.config = config
        print(f"Training started. Metrics will be saved to: {self.save_dir}")
        
    def end_training(self):
        """Mark the end of training."""
        self.training_end_time = time.time()
        
    def start_episode(self, episode_num: int):
        """Mark the start of an episode."""
        self.episode_start_times.append(time.time())
        
    def end_episode(self, episode_num: int, reward: float, length: int, 
                    losses: Optional[Dict[str, float]] = None):
        """
        Record episode metrics.
        
        Args:
            episode_num: Episode number
            reward: Total episode reward
            length: Episode length (number of steps)
            losses: Dictionary of losses from last update (optional)
        """
        self.episode_end_times.append(time.time())
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if losses:
            for key, value in losses.items():
                self.episode_losses[key].append(value)
        
        # Calculate windowed averages
        if len(self.episode_rewards) >= self.window_size:
            window_rewards = self.episode_rewards[-self.window_size:]
            window_lengths = self.episode_lengths[-self.window_size:]
            self.avg_rewards_windowed.append(np.mean(window_rewards))
            self.avg_lengths_windowed.append(np.mean(window_lengths))
        else:
            # Use all available data if window not full yet
            if len(self.episode_rewards) > 0:
                self.avg_rewards_windowed.append(np.mean(self.episode_rewards))
                self.avg_lengths_windowed.append(np.mean(self.episode_lengths))
    
    def record_step_metrics(self, reward: float, q1_loss: Optional[float] = None,
                           q2_loss: Optional[float] = None, policy_loss: Optional[float] = None,
                           alpha_loss: Optional[float] = None, alpha_value: Optional[float] = None,
                           buffer_size: Optional[int] = None, step_time: Optional[float] = None):
        """
        Record step-level metrics.
        
        Args:
            reward: Step reward
            q1_loss: Q1 network loss (optional)
            q2_loss: Q2 network loss (optional)
            policy_loss: Policy network loss (optional)
            alpha_loss: Alpha (temperature) loss (optional)
            alpha_value: Current alpha value (optional)
            buffer_size: Current replay buffer size (optional)
            step_time: Time taken for this step (optional)
        """
        self.step_rewards.append(reward)
        if q1_loss is not None:
            self.step_q1_losses.append(q1_loss)
        if q2_loss is not None:
            self.step_q2_losses.append(q2_loss)
        if policy_loss is not None:
            self.step_policy_losses.append(policy_loss)
        if alpha_loss is not None:
            self.step_alpha_losses.append(alpha_loss)
        if alpha_value is not None:
            self.step_alpha_values.append(alpha_value)
        if buffer_size is not None:
            self.step_buffer_sizes.append(buffer_size)
        if step_time is not None:
            self.step_times.append(step_time)
    
    def save_metrics(self, filename: str = "training_metrics.json"):
        """Save all metrics to JSON file."""
        metrics = {
            "config": self.config,
            "training_time": {
                "total_seconds": self.training_end_time - self.training_start_time if self.training_end_time else None,
                "total_hours": (self.training_end_time - self.training_start_time) / 3600 if self.training_end_time else None,
                "start_time": self.training_start_time,
                "end_time": self.training_end_time,
            },
            "episode_metrics": {
                "rewards": self.episode_rewards,
                "lengths": self.episode_lengths,
                "avg_rewards_windowed": self.avg_rewards_windowed,
                "avg_lengths_windowed": self.avg_lengths_windowed,
                "episode_times": [
                    end - start for start, end in zip(self.episode_start_times, self.episode_end_times)
                ] if len(self.episode_start_times) == len(self.episode_end_times) else [],
            },
            "step_metrics": {
                "rewards": self.step_rewards,
                "q1_losses": self.step_q1_losses,
                "q2_losses": self.step_q2_losses,
                "policy_losses": self.step_policy_losses,
                "alpha_losses": self.step_alpha_losses,
                "alpha_values": self.step_alpha_values,
                "buffer_sizes": self.step_buffer_sizes,
                "step_times": self.step_times,
            },
            "statistics": self._compute_statistics(),
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_list(item) for item in obj]
            return obj
        
        metrics = convert_to_list(metrics)
        
        filepath = self.metrics_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to: {filepath}")
        return filepath
    
    def _compute_statistics(self) -> Dict:
        """Compute summary statistics."""
        stats = {}
        
        if self.episode_rewards:
            stats["episode_rewards"] = {
                "mean": float(np.mean(self.episode_rewards)),
                "std": float(np.std(self.episode_rewards)),
                "min": float(np.min(self.episode_rewards)),
                "max": float(np.max(self.episode_rewards)),
                "final_100_mean": float(np.mean(self.episode_rewards[-100:])) if len(self.episode_rewards) >= 100 else None,
            }
        
        if self.episode_lengths:
            stats["episode_lengths"] = {
                "mean": float(np.mean(self.episode_lengths)),
                "std": float(np.std(self.episode_lengths)),
                "min": int(np.min(self.episode_lengths)),
                "max": int(np.max(self.episode_lengths)),
            }
        
        if self.step_times:
            stats["step_times"] = {
                "mean_ms": float(np.mean(self.step_times) * 1000),
                "std_ms": float(np.std(self.step_times) * 1000),
                "min_ms": float(np.min(self.step_times) * 1000),
                "max_ms": float(np.max(self.step_times) * 1000),
            }
        
        if self.episode_start_times and self.episode_end_times and len(self.episode_start_times) == len(self.episode_end_times):
            episode_times = [end - start for start, end in zip(self.episode_start_times, self.episode_end_times)]
            stats["episode_times"] = {
                "mean_seconds": float(np.mean(episode_times)),
                "std_seconds": float(np.std(episode_times)),
                "total_seconds": float(np.sum(episode_times)),
            }
        
        return stats
    
    def plot_all(self, save_plots: bool = True, show_plots: bool = False):
        """Generate and save all plots."""
        print("Generating plots...")
        
        self.plot_rewards(save=save_plots, show=show_plots)
        self.plot_lengths(save=save_plots, show=show_plots)
        self.plot_losses(save=save_plots, show=show_plots)
        self.plot_alpha(save=save_plots, show=show_plots)
        self.plot_buffer_size(save=save_plots, show=show_plots)
        self.plot_training_time(save=save_plots, show=show_plots)
        
        print(f"Plots saved to: {self.plots_dir}")
    
    def plot_rewards(self, save: bool = True, show: bool = False):
        """Plot episode rewards."""
        if not self.episode_rewards:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Raw rewards
        episodes = range(1, len(self.episode_rewards) + 1)
        ax1.plot(episodes, self.episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        if self.avg_rewards_windowed:
            window_episodes = range(self.window_size, len(self.avg_rewards_windowed) + self.window_size)
            ax1.plot(window_episodes, self.avg_rewards_windowed, color='red', linewidth=2, label=f'Moving Average ({self.window_size})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribution
        ax2.hist(self.episode_rewards, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(np.mean(self.episode_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Reward Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'rewards.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_lengths(self, save: bool = True, show: bool = False):
        """Plot episode lengths."""
        if not self.episode_lengths:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        episodes = range(1, len(self.episode_lengths) + 1)
        ax.plot(episodes, self.episode_lengths, alpha=0.3, color='green', label='Episode Length')
        if self.avg_lengths_windowed:
            window_episodes = range(self.window_size, len(self.avg_lengths_windowed) + self.window_size)
            ax.plot(window_episodes, self.avg_lengths_windowed, color='red', linewidth=2, label=f'Moving Average ({self.window_size})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'lengths.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_losses(self, save: bool = True, show: bool = False):
        """Plot training losses."""
        if not self.step_q1_losses and not self.step_policy_losses:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        steps = range(len(self.step_q1_losses))
        
        # Q1 Loss
        if self.step_q1_losses:
            axes[0, 0].plot(steps, self.step_q1_losses, alpha=0.6, color='blue', label='Q1 Loss')
            axes[0, 0].set_xlabel('Update Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Q1 Network Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Q2 Loss
        if self.step_q2_losses:
            axes[0, 1].plot(steps, self.step_q2_losses, alpha=0.6, color='green', label='Q2 Loss')
            axes[0, 1].set_xlabel('Update Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Q2 Network Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Policy Loss
        if self.step_policy_losses:
            axes[1, 0].plot(steps, self.step_policy_losses, alpha=0.6, color='red', label='Policy Loss')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Policy Network Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Alpha Loss
        if self.step_alpha_losses:
            axes[1, 1].plot(steps, self.step_alpha_losses, alpha=0.6, color='purple', label='Alpha Loss')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Alpha (Temperature) Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'losses.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_alpha(self, save: bool = True, show: bool = False):
        """Plot alpha (temperature) value over time."""
        if not self.step_alpha_values:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        steps = range(len(self.step_alpha_values))
        ax.plot(steps, self.step_alpha_values, color='orange', linewidth=1.5)
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Alpha Value')
        ax.set_title('Alpha (Temperature) Value Over Time')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'alpha.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_buffer_size(self, save: bool = True, show: bool = False):
        """Plot replay buffer size over time."""
        if not self.step_buffer_sizes:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        steps = range(len(self.step_buffer_sizes))
        ax.plot(steps, self.step_buffer_sizes, color='teal', linewidth=1.5)
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Buffer Size')
        ax.set_title('Replay Buffer Size Over Time')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'buffer_size.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_training_time(self, save: bool = True, show: bool = False):
        """Plot training time metrics."""
        if not self.episode_start_times or not self.episode_end_times:
            return
        
        if len(self.episode_start_times) != len(self.episode_end_times):
            return
        
        episode_times = [end - start for start, end in zip(self.episode_start_times, self.episode_end_times)]
        cumulative_time = np.cumsum(episode_times)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Time per episode
        episodes = range(1, len(episode_times) + 1)
        ax1.plot(episodes, episode_times, alpha=0.6, color='blue', label='Time per Episode')
        ax1.axhline(np.mean(episode_times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(episode_times):.2f}s')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Time per Episode')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative time
        ax2.plot(episodes, cumulative_time, color='green', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative Time (seconds)')
        ax2.set_title('Cumulative Training Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'training_time.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()


def load_metrics(filepath: str) -> Dict:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_from_file(metrics_file: str, save_dir: Optional[str] = None, show: bool = False):
    """
    Load metrics from file and generate plots.
    
    Args:
        metrics_file: Path to metrics JSON file
        save_dir: Directory to save plots (default: same as metrics file)
        show: Whether to display plots
    """
    metrics = load_metrics(metrics_file)
    
    # Create a temporary metrics object to use plotting functions
    if save_dir is None:
        save_dir = str(Path(metrics_file).parent.parent)
    
    tracker = TrainingMetrics(save_dir=save_dir)
    
    # Restore data
    tracker.episode_rewards = metrics['episode_metrics']['rewards']
    tracker.episode_lengths = metrics['episode_metrics']['lengths']
    tracker.avg_rewards_windowed = metrics['episode_metrics']['avg_rewards_windowed']
    tracker.avg_lengths_windowed = metrics['episode_metrics']['avg_lengths_windowed']
    tracker.step_q1_losses = metrics['step_metrics']['q1_losses']
    tracker.step_q2_losses = metrics['step_metrics']['q2_losses']
    tracker.step_policy_losses = metrics['step_metrics']['policy_losses']
    tracker.step_alpha_losses = metrics['step_metrics']['alpha_losses']
    tracker.step_alpha_values = metrics['step_metrics']['alpha_values']
    tracker.step_buffer_sizes = metrics['step_metrics']['buffer_sizes']
    tracker.step_times = metrics['step_metrics']['step_times']
    
    # Generate plots
    tracker.plot_all(save_plots=True, show_plots=show)

