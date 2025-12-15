"""
Compare sample efficiency of REINFORCE, PPO, and SAC algorithms.

This version runs until each algorithm collects 5000 samples and tracks
rewards per sample for fair comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import sys

# Add src directory and project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'src' else script_dir

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the training functions
from reinforce_agent import REINFORCEAgent
from ppo_agent import PPOAgent
from sac_agent import SACAgent
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from duckietown.sdk.utils.loop_lane_position import get_closest_tile
from time import sleep
import torch


def train_until_samples(agent_class, agent_kwargs, env, target_samples=5000, max_episodes=1000, max_steps_per_episode=2000):
    """
    Train an agent until it collects target_samples, tracking rewards per sample.
    
    Returns:
        sample_rewards: List of rewards, one per sample/step
        sample_numbers: List of sample numbers (0, 1, 2, ...)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = agent_class(device=device, **agent_kwargs)
    
    sample_rewards = []
    sample_numbers = []
    total_samples = 0
    
    episode = 0
    
    while total_samples < target_samples and episode < max_episodes:
        # Reset environment
        obs, info = env.reset(curve_prob=0.6)
        
        if hasattr(agent, 'reset_episode'):
            agent.reset_episode()
        elif hasattr(agent, 'reset_batch'):
            agent.reset_batch()
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps_per_episode):
            # Select action
            if hasattr(agent, 'select_action'):
                action = agent.select_action(obs, apply_exploration=True)
            else:
                action = agent.select_action(obs, deterministic=False, apply_exploration=True)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store reward for this sample
            sample_rewards.append(reward)
            sample_numbers.append(total_samples)
            total_samples += 1
            
            # Store transition based on agent type
            done = terminated or truncated
            if agent_class.__name__ == 'REINFORCEAgent':
                agent.store_reward(reward)
            elif agent_class.__name__ == 'PPOAgent':
                agent.store_transition(reward, done)
            elif agent_class.__name__ == 'SACAgent':
                agent.store_transition(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            episode_length += 1
            
            # Update agent periodically (except REINFORCE which updates at end of episode)
            if agent_class.__name__ == 'SACAgent':
                # SAC updates every step after warmup
                if total_samples >= 1000 and len(agent.replay_buffer) > 256:
                    agent.update(256)
            
            if terminated or truncated:
                break
            
            obs = next_obs
            sleep(0.01)  # Small delay
        
        # Update agent at end of episode (for REINFORCE)
        if agent_class.__name__ == 'REINFORCEAgent':
            agent.update()
        elif agent_class.__name__ == 'PPOAgent':
            if len(agent.batch_obs) > 0:
                agent.update()
        
        episode += 1
        
        if episode % 10 == 0:
            print(f"  Episode {episode}, Samples: {total_samples}/{target_samples}, Last episode reward: {episode_reward:.2f}")
        
        if total_samples >= target_samples:
            break
    
    # Truncate to exactly target_samples
    sample_rewards = sample_rewards[:target_samples]
    sample_numbers = list(range(target_samples))
    
    return sample_rewards, sample_numbers, episode


def train_all_algorithms(target_samples=5000, output_dir="sample_efficiency_comparison"):
    """
    Train REINFORCE, PPO, and SAC until each collects target_samples.
    
    Args:
        target_samples: Number of samples to collect for each algorithm
        output_dir: Directory to save results and plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    print("=" * 80)
    print("SAMPLE EFFICIENCY COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"Training each algorithm until {target_samples} samples collected")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Train REINFORCE
    print("\n" + "=" * 80)
    print("TRAINING REINFORCE")
    print("=" * 80)
    try:
        env = DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0")
        reinforce_rewards, reinforce_samples, reinforce_episodes = train_until_samples(
            REINFORCEAgent,
            {'obs_dim': 2, 'action_dim': 2, 'lr': 3e-4, 'gamma': 0.99},
            env,
            target_samples=target_samples
        )
        
        results['REINFORCE'] = {
            'sample_rewards': reinforce_rewards,
            'sample_numbers': reinforce_samples,
            'total_samples': len(reinforce_rewards),
            'episodes': reinforce_episodes
        }
        
        print(f"\n✓ REINFORCE completed: {reinforce_episodes} episodes, {len(reinforce_rewards)} samples")
        
        # Cleanup
        try:
            env.robot.camera.stop()
        except:
            pass
        env.robot.motors.stop()
        
    except Exception as e:
        print(f"\n✗ REINFORCE failed: {e}")
        import traceback
        traceback.print_exc()
        results['REINFORCE'] = None
    
    # Train PPO
    print("\n" + "=" * 80)
    print("TRAINING PPO")
    print("=" * 80)
    try:
        env = DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0")
        ppo_rewards, ppo_samples, ppo_episodes = train_until_samples(
            PPOAgent,
            {'obs_dim': 2, 'action_dim': 2, 'lr': 3e-4, 'gamma': 0.99, 'eps_clip': 0.2, 'k_epochs': 4, 'use_value': True},
            env,
            target_samples=target_samples
        )
        
        results['PPO'] = {
            'sample_rewards': ppo_rewards,
            'sample_numbers': ppo_samples,
            'total_samples': len(ppo_rewards),
            'episodes': ppo_episodes
        }
        
        print(f"\n✓ PPO completed: {ppo_episodes} episodes, {len(ppo_rewards)} samples")
        
        # Cleanup
        try:
            env.robot.camera.stop()
        except:
            pass
        env.robot.motors.stop()
        
    except Exception as e:
        print(f"\n✗ PPO failed: {e}")
        import traceback
        traceback.print_exc()
        results['PPO'] = None
    
    # Train SAC
    print("\n" + "=" * 80)
    print("TRAINING SAC")
    print("=" * 80)
    try:
        env = DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0", include_curve_flag=True)
        obs_dim = int(np.prod(env.observation_space.shape))
        sac_rewards, sac_samples, sac_episodes = train_until_samples(
            SACAgent,
            {'obs_dim': obs_dim, 'action_dim': 2, 'lr': 3e-4, 'gamma': 0.99, 'tau': 0.005, 'alpha': 0.2, 'auto_alpha': True},
            env,
            target_samples=target_samples
        )
        
        results['SAC'] = {
            'sample_rewards': sac_rewards,
            'sample_numbers': sac_samples,
            'total_samples': len(sac_rewards),
            'episodes': sac_episodes
        }
        
        print(f"\n✓ SAC completed: {sac_episodes} episodes, {len(sac_rewards)} samples")
        
        # Cleanup
        try:
            env.robot.camera.stop()
        except:
            pass
        env.robot.motors.stop()
        
    except Exception as e:
        print(f"\n✗ SAC failed: {e}")
        import traceback
        traceback.print_exc()
        results['SAC'] = None
    
    # Save results to JSON
    results_file = os.path.join(output_dir, "sample_efficiency_results.json")
    with open(results_file, 'w') as f:
        json_results = {}
        for algo_name, algo_results in results.items():
            if algo_results is not None:
                json_results[algo_name] = algo_results
            else:
                json_results[algo_name] = None
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


def plot_sample_efficiency(results, output_dir="sample_efficiency_comparison"):
    """
    Generate comparison plots showing reward vs sample number.
    
    Args:
        results: Dictionary containing results from all algorithms
        output_dir: Directory to save plots
    """
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Sample Efficiency Comparison: Reward per Sample', fontsize=16, fontweight='bold')
    
    # Define colors
    colors = {
        'REINFORCE': '#1f77b4',  # Blue
        'PPO': '#ff7f0e',        # Orange
        'SAC': '#2ca02c'          # Green
    }
    
    # Plot 1: Raw reward per sample
    ax1 = axes[0]
    for algo_name, algo_results in results.items():
        if algo_results is not None and 'sample_rewards' in algo_results:
            samples = np.array(algo_results['sample_numbers'])
            rewards = np.array(algo_results['sample_rewards'])
            
            # Plot raw rewards (light, transparent)
            ax1.plot(samples, rewards, alpha=0.2, color=colors[algo_name], linewidth=0.5, label=f'{algo_name} (raw)')
            
            # Plot moving average (bold)
            window = 100  # Average over 100 samples
            if len(rewards) >= window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                moving_samples = samples[window-1:]
                ax1.plot(moving_samples, moving_avg, color=colors[algo_name], 
                        linewidth=2.5, label=f'{algo_name} (100-sample avg)', alpha=0.9)
    
    ax1.set_xlabel('Sample Number', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Reward per Sample (with 100-sample moving average)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5000)
    
    # Plot 2: Cumulative average reward
    ax2 = axes[1]
    for algo_name, algo_results in results.items():
        if algo_results is not None and 'sample_rewards' in algo_results:
            samples = np.array(algo_results['sample_numbers'])
            rewards = np.array(algo_results['sample_rewards'])
            
            # Compute cumulative average
            cumulative_avg = np.cumsum(rewards) / (samples + 1)
            
            ax2.plot(samples, cumulative_avg, color=colors[algo_name], 
                    linewidth=2.5, label=algo_name, alpha=0.9)
    
    ax2.set_xlabel('Sample Number', fontsize=12)
    ax2.set_ylabel('Cumulative Average Reward', fontsize=12)
    ax2.set_title('Cumulative Average Reward vs Sample Number', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 5000)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, "sample_efficiency_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_file}")
    
    plot_file_pdf = os.path.join(output_dir, "sample_efficiency_comparison.pdf")
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_file_pdf}")
    
    plt.close()
    
    # Create summary plot (focused on moving average)
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for algo_name, algo_results in results.items():
        if algo_results is not None and 'sample_rewards' in algo_results:
            samples = np.array(algo_results['sample_numbers'])
            rewards = np.array(algo_results['sample_rewards'])
            
            # Plot moving average
            window = 100
            if len(rewards) >= window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                moving_samples = samples[window-1:]
                ax.plot(moving_samples, moving_avg, color=colors[algo_name], 
                       linewidth=3, label=algo_name, alpha=0.9)
    
    ax.set_xlabel('Sample Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward (100-sample window)', fontsize=14, fontweight='bold')
    ax.set_title('Sample Efficiency Comparison: REINFORCE vs PPO vs SAC', 
                fontsize=16, fontweight='bold')
    ax.set_xlim(0, 5000)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    summary_file = os.path.join(output_dir, "sample_efficiency_summary.png")
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    print(f"✓ Summary plot saved to {summary_file}")
    
    summary_file_pdf = os.path.join(output_dir, "sample_efficiency_summary.pdf")
    plt.savefig(summary_file_pdf, bbox_inches='tight')
    print(f"✓ Summary plot saved to {summary_file_pdf}")
    
    plt.close()


def print_summary(results):
    """Print a summary of the results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for algo_name, algo_results in results.items():
        if algo_results is not None:
            rewards = np.array(algo_results['sample_rewards'])
            print(f"\n{algo_name}:")
            print(f"  Total Samples: {len(rewards):,}")
            print(f"  Episodes: {algo_results.get('episodes', 'N/A')}")
            print(f"  Average Reward (all samples): {np.mean(rewards):.3f}")
            print(f"  Average Reward (first 500): {np.mean(rewards[:500]):.3f}")
            print(f"  Average Reward (last 500): {np.mean(rewards[-500:]):.3f}")
            print(f"  Best 100-sample average: {np.max([np.mean(rewards[i:i+100]) for i in range(len(rewards)-99)]):.3f}")
        else:
            print(f"\n{algo_name}: FAILED")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare sample efficiency of REINFORCE, PPO, and SAC')
    parser.add_argument('--target_samples', type=int, default=5000,
                       help='Number of samples to collect for each algorithm (default: 5000)')
    parser.add_argument('--output_dir', type=str, default='sample_efficiency_comparison',
                       help='Directory to save results and plots (default: sample_efficiency_comparison)')
    
    args = parser.parse_args()
    
    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    # Train all algorithms
    results = train_all_algorithms(
        target_samples=args.target_samples,
        output_dir=output_dir
    )
    
    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    plot_sample_efficiency(results, output_dir=output_dir)
    
    # Print summary
    print_summary(results)
    
    print(f"\n✓ All results saved to: {output_dir}/")
    print("=" * 80)

