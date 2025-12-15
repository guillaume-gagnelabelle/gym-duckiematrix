"""
Compare sample efficiency of REINFORCE, PPO, and SAC algorithms.

This script trains each algorithm for 100 episodes and generates comparison plots
showing sample efficiency (reward vs total samples used).
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
from reinforce_agent import train_reinforce
from ppo_agent import train_ppo
from sac_agent import train_sac


def train_all_algorithms(num_episodes=150, max_steps_per_episode=2000, output_dir="sample_efficiency_comparison"):
    """
    Train REINFORCE, PPO, and SAC for the same number of episodes and collect metrics.
    
    Args:
        num_episodes: Number of episodes to train each algorithm
        max_steps_per_episode: Maximum steps per episode
        output_dir: Directory to save results and plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    print("=" * 80)
    print("SAMPLE EFFICIENCY COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"Training each algorithm for {num_episodes} episodes")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Train REINFORCE
    print("\n" + "=" * 80)
    print("TRAINING REINFORCE")
    print("=" * 80)
    try:
        reinforce_agent, reinforce_rewards, reinforce_lengths = train_reinforce(
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            save_freq=1000  # Don't save during this experiment
        )
        
        # Calculate cumulative samples
        reinforce_samples = np.cumsum(reinforce_lengths)
        
        results['REINFORCE'] = {
            'episode_rewards': reinforce_rewards,
            'episode_lengths': reinforce_lengths,
            'cumulative_samples': reinforce_samples.tolist(),
            'total_samples': int(reinforce_samples[-1]) if len(reinforce_samples) > 0 else 0,
            'final_avg_reward': np.mean(reinforce_rewards[-10:]) if len(reinforce_rewards) >= 10 else np.mean(reinforce_rewards) if len(reinforce_rewards) > 0 else 0
        }
        
        print(f"\n✓ REINFORCE completed: {len(reinforce_rewards)} episodes, {results['REINFORCE']['total_samples']} total samples")
        
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
        ppo_agent, ppo_rewards, ppo_lengths = train_ppo(
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            batch_size=2048,
            save_freq=1000,  # Don't save during this experiment
            use_value=True
        )
        
        # Calculate cumulative samples
        ppo_samples = np.cumsum(ppo_lengths)
        
        results['PPO'] = {
            'episode_rewards': ppo_rewards,
            'episode_lengths': ppo_lengths,
            'cumulative_samples': ppo_samples.tolist(),
            'total_samples': int(ppo_samples[-1]) if len(ppo_samples) > 0 else 0,
            'final_avg_reward': np.mean(ppo_rewards[-10:]) if len(ppo_rewards) >= 10 else np.mean(ppo_rewards) if len(ppo_rewards) > 0 else 0
        }
        
        print(f"\n✓ PPO completed: {len(ppo_rewards)} episodes, {results['PPO']['total_samples']} total samples")
        
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
        sac_agent, sac_rewards, sac_lengths = train_sac(
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            batch_size=256,
            update_freq=1,
            save_freq=1000,  # Don't save during this experiment
            use_gym_mode=False,  # Use real-time mode as requested
            save_metrics=False  # We'll track metrics ourselves
        )
        
        # Calculate cumulative samples
        sac_samples = np.cumsum(sac_lengths)
        
        results['SAC'] = {
            'episode_rewards': sac_rewards,
            'episode_lengths': sac_lengths,
            'cumulative_samples': sac_samples.tolist(),
            'total_samples': int(sac_samples[-1]) if len(sac_samples) > 0 else 0,
            'final_avg_reward': np.mean(sac_rewards[-10:]) if len(sac_rewards) >= 10 else np.mean(sac_rewards) if len(sac_rewards) > 0 else 0
        }
        
        print(f"\n✓ SAC completed: {len(sac_rewards)} episodes, {results['SAC']['total_samples']} total samples")
        
    except Exception as e:
        print(f"\n✗ SAC failed: {e}")
        import traceback
        traceback.print_exc()
        results['SAC'] = None
    
    # Save results to JSON
    results_file = os.path.join(output_dir, "sample_efficiency_results.json")
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
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
    Generate comparison plots for sample efficiency.
    
    Args:
        results: Dictionary containing results from all algorithms
        output_dir: Directory to save plots
    """
    # Find minimum total samples (truncate to SAC's sample count for fair comparison)
    min_samples = None
    for algo_name, algo_results in results.items():
        if algo_results is not None and 'total_samples' in algo_results:
            total = algo_results['total_samples']
            if min_samples is None or total < min_samples:
                min_samples = total
    
    if min_samples is None:
        min_samples = float('inf')
    
    print(f"\nTruncating all algorithms to {min_samples:,} samples (SAC's total) for fair comparison")
    
    # Truncate all algorithms' data to min_samples
    # We need to interpolate or find episodes that fit within the sample budget
    truncated_results = {}
    for algo_name, algo_results in results.items():
        if algo_results is not None and len(algo_results['cumulative_samples']) > 0:
            samples = np.array(algo_results['cumulative_samples'])
            rewards = np.array(algo_results['episode_rewards'])
            lengths = np.array(algo_results['episode_lengths'])
            
            # Find the last episode index where cumulative samples <= min_samples
            # Always include the first episode (index 0)
            valid_indices = np.where(samples <= min_samples)[0]
            if len(valid_indices) > 0:
                # Always include first episode
                if valid_indices[0] != 0:
                    valid_indices = np.concatenate([[0], valid_indices])
                # Remove duplicates and sort
                valid_indices = np.unique(valid_indices)
                
                truncated_results[algo_name] = {
                    'episode_rewards': rewards[valid_indices].tolist(),
                    'episode_lengths': lengths[valid_indices].tolist(),
                    'cumulative_samples': samples[valid_indices].tolist(),
                    'total_samples': int(samples[valid_indices][-1]) if len(valid_indices) > 0 else 0
                }
            else:
                # If no episodes fit, at least include the first one
                truncated_results[algo_name] = {
                    'episode_rewards': [rewards[0]],
                    'episode_lengths': [lengths[0]],
                    'cumulative_samples': [samples[0]],
                    'total_samples': int(samples[0])
                }
        else:
            truncated_results[algo_name] = algo_results
    
    # Use truncated results for plotting
    results = truncated_results
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sample Efficiency Comparison: REINFORCE vs PPO vs SAC (Truncated to SAC\'s Sample Count)', fontsize=16, fontweight='bold')
    
    # Define colors and styles
    colors = {
        'REINFORCE': '#1f77b4',  # Blue
        'PPO': '#ff7f0e',        # Orange
        'SAC': '#2ca02c'          # Green
    }
    
    # Plot 1: Reward vs Episodes
    ax1 = axes[0, 0]
    for algo_name, algo_results in results.items():
        if algo_results is not None:
            episodes = np.arange(1, len(algo_results['episode_rewards']) + 1)
            rewards = algo_results['episode_rewards']
            
            # Plot individual episodes (light)
            ax1.plot(episodes, rewards, alpha=0.3, color=colors[algo_name], linewidth=0.5)
            
            # Plot moving average (bold)
            window = min(10, len(rewards) // 4) if len(rewards) > 0 else 1
            if window > 1:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                moving_episodes = episodes[window-1:]
                ax1.plot(moving_episodes, moving_avg, color=colors[algo_name], 
                        linewidth=2, label=f'{algo_name} (avg)')
            else:
                ax1.plot(episodes, rewards, color=colors[algo_name], 
                        linewidth=2, label=algo_name)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('Reward vs Episodes', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward vs Total Samples (Sample Efficiency)
    # Interpolate all algorithms to same sample points for fair comparison
    ax2 = axes[0, 1]
    # Create sample points for interpolation (start at 0, go to min_samples)
    sample_points = np.linspace(0, min_samples, 100) if min_samples != float('inf') else None
    
    for algo_name, algo_results in results.items():
        if algo_results is not None and len(algo_results['cumulative_samples']) > 0:
            samples = np.array(algo_results['cumulative_samples'])
            rewards = np.array(algo_results['episode_rewards'])
            
            # Compute running average reward
            window = 10
            if len(rewards) >= window:
                running_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                running_samples = samples[window-1:]
            else:
                running_avg = rewards
                running_samples = samples
            
            # Interpolate to common sample points if truncating
            if sample_points is not None and len(running_samples) > 1:
                # Only interpolate within the range we have data for
                valid_mask = running_samples <= min_samples
                if np.any(valid_mask):
                    valid_samples = running_samples[valid_mask]
                    valid_rewards = running_avg[valid_mask] if len(running_avg) == len(running_samples) else running_avg[:len(valid_samples)]
                    
                    # Interpolate
                    interp_rewards = np.interp(sample_points, valid_samples, valid_rewards, 
                                             left=valid_rewards[0] if len(valid_rewards) > 0 else 0,
                                             right=valid_rewards[-1] if len(valid_rewards) > 0 else 0)
                    ax2.plot(sample_points, interp_rewards, color=colors[algo_name], 
                            linewidth=2.5, label=algo_name, alpha=0.8)
                else:
                    # Fallback: plot what we have
                    ax2.plot(running_samples, running_avg, color=colors[algo_name], 
                            linewidth=2.5, label=algo_name, alpha=0.8)
            else:
                # No truncation needed, plot normally
                ax2.plot(running_samples, running_avg, color=colors[algo_name], 
                        linewidth=2.5, label=algo_name, alpha=0.8)
    
    ax2.set_xlabel('Total Samples (Steps)', fontsize=12)
    ax2.set_ylabel('Average Reward (10-episode window)', fontsize=12)
    ax2.set_title('Sample Efficiency: Average Reward vs Total Samples', fontsize=13, fontweight='bold')
    if sample_points is not None:
        ax2.set_xlim(0, min_samples)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average Reward vs Episodes (smoothed)
    ax3 = axes[1, 0]
    for algo_name, algo_results in results.items():
        if algo_results is not None:
            episodes = np.arange(1, len(algo_results['episode_rewards']) + 1)
            rewards = np.array(algo_results['episode_rewards'])
            
            # Compute running average
            window = 10
            if len(rewards) >= window:
                running_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                running_episodes = episodes[window-1:]
                ax3.plot(running_episodes, running_avg, color=colors[algo_name], 
                        linewidth=2.5, label=algo_name)
            else:
                # If not enough episodes, just plot the mean
                mean_reward = np.mean(rewards) if len(rewards) > 0 else 0
                ax3.axhline(y=mean_reward, color=colors[algo_name], 
                           linewidth=2.5, label=f'{algo_name} (mean)', linestyle='--')
    
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Average Reward (10-episode window)', fontsize=12)
    ax3.set_title('Learning Progress: Average Reward vs Episodes', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average Reward vs Total Samples (smoothed, sample efficiency)
    # Use same interpolation approach as Plot 2
    ax4 = axes[1, 1]
    
    for algo_name, algo_results in results.items():
        if algo_results is not None and len(algo_results['cumulative_samples']) > 0:
            samples = np.array(algo_results['cumulative_samples'])
            rewards = np.array(algo_results['episode_rewards'])
            
            # Compute running average
            window = 10
            if len(rewards) >= window:
                running_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                running_samples = samples[window-1:]
            else:
                running_avg = rewards
                running_samples = samples
            
            # Interpolate to common sample points if truncating
            if sample_points is not None and len(running_samples) > 1:
                valid_mask = running_samples <= min_samples
                if np.any(valid_mask):
                    valid_samples = running_samples[valid_mask]
                    valid_rewards = running_avg[valid_mask] if len(running_avg) == len(running_samples) else running_avg[:len(valid_samples)]
                    
                    # Interpolate
                    interp_rewards = np.interp(sample_points, valid_samples, valid_rewards,
                                             left=valid_rewards[0] if len(valid_rewards) > 0 else 0,
                                             right=valid_rewards[-1] if len(valid_rewards) > 0 else 0)
                    ax4.plot(sample_points, interp_rewards, color=colors[algo_name], 
                            linewidth=2.5, label=algo_name, alpha=0.8)
                else:
                    ax4.plot(running_samples, running_avg, color=colors[algo_name], 
                            linewidth=2.5, label=algo_name, alpha=0.8)
            else:
                ax4.plot(running_samples, running_avg, color=colors[algo_name], 
                        linewidth=2.5, label=algo_name, alpha=0.8)
    
    ax4.set_xlabel('Total Samples (Steps)', fontsize=12)
    ax4.set_ylabel('Average Reward (10-episode window)', fontsize=12)
    ax4.set_title('Sample Efficiency: Average Reward vs Total Samples', fontsize=13, fontweight='bold')
    if sample_points is not None:
        ax4.set_xlim(0, min_samples)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, "sample_efficiency_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_file}")
    
    # Also save as PDF
    plot_file_pdf = os.path.join(output_dir, "sample_efficiency_comparison.pdf")
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_file_pdf}")
    
    plt.close()
    
    # Create a summary plot (single figure, focused on sample efficiency)
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for algo_name, algo_results in results.items():
        if algo_results is not None and len(algo_results['cumulative_samples']) > 0:
            samples = np.array(algo_results['cumulative_samples'])
            rewards = np.array(algo_results['episode_rewards'])
            
            # Compute running average
            window = 10
            if len(rewards) >= window:
                running_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                running_samples = samples[window-1:]
            else:
                running_avg = rewards
                running_samples = samples
            
            # Interpolate to common sample points
            if sample_points is not None and len(running_samples) > 1:
                valid_mask = running_samples <= min_samples
                if np.any(valid_mask):
                    valid_samples = running_samples[valid_mask]
                    valid_rewards = running_avg[valid_mask] if len(running_avg) == len(running_samples) else running_avg[:len(valid_samples)]
                    
                    # Interpolate
                    interp_rewards = np.interp(sample_points, valid_samples, valid_rewards,
                                             left=valid_rewards[0] if len(valid_rewards) > 0 else 0,
                                             right=valid_rewards[-1] if len(valid_rewards) > 0 else 0)
                    ax.plot(sample_points, interp_rewards, color=colors[algo_name], 
                           linewidth=3, label=algo_name, alpha=0.9)
                else:
                    ax.plot(running_samples, running_avg, color=colors[algo_name], 
                           linewidth=3, label=algo_name, alpha=0.9)
            else:
                ax.plot(running_samples, running_avg, color=colors[algo_name], 
                       linewidth=3, label=algo_name, alpha=0.9)
    
    ax.set_xlabel('Total Samples (Steps)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward (10-episode window)', fontsize=14, fontweight='bold')
    ax.set_title(f'Sample Efficiency Comparison: REINFORCE vs PPO vs SAC (Truncated to {min_samples:,} samples)', 
                fontsize=16, fontweight='bold')
    if sample_points is not None:
        ax.set_xlim(0, min_samples)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary plot
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
            print(f"\n{algo_name}:")
            print(f"  Episodes: {len(algo_results['episode_rewards'])}")
            print(f"  Total Samples: {algo_results['total_samples']:,}")
            print(f"  Final Average Reward (last 10): {algo_results['final_avg_reward']:.2f}")
            print(f"  Overall Average Reward: {np.mean(algo_results['episode_rewards']):.2f}")
            print(f"  Best Episode Reward: {np.max(algo_results['episode_rewards']):.2f}")
            print(f"  Average Episode Length: {np.mean(algo_results['episode_lengths']):.1f}")
        else:
            print(f"\n{algo_name}: FAILED")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare sample efficiency of REINFORCE, PPO, and SAC')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes to train each algorithm (default: 100)')
    parser.add_argument('--max_steps', type=int, default=2000,
                       help='Maximum steps per episode (default: 2000)')
    parser.add_argument('--output_dir', type=str, default='sample_efficiency_comparison',
                       help='Directory to save results and plots (default: sample_efficiency_comparison)')
    
    args = parser.parse_args()
    
    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    # Train all algorithms
    results = train_all_algorithms(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
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

