"""
Compare and visualize results from multiple training experiments.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict


def load_comparison_results(results_file: str) -> List[Dict]:
    """Load comparison results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_comparison(results_file: str, save_dir: str = None):
    """
    Generate comparison plots from experiment results.
    
    Args:
        results_file: Path to comparison_results.json
        save_dir: Directory to save plots (default: same as results file)
    """
    results = load_comparison_results(results_file)
    
    if save_dir is None:
        save_dir = str(Path(results_file).parent)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Separate real-time and gym mode results
    realtime_results = [r for r in results if not r.get("use_gym_mode", False)]
    gym_results = [r for r in results if r.get("use_gym_mode", False)]
    
    # Sort gym results by step duration
    gym_results.sort(key=lambda x: x.get("step_duration", 0))
    
    # Extract data
    step_durations = [r["step_duration"] for r in gym_results]
    avg_rewards = [r.get("avg_reward", 0) for r in gym_results]
    reward_stds = [r.get("reward_std", 0) for r in gym_results]
    avg_lengths = [r.get("avg_length", 0) for r in gym_results]
    max_rewards = [r.get("max_reward", 0) for r in gym_results]
    
    realtime_reward = realtime_results[0].get("avg_reward", 0) if realtime_results else None
    realtime_reward_std = realtime_results[0].get("reward_std", 0) if realtime_results else None
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Average Reward vs Step Duration
    ax = axes[0, 0]
    if step_durations and avg_rewards:
        ax.errorbar(step_durations, avg_rewards, yerr=reward_stds, 
                   marker='o', linestyle='-', linewidth=2, markersize=8,
                   label='Gym Mode', capsize=5)
    if realtime_reward is not None:
        ax.axhline(realtime_reward, color='red', linestyle='--', linewidth=2,
                  label=f'Real-time Baseline ({realtime_reward:.2f})')
        if realtime_reward_std:
            ax.fill_between(ax.get_xlim(), 
                           realtime_reward - realtime_reward_std,
                           realtime_reward + realtime_reward_std,
                           alpha=0.2, color='red')
    ax.set_xlabel('Step Duration (seconds)')
    ax.set_ylabel('Average Reward')
    ax.set_title('Performance vs Step Duration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Max Reward vs Step Duration
    ax = axes[0, 1]
    if step_durations and max_rewards:
        ax.plot(step_durations, max_rewards, marker='s', linestyle='-', 
               linewidth=2, markersize=8, label='Gym Mode', color='green')
    if realtime_results:
        realtime_max = realtime_results[0].get("max_reward", 0)
        ax.axhline(realtime_max, color='red', linestyle='--', linewidth=2,
                  label=f'Real-time Baseline ({realtime_max:.2f})')
    ax.set_xlabel('Step Duration (seconds)')
    ax.set_ylabel('Max Reward')
    ax.set_title('Best Performance vs Step Duration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Average Episode Length vs Step Duration
    ax = axes[1, 0]
    if step_durations and avg_lengths:
        ax.plot(step_durations, avg_lengths, marker='^', linestyle='-',
               linewidth=2, markersize=8, label='Gym Mode', color='purple')
    if realtime_results:
        realtime_length = realtime_results[0].get("avg_length", 0)
        ax.axhline(realtime_length, color='red', linestyle='--', linewidth=2,
                  label=f'Real-time Baseline ({realtime_length:.1f})')
    ax.set_xlabel('Step Duration (seconds)')
    ax.set_ylabel('Average Episode Length')
    ax.set_title('Episode Length vs Step Duration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Performance Degradation (relative to real-time)
    ax = axes[1, 1]
    if realtime_reward is not None and step_durations and avg_rewards:
        degradation = [(realtime_reward - reward) / realtime_reward * 100 
                      for reward in avg_rewards]
        ax.plot(step_durations, degradation, marker='o', linestyle='-',
               linewidth=2, markersize=8, color='orange')
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Step Duration (seconds)')
        ax.set_ylabel('Performance Degradation (%)')
        ax.set_title('Performance Degradation vs Step Duration')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Real-time baseline required', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Performance Degradation vs Step Duration')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'delay_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_dir / 'delay_comparison.png'}")
    plt.close()
    
    # Create summary table
    print_summary_table(results, realtime_results)
    
    # Save summary
    summary = {
        "realtime_baseline": realtime_results[0] if realtime_results else None,
        "gym_mode_results": gym_results,
        "performance_degradation": {}
    }
    
    if realtime_reward is not None:
        for result in gym_results:
            step_dur = result["step_duration"]
            reward = result.get("avg_reward", 0)
            degradation = (realtime_reward - reward) / realtime_reward * 100
            summary["performance_degradation"][f"{step_dur}s"] = degradation
    
    summary_file = save_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")


def print_summary_table(results: List[Dict], realtime_results: List[Dict]):
    """Print a formatted summary table."""
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*80)
    
    # Real-time baseline
    if realtime_results:
        rt = realtime_results[0]
        print(f"\nReal-time Baseline:")
        print(f"  Average Reward: {rt.get('avg_reward', 'N/A'):.2f} ± {rt.get('reward_std', 0):.2f}")
        print(f"  Average Length: {rt.get('avg_length', 'N/A'):.1f} ± {rt.get('length_std', 0):.1f}")
        print(f"  Max Reward: {rt.get('max_reward', 'N/A'):.2f}")
        print(f"  Min Reward: {rt.get('min_reward', 'N/A'):.2f}")
    
    # Gym mode results
    gym_results = [r for r in results if r.get("use_gym_mode", False)]
    gym_results.sort(key=lambda x: x.get("step_duration", 0))
    
    if gym_results:
        print(f"\nGym Mode Results:")
        print(f"{'Step Duration':<15} {'Avg Reward':<15} {'Avg Length':<15} {'Max Reward':<15} {'Degradation':<15}")
        print("-" * 80)
        
        realtime_reward = realtime_results[0].get("avg_reward", 0) if realtime_results else None
        
        for result in gym_results:
            step_dur = result["step_duration"]
            reward = result.get("avg_reward", 0)
            length = result.get("avg_length", 0)
            max_r = result.get("max_reward", 0)
            
            if realtime_reward is not None:
                degradation = (realtime_reward - reward) / realtime_reward * 100
                deg_str = f"{degradation:.1f}%"
            else:
                deg_str = "N/A"
            
            print(f"{step_dur:<15.3f} {reward:<15.2f} {length:<15.1f} {max_r:<15.2f} {deg_str:<15}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to comparison_results.json')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save plots (default: same as results file)')
    
    args = parser.parse_args()
    
    plot_comparison(args.results_file, args.save_dir)

