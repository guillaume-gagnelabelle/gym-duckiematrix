"""
Analyze results from hyperparameter tuning experiments.

This script reads the results from hyperparameter tuning and generates:
1. A summary table of all configurations and their final performance
2. Plots comparing different hyperparameter values
3. Identifies the best performing configurations

Usage:
    python src/analyze_hyperparameter_tuning.py --experiment_dir hyperparameter_tuning/tuning_20251212_120000
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob


def load_metrics_from_dir(metrics_dir):
    """Load training metrics from a metrics directory."""
    metrics_file = Path(metrics_dir) / "training_metrics.json"
    if not metrics_file.exists():
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def extract_final_performance(metrics):
    """Extract final performance metrics from training metrics."""
    if metrics is None:
        return None
    
    episode_rewards = metrics.get('episode_rewards', [])
    episode_lengths = metrics.get('episode_lengths', [])
    
    if not episode_rewards:
        return None
    
    # Get last 10% of episodes for final performance
    n_episodes = len(episode_rewards)
    final_start = max(0, int(0.9 * n_episodes))
    final_rewards = episode_rewards[final_start:]
    final_lengths = episode_lengths[final_start:] if episode_lengths else []
    
    return {
        'mean_reward': np.mean(final_rewards),
        'std_reward': np.std(final_rewards),
        'max_reward': np.max(final_rewards),
        'min_reward': np.min(final_rewards),
        'mean_length': np.mean(final_lengths) if final_lengths else None,
        'total_episodes': n_episodes,
        'final_10pct_mean': np.mean(final_rewards),
        'final_10pct_std': np.std(final_rewards),
    }


def analyze_hyperparameter_tuning(experiment_dir):
    """Analyze hyperparameter tuning results."""
    experiment_dir = Path(experiment_dir)
    
    # Load hyperparameter grid
    grid_file = experiment_dir / "hyperparameter_grid.json"
    if not grid_file.exists():
        print(f"Error: Could not find hyperparameter_grid.json in {experiment_dir}")
        return
    
    with open(grid_file, 'r') as f:
        grid_data = json.load(f)
    
    configs = grid_data['configs']
    
    # Load progress
    progress_file = experiment_dir / "progress.json"
    progress = {'completed': [], 'failed': []}
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    
    # Collect results for each configuration
    results = []
    
    for idx, config in enumerate(configs):
        config_name = f"config_{idx}_{config_to_string(config)}"
        config_dir = experiment_dir / config_name
        
        if not config_dir.exists():
            continue
        
        metrics_dir = config_dir / "training_logs"
        metrics = load_metrics_from_dir(metrics_dir)
        performance = extract_final_performance(metrics)
        
        if performance is None:
            continue
        
        result = {
            'config_name': config_name,
            'config_idx': idx,
            **config,
            **performance
        }
        results.append(result)
    
    if not results:
        print("No results found. Make sure training has completed for at least one configuration.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by final performance
    df = df.sort_values('final_10pct_mean', ascending=False)
    
    # Print summary
    print("\n" + "="*80)
    print("Hyperparameter Tuning Results Summary")
    print("="*80)
    print(f"\nTotal configurations tested: {len(results)}")
    print(f"Completed: {len(progress.get('completed', []))}")
    print(f"Failed: {len(progress.get('failed', []))}")
    
    print("\n" + "-"*80)
    print("Top Configurations (by final 10% mean reward):")
    print("-"*80)
    top_cols = ['config_idx', 'lr', 'hidden_dim', 'tau', 'alpha', 
                'final_10pct_mean', 'final_10pct_std', 'total_episodes']
    print(df[top_cols].head(min(10, len(df))).to_string(index=False))
    
    print("\n" + "-"*80)
    print("Best Configuration:")
    print("-"*80)
    best = df.iloc[0]
    print(f"Config Index: {best['config_idx']}")
    print(f"Config Name: {best['config_name']}")
    print(f"Hyperparameters:")
    for key in ['lr', 'hidden_dim', 'tau', 'alpha']:
        print(f"  {key}: {best[key]}")
    print(f"Performance:")
    print(f"  Final 10% Mean Reward: {best['final_10pct_mean']:.2f} ± {best['final_10pct_std']:.2f}")
    print(f"  Max Reward: {best['max_reward']:.2f}")
    print(f"  Min Reward: {best['min_reward']:.2f}")
    
    # Save summary to CSV
    summary_file = experiment_dir / "hyperparameter_tuning_summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
    
    # Generate plots
    plot_dir = experiment_dir / "analysis_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Plot 1: Performance by hyperparameter
    hyperparams_to_plot = ['lr', 'hidden_dim', 'tau', 'alpha']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, hp in enumerate(hyperparams_to_plot):
        ax = axes[idx]
        
        # Group by hyperparameter value
        grouped = df.groupby(hp)['final_10pct_mean'].agg(['mean', 'std', 'count'])
        
        x = grouped.index
        y = grouped['mean']
        yerr = grouped['std'] / np.sqrt(grouped['count'])  # Standard error
        
        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
        ax.set_xlabel(hp, fontsize=12)
        ax.set_ylabel('Final 10% Mean Reward', fontsize=12)
        ax.set_title(f'Performance vs {hp}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / "hyperparameter_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {plot_dir}/")
    
    # Plot 2: Learning curves for top 5 configurations
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_5_configs = df.head(5)
    
    for _, row in top_5_configs.iterrows():
        config_idx = row['config_idx']
        config_name = row['config_name']
        config_dir = experiment_dir / config_name
        metrics_dir = config_dir / "training_logs"
        metrics = load_metrics_from_dir(metrics_dir)
        
        if metrics is None:
            continue
        
        episode_rewards = metrics.get('episode_rewards', [])
        if not episode_rewards:
            continue
        
        episodes = np.arange(1, len(episode_rewards) + 1)
        
        # Compute moving average
        window = max(10, len(episode_rewards) // 20)
        if len(episode_rewards) > window:
            moving_avg = pd.Series(episode_rewards).rolling(window=window, center=True).mean()
            ax.plot(episodes, moving_avg, label=f"Config {config_idx}", linewidth=2, alpha=0.8)
        else:
            ax.plot(episodes, episode_rewards, label=f"Config {config_idx}", linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Reward (Moving Average)', fontsize=14)
    ax.set_title('Learning Curves: Top 5 Configurations', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / "top5_learning_curves.png", dpi=150, bbox_inches='tight')
    
    print("\nAnalysis complete!")


def config_to_string(config):
    """Convert a configuration dictionary to a string identifier (same as in hyperparameter_tuning.py)."""
    parts = []
    for key in sorted(config.keys()):
        value = config[key]
        if isinstance(value, float):
            if value < 1 and abs(value) < 0.1:
                exp = int(np.log10(value))
                coeff = value / (10 ** exp)
                parts.append(f"{key}_{coeff:.0f}e{exp}".replace('-', 'm'))
            else:
                parts.append(f"{key}_{str(value).replace('.', 'p')}")
        else:
            parts.append(f"{key}_{value}")
    return "_".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze hyperparameter tuning results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to the hyperparameter tuning experiment directory')
    
    args = parser.parse_args()
    
    analyze_hyperparameter_tuning(args.experiment_dir)

