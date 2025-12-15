"""
Hyperparameter tuning script for SAC agent.

This script runs a grid search over hyperparameters, training each configuration
for a specified number of episodes. It supports resuming from a specific configuration
if training is interrupted.

Usage:
    python src/hyperparameter_tuning.py
    python src/hyperparameter_tuning.py --resume_from config_0
    python src/hyperparameter_tuning.py --only config_5 config_7
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from itertools import product
import numpy as np


def generate_hyperparameter_grid():
    """
    Define the hyperparameter grid to search over.
    
    Focused grid on the 4 most important hyperparameters.
    Current grid: 2 * 2 * 2 * 2 = 16 configurations
    
    Returns:
        List of dictionaries, each containing a hyperparameter configuration.
    """
    # Define hyperparameter ranges
    # Focused grid on key hyperparameters: lr, hidden_dim, tau, alpha
    hyperparams = {
        'lr': [1e-4, 3e-4],  # Learning rate (2 values)
        'hidden_dim': [128, 256],  # Network hidden dimension (2 values)
        'tau': [0.005, 0.01],  # Soft update coefficient (2 values)
        'alpha': [0.1, 0.2],  # Entropy coefficient initial value (2 values)
    }
    
    # Generate all combinations
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    
    configs = []
    for combo in product(*values):
        config = dict(zip(keys, combo))
        # Add default values for other hyperparameters
        config['batch_size'] = 256  # Fixed default
        config['update_freq'] = 1  # Fixed default
        configs.append(config)
    
    return configs


def config_to_string(config):
    """Convert a configuration dictionary to a string identifier."""
    parts = []
    for key in sorted(config.keys()):
        value = config[key]
        if isinstance(value, float):
            # Format float to avoid decimal points in name
            if value < 1 and abs(value) < 0.1:
                # Use scientific notation for small numbers (e.g., 1e-4)
                exp = int(np.log10(value))
                coeff = value / (10 ** exp)
                parts.append(f"{key}_{coeff:.0f}e{exp}".replace('-', 'm'))
            else:
                # Format as decimal (e.g., 0.005 -> 0p005)
                parts.append(f"{key}_{str(value).replace('.', 'p')}")
        else:
            parts.append(f"{key}_{value}")
    return "_".join(parts)


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    policy_files = list(Path(checkpoint_dir).glob("sac_policy_ep*.pth"))
    if not policy_files:
        return None, 0
    
    # Extract episode numbers
    episodes = []
    for f in policy_files:
        try:
            ep_num = int(f.stem.split('_ep')[1])
            episodes.append(ep_num)
        except:
            continue
    
    if not episodes:
        return None, 0
    
    latest_ep = max(episodes)
    return latest_ep, latest_ep


def is_config_complete(config_dir, target_episodes):
    """Check if a configuration has completed training."""
    latest_ep, _ = find_latest_checkpoint(config_dir)
    return latest_ep is not None and latest_ep >= target_episodes


def run_hyperparameter_tuning(
    base_dir="hyperparameter_tuning",
    num_episodes=400,
    max_steps_per_episode=2000,
    save_freq=100,
    use_gym_mode=False,
    step_duration=0.1,
    resume_from=None,
    only_configs=None,
    skip_completed=True
):
    """
    Run hyperparameter tuning experiments.
    
    Args:
        base_dir: Base directory for all experiments
        num_episodes: Number of episodes to train each configuration
        max_steps_per_episode: Maximum steps per episode
        save_freq: Frequency to save checkpoints
        use_gym_mode: Whether to use gym mode (faster)
        step_duration: Step duration for gym mode
        resume_from: Configuration name to resume from (None = start from beginning)
        only_configs: List of configuration names to run (None = run all)
        skip_completed: Whether to skip configurations that are already complete
    """
    # Generate hyperparameter grid
    configs = generate_hyperparameter_grid()
    print(f"Generated {len(configs)} hyperparameter configurations")
    print(f"Each configuration will train for {num_episodes} episodes")
    print(f"Estimated total: {len(configs)} configurations × {num_episodes} episodes = {len(configs) * num_episodes} episodes")
    
    # Create base directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"tuning_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save hyperparameter grid
    grid_file = exp_dir / "hyperparameter_grid.json"
    with open(grid_file, 'w') as f:
        json.dump({
            'configs': configs,
            'num_episodes': num_episodes,
            'max_steps_per_episode': max_steps_per_episode,
            'save_freq': save_freq,
            'use_gym_mode': use_gym_mode,
            'step_duration': step_duration,
        }, f, indent=2)
    print(f"Saved hyperparameter grid to {grid_file}")
    
    # Track progress
    progress_file = exp_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {
            'completed': [],
            'in_progress': [],
            'failed': []
        }
    
    # Determine starting index
    start_idx = 0
    if resume_from:
        # Find the index of the configuration to resume from
        for idx, config in enumerate(configs):
            config_name = config_to_string(config)
            if config_name == resume_from:
                start_idx = idx
                print(f"Resuming from configuration: {resume_from} (index {idx})")
                break
        else:
            print(f"Warning: Configuration '{resume_from}' not found. Starting from beginning.")
    
    # Run each configuration
    for idx, config in enumerate(configs[start_idx:], start=start_idx):
        config_name = config_to_string(config)
        config_name = f"config_{idx}_{config_name}"
        
        # Skip if only running specific configs
        if only_configs and config_name not in only_configs:
            print(f"Skipping {config_name} (not in --only list)")
            continue
        
        # Skip if already completed
        config_dir = exp_dir / config_name / "checkpoints"
        if skip_completed and is_config_complete(config_dir, num_episodes):
            print(f"Skipping {config_name} (already completed)")
            progress['completed'].append(config_name)
            continue
        
        print(f"\n{'='*80}")
        print(f"Configuration {idx+1}/{len(configs)}: {config_name}")
        print(f"{'='*80}")
        print(f"Hyperparameters: {json.dumps(config, indent=2)}")
        
        # Check if we can resume this configuration
        latest_ep, _ = find_latest_checkpoint(config_dir)
        start_episode = 0
        policy_checkpoint = None
        q1_checkpoint = None
        q2_checkpoint = None
        
        if latest_ep is not None and latest_ep > 0:
            # Resume from latest checkpoint
            start_episode = latest_ep
            policy_checkpoint = str(config_dir / f"sac_policy_ep{latest_ep}.pth")
            q1_checkpoint = str(config_dir / f"sac_q1_ep{latest_ep}.pth")
            q2_checkpoint = str(config_dir / f"sac_q2_ep{latest_ep}.pth")
            print(f"Resuming from episode {latest_ep}")
        
        # Create directories
        config_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = exp_dir / config_name / "training_logs"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            sys.executable, "src/sac_agent.py",
            "--num_episodes", str(num_episodes),
            "--max_steps_per_episode", str(max_steps_per_episode),
            "--batch_size", str(config['batch_size']),
            "--update_freq", str(config['update_freq']),
            "--save_freq", str(save_freq),
            "--checkpoint_dir", str(config_dir),
            "--metrics_dir", str(metrics_dir),
            "--start_episode", str(start_episode),
        ]
        
        # Add hyperparameters (we'll need to modify sac_agent.py to accept these)
        # For now, we'll pass them via environment variables or modify the script
        # Actually, we need to modify train_sac to accept these hyperparameters
        
        if use_gym_mode:
            cmd.extend(["--gym_mode", "--step_duration", str(step_duration)])
        
        if policy_checkpoint:
            cmd.extend([
                "--policy_checkpoint", policy_checkpoint,
                "--q1_checkpoint", q1_checkpoint,
                "--q2_checkpoint", q2_checkpoint,
            ])
        
        # Save configuration to file for the training script to read
        config_file = exp_dir / config_name / "hyperparams.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        cmd.extend(["--hyperparams_file", str(config_file)])
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Checkpoints: {config_dir}")
        print(f"Metrics: {metrics_dir}")
        
        try:
            # Run training
            result = subprocess.run(cmd, check=True)
            
            # Mark as completed
            if config_name not in progress['completed']:
                progress['completed'].append(config_name)
            if config_name in progress['in_progress']:
                progress['in_progress'].remove(config_name)
            
            print(f"✓ Completed {config_name}")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed {config_name}: {e}")
            if config_name not in progress['failed']:
                progress['failed'].append(config_name)
            if config_name in progress['in_progress']:
                progress['in_progress'].remove(config_name)
        
        # Save progress
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Hyperparameter Tuning Summary")
    print(f"{'='*80}")
    print(f"Total configurations: {len(configs)}")
    print(f"Completed: {len(progress['completed'])}")
    print(f"Failed: {len(progress['failed'])}")
    print(f"Results saved to: {exp_dir}")
    print(f"\nCompleted configurations:")
    for config_name in progress['completed']:
        print(f"  - {config_name}")
    if progress['failed']:
        print(f"\nFailed configurations:")
        for config_name in progress['failed']:
            print(f"  - {config_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for SAC agent')
    parser.add_argument('--base_dir', type=str, default='hyperparameter_tuning',
                        help='Base directory for experiments (default: hyperparameter_tuning)')
    parser.add_argument('--num_episodes', type=int, default=400,
                        help='Number of episodes per configuration (default: 400)')
    parser.add_argument('--max_steps_per_episode', type=int, default=2000,
                        help='Maximum steps per episode (default: 2000)')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Checkpoint save frequency (default: 100)')
    parser.add_argument('--gym_mode', action='store_true',
                        help='Use gym mode (faster, default: False = real-time)')
    parser.add_argument('--step_duration', type=float, default=0.1,
                        help='Step duration for gym mode (default: 0.1)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Configuration name to resume from (e.g., config_0_lr_3e-04_batch_size_256)')
    parser.add_argument('--only', nargs='+', default=None,
                        help='Only run these specific configuration names')
    parser.add_argument('--no_skip_completed', dest='skip_completed', action='store_false',
                        help='Do not skip completed configurations')
    
    args = parser.parse_args()
    
    run_hyperparameter_tuning(
        base_dir=args.base_dir,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        save_freq=args.save_freq,
        use_gym_mode=args.gym_mode,
        step_duration=args.step_duration,
        resume_from=args.resume_from,
        only_configs=args.only,
        skip_completed=args.skip_completed
    )

