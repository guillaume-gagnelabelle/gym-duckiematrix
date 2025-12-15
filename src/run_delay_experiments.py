"""
Run training experiments with different step durations to compare performance.
Trains multiple agents with varying delays and evaluates them.
"""

import argparse
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import sys


def find_latest_checkpoint(checkpoint_dir: str):
    """Find the latest checkpoint in a directory."""
    import glob
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None, 0
    
    # Find all policy checkpoints
    policy_files = list(checkpoint_dir.glob("sac_policy_ep*.pth"))
    if not policy_files:
        return None, 0
    
    # Extract episode numbers and find the latest
    episodes = []
    for f in policy_files:
        try:
            # Extract episode number from filename like "sac_policy_ep400.pth"
            ep_num = int(f.stem.split("_ep")[1])
            episodes.append((ep_num, f))
        except (IndexError, ValueError):
            continue
    
    if not episodes:
        return None, 0
    
    # Return the latest checkpoint
    latest_ep, latest_file = max(episodes, key=lambda x: x[0])
    return str(latest_file), latest_ep


def run_training_experiment(experiment_name: str, use_gym_mode: bool, step_duration: float,
                           num_episodes: int, max_steps_per_episode: int, 
                           checkpoint_dir: str, metrics_dir: str, 
                           batch_size: int = 256, update_freq: int = 1, save_freq: int = 50,
                           resume_from_existing: bool = True):
    """
    Run a single training experiment.
    
    Returns:
        Path to the final checkpoint, or None if training failed
    """
    # Check for existing checkpoints if resuming
    start_episode = 0
    policy_checkpoint = None
    q1_checkpoint = None
    q2_checkpoint = None
    
    if resume_from_existing:
        latest_policy, latest_ep = find_latest_checkpoint(checkpoint_dir)
        if latest_policy:
            start_episode = latest_ep
            policy_checkpoint = latest_policy
            # Find corresponding Q checkpoints
            checkpoint_path = Path(latest_policy)
            q1_path = checkpoint_path.parent / f"sac_q1_ep{latest_ep}.pth"
            q2_path = checkpoint_path.parent / f"sac_q2_ep{latest_ep}.pth"
            if q1_path.exists():
                q1_checkpoint = str(q1_path)
            if q2_path.exists():
                q2_checkpoint = str(q2_path)
            
            # Check if already complete
            final_checkpoint = Path(checkpoint_dir) / "sac_policy_final.pth"
            if final_checkpoint.exists() or latest_ep >= num_episodes:
                print(f"\n{'='*60}")
                print(f"Skipping: {experiment_name} (already complete at episode {latest_ep})")
                print(f"{'='*60}\n")
                return str(final_checkpoint) if final_checkpoint.exists() else latest_policy
    
    remaining_episodes = num_episodes - start_episode
    if remaining_episodes <= 0:
        print(f"\n{'='*60}")
        print(f"Skipping: {experiment_name} (already complete)")
        print(f"{'='*60}\n")
        final_checkpoint = Path(checkpoint_dir) / "sac_policy_final.pth"
        return str(final_checkpoint) if final_checkpoint.exists() else None
    
    print(f"\n{'='*60}")
    print(f"Training: {experiment_name}")
    print(f"  Mode: {'GYM' if use_gym_mode else 'REAL-TIME'}")
    if use_gym_mode:
        print(f"  Step Duration: {step_duration}s")
    if start_episode > 0:
        print(f"  Resuming from episode {start_episode}")
        print(f"  Remaining episodes: {remaining_episodes}")
    else:
        print(f"  Episodes: {num_episodes}")
    print(f"{'='*60}\n")
    
    # Build command
    cmd = [
        "python", "src/sac_agent.py",
        "--num_episodes", str(remaining_episodes),
        "--max_steps_per_episode", str(max_steps_per_episode),
        "--batch_size", str(batch_size),
        "--update_freq", str(update_freq),
        "--save_freq", str(save_freq),
        "--checkpoint_dir", checkpoint_dir,
        "--metrics_dir", metrics_dir,
        "--start_episode", str(start_episode),
    ]
    
    if use_gym_mode:
        cmd.extend(["--gym_mode", "--step_duration", str(step_duration)])
    
    if policy_checkpoint:
        cmd.extend(["--policy_checkpoint", policy_checkpoint])
    if q1_checkpoint:
        cmd.extend(["--q1_checkpoint", q1_checkpoint])
    if q2_checkpoint:
        cmd.extend(["--q2_checkpoint", q2_checkpoint])
    
    # Run training - don't capture output so we can see progress in real-time
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)  # No capture_output - prints directly to console
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for {experiment_name}")
        return None
    
    # Return path to final checkpoint
    final_checkpoint = os.path.join(checkpoint_dir, "sac_policy_final.pth")
    return final_checkpoint if os.path.exists(final_checkpoint) else None


def run_evaluation(experiment_name: str, checkpoint_path: str, use_gym_mode: bool,
                  step_duration: float, num_episodes: int, max_steps: int):
    """
    Run evaluation on a trained agent.
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating: {experiment_name}")
    
    cmd = [
        "python", "src/sac_inference.py",
        "--policy_checkpoint", checkpoint_path,
        "--num_episodes", str(num_episodes),
        "--max_steps", str(max_steps),
        "--no_render",
        "--save_metrics",  # Save metrics to JSON
    ]
    
    if use_gym_mode:
        cmd.extend(["--gym_mode", "--step_duration", str(step_duration)])
    
    result = subprocess.run(cmd)  # No capture_output - prints directly to console
    
    if result.returncode != 0:
        print(f"ERROR: Evaluation failed for {experiment_name}")
        return None
    
    # Try to load metrics from JSON file (preferred method)
    metrics_file = Path(checkpoint_path).parent / "evaluation_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            eval_metrics = json.load(f)
    else:
        # If JSON file doesn't exist, create empty metrics dict
        # (We can't parse stdout since we're not capturing it)
        eval_metrics = {
            "avg_reward": None,
            "avg_length": None,
            "note": "Metrics not available - check evaluation output or JSON file"
        }
    
    # Combine with experiment info
    metrics = {
        "experiment_name": experiment_name,
        "use_gym_mode": use_gym_mode,
        "step_duration": step_duration if use_gym_mode else 0.0,
        "checkpoint": checkpoint_path,
        **eval_metrics
    }
    
    return metrics


def run_delay_experiments(step_durations: list, num_episodes: int, max_steps_per_episode: int,
                         eval_episodes: int = 20, eval_max_steps: int = 2000,
                         base_dir: str = "delay_experiments", 
                         include_realtime: bool = True,
                         batch_size: int = 256, update_freq: int = 1, save_freq: int = 100,
                         experiment_name: str = None, resume_experiment: str = None):
    """
    Run complete delay comparison experiments.
    
    Args:
        step_durations: List of step durations to test (e.g., [0.05, 0.1, 0.2, 0.5])
        num_episodes: Number of training episodes per experiment
        max_steps_per_episode: Maximum steps per episode during training
        eval_episodes: Number of episodes for evaluation
        eval_max_steps: Maximum steps per episode during evaluation
        base_dir: Base directory for all experiments
        include_realtime: Whether to include real-time baseline
        batch_size: Batch size for training
        update_freq: Update frequency
        save_freq: Save frequency
        experiment_name: Optional name for this experiment run (default: uses timestamp)
        resume_experiment: Path to existing experiment directory to resume from (e.g., "delay_experiments/experiments_20251211_182526")
    """
    # Use existing experiment directory if resuming
    if resume_experiment:
        exp_dir = Path(resume_experiment)
        if not exp_dir.exists():
            raise ValueError(f"Experiment directory does not exist: {resume_experiment}")
        print(f"\n{'='*60}")
        print(f"RESUMING EXISTING EXPERIMENT")
        print(f"Directory: {exp_dir}")
        print(f"{'='*60}\n")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
        else:
            exp_dir = Path(base_dir) / f"experiments_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = exp_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"DELAY EXPERIMENT SUITE")
    print(f"{'='*60}")
    print(f"Base directory: {exp_dir}")
    print(f"Step durations to test: {step_durations}")
    print(f"Training episodes: {num_episodes}")
    print(f"Evaluation episodes: {eval_episodes}")
    print(f"{'='*60}\n")
    
    all_results = []
    
    # 1. Train real-time baseline (if requested)
    if include_realtime:
        exp_name = "realtime_baseline"
        checkpoint_dir = str(exp_dir / "checkpoints" / exp_name)
        metrics_dir = str(exp_dir / "metrics" / exp_name)
        
        checkpoint = run_training_experiment(
            experiment_name=exp_name,
            use_gym_mode=False,
            step_duration=0.0,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            checkpoint_dir=checkpoint_dir,
            metrics_dir=metrics_dir,
            batch_size=batch_size,
            update_freq=update_freq,
            save_freq=save_freq,
            resume_from_existing=resume_experiment is not None
        )
        
        if checkpoint:
            # Evaluate
            eval_results = run_evaluation(
                experiment_name=exp_name,
                checkpoint_path=checkpoint,
                use_gym_mode=False,
                step_duration=0.0,
                num_episodes=eval_episodes,
                max_steps=eval_max_steps
            )
            if eval_results:
                all_results.append(eval_results)
    
    # 2. Train gym mode with different step durations
    for step_duration in step_durations:
        exp_name = f"gym_mode_{step_duration}s"
        checkpoint_dir = str(exp_dir / "checkpoints" / exp_name)
        metrics_dir = str(exp_dir / "metrics" / exp_name)
        
        checkpoint = run_training_experiment(
            experiment_name=exp_name,
            use_gym_mode=True,
            step_duration=step_duration,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            checkpoint_dir=checkpoint_dir,
            metrics_dir=metrics_dir,
            batch_size=batch_size,
            update_freq=update_freq,
            save_freq=save_freq,
            resume_from_existing=resume_experiment is not None
        )
        
        if checkpoint:
            # Evaluate
            eval_results = run_evaluation(
                experiment_name=exp_name,
                checkpoint_path=checkpoint,
                use_gym_mode=True,
                step_duration=step_duration,
                num_episodes=eval_episodes,
                max_steps=eval_max_steps
            )
            if eval_results:
                all_results.append(eval_results)
    
    # 3. Save comparison results
    results_file = results_dir / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}\n")
    
    # 4. Generate comparison plots
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from compare_experiments import plot_comparison
        plot_comparison(str(results_file), str(results_dir))
    except Exception as e:
        print(f"Warning: Could not generate comparison plots: {e}")
        print("You can generate plots later with:")
        print(f"  python src/compare_experiments.py --results_file {results_file}")
    
    return all_results, str(exp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run delay comparison experiments')
    parser.add_argument('--step_durations', type=float, nargs='+', 
                        default=[0.0, 0.1, 0.5, 1.0],
                        help='Step durations to test in gym mode (default: 0.0 0.1 0.5 1.0)')
    parser.add_argument('--num_episodes', type=int, default=400,
                        help='Number of training episodes per experiment (default: 400)')
    parser.add_argument('--max_steps_per_episode', type=int, default=2000,
                        help='Maximum steps per episode during training (default: 2000)')
    parser.add_argument('--eval_episodes', type=int, default=20,
                        help='Number of episodes for evaluation (default: 20)')
    parser.add_argument('--eval_max_steps', type=int, default=2000,
                        help='Maximum steps per episode during evaluation (default: 2000)')
    parser.add_argument('--base_dir', type=str, default='delay_experiments',
                        help='Base directory for experiments (default: delay_experiments)')
    parser.add_argument('--realtime', action='store_true',
                        help='Include real-time baseline experiment (default: False - skips real-time)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training (default: 256)')
    parser.add_argument('--update_freq', type=int, default=1,
                        help='Update frequency (default: 1)')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Save frequency - checkpoints saved every N episodes (default: 100)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Optional name for this experiment run (default: uses timestamp only)')
    parser.add_argument('--resume_experiment', type=str, default=None,
                        help='Path to existing experiment directory to resume from (e.g., "delay_experiments/experiments_20251211_182526")')
    
    args = parser.parse_args()
    
    results, exp_dir = run_delay_experiments(
        step_durations=args.step_durations,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        base_dir=args.base_dir,
        include_realtime=args.realtime,  # Default is False (skip real-time)
        batch_size=args.batch_size,
        update_freq=args.update_freq,
        save_freq=args.save_freq,
        experiment_name=args.experiment_name,
        resume_experiment=args.resume_experiment
    )
    
    print(f"\nExperiment directory: {exp_dir}")
    print("You can now analyze the results and compare performance!")

