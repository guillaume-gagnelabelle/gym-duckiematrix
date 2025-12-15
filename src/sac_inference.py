"""
Run inference/evaluation with a trained SAC agent.
No exploration - uses deterministic policy.
"""

import torch
import numpy as np
import argparse
import time
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from gym_duckiematrix.DB21J_gym import DuckiematrixDB21JEnvGym
from sac_agent import SACAgent
from time import sleep


def run_inference(policy_path, q1_path=None, q2_path=None, num_episodes=10, max_steps=2000, 
                  render=True, use_gym_mode=False, step_duration=0.1):
    """
    Run inference with a trained SAC agent.
    
    Args:
        policy_path: Path to saved policy network
        q1_path: Path to saved Q1 network (optional)
        q2_path: Path to saved Q2 network (optional)
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Whether to add small delay for visualization
        use_gym_mode: Whether to use gym mode (faster, non-real-time) (default: False)
        step_duration: Step duration for gym mode in seconds (default: 0.1)
    """
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
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,  # Not used during inference
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        device=device,
    )
    
    # Load checkpoint
    agent.load_checkpoint(policy_path, q1_path, q2_path)
    print(f"Loaded model from {policy_path}")
    print("Running in deterministic mode (no exploration)\n")
    
    # Benchmark forward pass time
    print("Benchmarking forward pass time...")
    forward_times = []
    num_warmup = 10
    num_tests = 10000
    
    # Warmup with varying observations
    for _ in range(num_warmup):
        test_obs = env.observation_space.sample()
        _ = agent.select_action(test_obs, deterministic=True, apply_exploration=False)
    
    # Measure forward pass times with varying observations (more realistic)
    # If using GPU, synchronize to get accurate timing
    if device == 'cuda':
        torch.cuda.synchronize()
    
    for _ in range(num_tests):
        test_obs = env.observation_space.sample()  # Different observation each time
        if device == 'cuda':
            torch.cuda.synchronize()  # Wait for any pending GPU operations
        start_time = time.perf_counter()
        _ = agent.select_action(test_obs, deterministic=True, apply_exploration=False)
        if device == 'cuda':
            torch.cuda.synchronize()  # Wait for GPU operations to complete
        end_time = time.perf_counter()
        forward_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    avg_time = np.median(forward_times)
    std_time = np.std(forward_times)
    min_time = np.min(forward_times)
    max_time = np.max(forward_times)
    
    print(f"Forward pass timing (over {num_tests} runs):")
    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Std Dev: {std_time:.3f} ms")
    print(f"  Min: {min_time:.3f} ms")
    print(f"  Max: {max_time:.3f} ms")
    print(f"  Frequency: {1000/avg_time:.1f} Hz (actions per second)\n")
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    action_times = []  # Track action selection times during episodes
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        for step in range(max_steps):
            # Select action deterministically (no exploration) and measure time
            if device == 'cuda':
                torch.cuda.synchronize()  # Wait for any pending GPU operations
            action_start = time.perf_counter()
            action = agent.select_action(obs, deterministic=True, apply_exploration=False)
            if device == 'cuda':
                torch.cuda.synchronize()  # Wait for GPU operations to complete
            action_end = time.perf_counter()
            action_time = (action_end - action_start) * 1000  # Convert to milliseconds
            action_times.append(action_time)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Only add delay in regular mode (gym mode handles timing internally)
            if render and not use_gym_mode:
                sleep(0.01)  # Small delay for visualization
            
            # Check if episode is done
            if terminated or truncated:
                print(f"  Terminated at step {step + 1}")
                break
            
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}\n")
    
    # Calculate summary statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    max_reward = np.max(episode_rewards)
    min_reward = np.min(episode_rewards)
    
    # Print summary
    print("=" * 50)
    print("Inference Summary:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Avg Length: {avg_length:.1f} ± {std_length:.1f}")
    print(f"  Max Reward: {max_reward:.2f}")
    print(f"  Min Reward: {min_reward:.2f}")
    print()
    print("Action Selection Timing (during episodes):")
    if action_times:
        print(f"  Average: {np.median(action_times):.3f} ms")
        print(f"  Std Dev: {np.std(action_times):.3f} ms")
        print(f"  Min: {np.min(action_times):.3f} ms")
        print(f"  Max: {np.max(action_times):.3f} ms")
        print(f"  Total actions: {len(action_times)}")
    print("=" * 50)
    
    # Cleanup
    env.robot.motors.stop()
    
    # Return metrics dictionary for easier parsing
    metrics = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "avg_reward": float(avg_reward),
        "reward_std": float(std_reward),
        "avg_length": float(avg_length),
        "length_std": float(std_length),
        "max_reward": float(max_reward),
        "min_reward": float(min_reward),
        "num_episodes": num_episodes,
    }
    
    return episode_rewards, episode_lengths, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with trained SAC agent')
    parser.add_argument('--policy_checkpoint', type=str, required=True,
                        help='Path to policy checkpoint (required)')
    parser.add_argument('--q1_checkpoint', type=str, default=None,
                        help='Path to Q1 checkpoint (optional)')
    parser.add_argument('--q2_checkpoint', type=str, default=None,
                        help='Path to Q2 checkpoint (optional)')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to run (default: 10)')
    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Maximum steps per episode (default: 2000)')
    parser.add_argument('--no_render', action='store_true',
                        help='Disable rendering delay (faster)')
    parser.add_argument('--gym_mode', action='store_true',
                        help='Use gym mode (faster, non-real-time simulation)')
    parser.add_argument('--step_duration', type=float, default=0.1,
                        help='Step duration for gym mode in seconds (default: 0.1)')
    parser.add_argument('--save_metrics', action='store_true',
                        help='Save evaluation metrics to JSON file')
    
    args = parser.parse_args()
    
    # Run inference
    rewards, lengths, metrics = run_inference(
        policy_path=args.policy_checkpoint,
        q1_path=args.q1_checkpoint,
        q2_path=args.q2_checkpoint,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render=not args.no_render,
        use_gym_mode=args.gym_mode,
        step_duration=args.step_duration
    )
    
    # Optionally save metrics to file
    if args.save_metrics:
        import json
        from pathlib import Path
        
        metrics_file = Path(args.policy_checkpoint).parent / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nEvaluation metrics saved to: {metrics_file}")

