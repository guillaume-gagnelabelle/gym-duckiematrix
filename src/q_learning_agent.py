"""
Tabular Q-learning agent for the Duckiematrix DB21J environment.

Compared to PPO / REINFORCE, this agent discretizes both the observation
space (signed lane offset + heading error) and the continuous wheel commands.
This dramatically reduces the learning search space and makes it easier
to recover good lane-keeping policies from a small amount of experience.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from duckietown.sdk.utils.loop_lane_position import get_closest_tile
from time import sleep


# Discretization helpers -----------------------------------------------------

def build_bins(low: float, high: float, buckets: int) -> np.ndarray:
    """Return monotonically increasing split points for np.digitize."""
    if buckets < 2:
        raise ValueError("Need at least 2 buckets to discretize a range.")
    return np.linspace(low, high, buckets - 1, dtype=np.float32)


DISTANCE_BINS = build_bins(low=-0.30, high=0.30, buckets=11)  # 10 segments
THETA_BINS = build_bins(low=-np.pi, high=np.pi, buckets=21)   # 20 segments

# Action set is expressed as (left_wheel, right_wheel) PWM commands in [-1, 1]
ACTION_SET = np.array(
    [
        [0.65, 0.65],   # fast forward
        [0.45, 0.45],   # cruise forward
        [0.25, 0.25],   # slow forward
        [0.60, 0.20],   # gentle left
        [0.20, 0.60],   # gentle right
        [0.55, -0.10],  # pivot left
        [-0.10, 0.55],  # pivot right
        [-0.30, -0.30], # short reverse to re-center
    ],
    dtype=np.float32,
)


@dataclass
class ObservationDiscretizer:
    """Map continuous (d_signed, theta) observations to discrete bins."""

    distance_bins: np.ndarray
    theta_bins: np.ndarray

    def encode(self, obs: np.ndarray) -> Tuple[int, int]:
        safe_obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        d_signed, theta = float(safe_obs[0]), float(safe_obs[1])
        d_idx = int(np.clip(np.digitize(d_signed, self.distance_bins), 0, len(self.distance_bins)))
        theta_idx = int(np.clip(np.digitize(theta, self.theta_bins), 0, len(self.theta_bins)))
        return d_idx, theta_idx


class DiscreteQLearningAgent:
    """Tabular Q-learning over discretized DB21J observations and actions."""

    def __init__(
        self,
        dist_bins: np.ndarray,
        theta_bins: np.ndarray,
        actions: np.ndarray,
        alpha: float = 0.2,
        gamma: float = 0.995,
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.discretizer = ObservationDiscretizer(dist_bins, theta_bins)
        table_shape = (len(dist_bins) + 1, len(theta_bins) + 1, len(actions))
        self.q_table = np.zeros(table_shape, dtype=np.float32)

    def select_action(self, state_idx: Tuple[int, int], epsilon: float) -> Tuple[int, np.ndarray]:
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(len(self.actions))
        else:
            d_idx, theta_idx = state_idx
            action_idx = int(np.argmax(self.q_table[d_idx, theta_idx]))
        return action_idx, self.actions[action_idx]

    def update(
        self,
        state_idx: Tuple[int, int],
        action_idx: int,
        reward: float,
        next_state_idx: Tuple[int, int],
        done: bool,
    ) -> None:
        d_idx, theta_idx = state_idx
        td_target = reward
        if not done:
            next_d, next_theta = next_state_idx
            td_target += self.gamma * np.max(self.q_table[next_d, next_theta])
        td_error = td_target - self.q_table[d_idx, theta_idx, action_idx]
        self.q_table[d_idx, theta_idx, action_idx] += self.alpha * td_error


# Training loop --------------------------------------------------------------

def train_q_learning(
    num_episodes: int = 1200,
    max_steps_per_episode: int = 800,
    alpha: float = 0.2,
    gamma: float = 0.995,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    curve_prob: float = 0.5,
) -> Tuple[DiscreteQLearningAgent, list, list]:
    """Train a discrete Q-learning agent on the Duckiematrix environment."""

    env = DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0")
    agent = DiscreteQLearningAgent(
        dist_bins=DISTANCE_BINS,
        theta_bins=THETA_BINS,
        actions=ACTION_SET,
        alpha=alpha,
        gamma=gamma,
    )

    epsilon = epsilon_start
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    reset_tile = None

    print("Starting Q-learning training loop...")
    print(f"Observation bins: distance={len(DISTANCE_BINS)+1}, theta={len(THETA_BINS)+1}")
    print(f"Discrete action count: {len(ACTION_SET)}")

    for episode in range(num_episodes):
        if reset_tile is not None:
            obs, info = env.reset(tile=reset_tile)
            reset_tile = None
        else:
            obs, info = env.reset(curve_prob=curve_prob)

        state_idx = agent.discretizer.encode(obs)
        ep_reward = 0.0
        ep_length = 0
        last_pose = None

        for step in range(max_steps_per_episode):
            action_idx, action = agent.select_action(state_idx, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            next_state_idx = agent.discretizer.encode(next_obs)
            agent.update(state_idx, action_idx, reward, next_state_idx, done)

            ep_reward += reward
            ep_length += 1

            if not terminated:
                last_pose = info.get("pose")

            state_idx = next_state_idx

            if done:
                if terminated:
                    terminated_pos = info.get("terminated_position")
                    if terminated_pos is None and last_pose is not None:
                        pose = last_pose["position"]
                        terminated_pos = (pose["x"], pose["y"], 0.0)
                    if terminated_pos is not None:
                        x, y, _ = terminated_pos
                        reset_tile = get_closest_tile(x, y)
                break

            sleep(0.01)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % 20 == 0:
            avg_reward = float(np.mean(episode_rewards[-20:]))
            avg_length = float(np.mean(episode_lengths[-20:]))
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Avg Reward (last 20): {avg_reward:.2f} | "
                f"Avg Len: {avg_length:.1f}"
            )

    env.robot.camera.stop()
    env.robot.motors.stop()
    return agent, episode_rewards, episode_lengths


def save_q_table(agent: DiscreteQLearningAgent, path: str = "q_learning_table.npy") -> None:
    np.save(path, agent.q_table)
    print(f"Saved Q-table to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning agent for Duckiematrix.")
    parser.add_argument("--episodes", type=int, default=1200, help="Number of training episodes.")
    parser.add_argument("--max_steps", type=int, default=800, help="Max steps per episode.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor.")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial exploration rate.")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final exploration rate.")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Multiplicative epsilon decay per ep.")
    parser.add_argument("--curve_prob", type=float, default=0.5, help="Probability reset selects a curve tile.")

    args = parser.parse_args()

    agent, rewards, lengths = train_q_learning(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        curve_prob=args.curve_prob,
    )

    save_q_table(agent)

