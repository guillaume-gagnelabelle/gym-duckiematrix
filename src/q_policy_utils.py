import math
import numpy as np


# Discrete action mapping for the Q-policy. Each entry is (left_pwm, right_pwm).
ACTION_MAP = np.array([
    [0.25, 1.00],
    [0.50, 1.00],
    [0.75, 1.00],
    [1.00, 1.00],
    [1.00, 0.75],
    [1.00, 0.50],
    [1.00, 0.25],
], dtype=np.float32)

N_ACTIONS = len(ACTION_MAP)


def action_from_index(action_idx: int) -> np.ndarray:
    return ACTION_MAP[action_idx]


def new_Q_value(Q_table, reward, state_new, terminated=False, discount_factor=0.995):
    if terminated:
        return reward
    return reward + discount_factor * np.max(Q_table[state_new])


def print_policy(Q, dims):
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                print(f"---------- (i={i}, j={j}, k={k}) ----------")
                print(np.around(Q[i, j, k], 2))
                print()


def discretizer(observation, bin_finder):
    d, theta, in_curve = observation[:]
    return tuple(map(int, bin_finder.transform([[d, theta, in_curve]])[0]))
