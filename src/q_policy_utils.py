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

def save_info(x_train, x_test, y_train_mean, y_train_std, y_test, dist_train_mean, dist_train_std, dist_test, time_test, DELAY):
    info = {}
    info["x_train"] = x_train
    info["x_test"] = x_test

    info["y_train_mean"] = y_train_mean
    info["y_train_std"] = y_train_std
    info["y_test"] = y_test

    info["dist_train_mean"] = dist_train_mean
    info["dist_train_std"] = dist_train_std
    info["dist_test"] = dist_test

    info["episode_time"] = time_test

    np.save("experiment_delay%d_t05"%(DELAY), info)
