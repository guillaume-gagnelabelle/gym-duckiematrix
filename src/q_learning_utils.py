import numpy as np
import math
from time import sleep

"""
 * @author Guillaume Gagné-Labelle, Gabriel Sasseville, Nico Bosteels
 * @date Dec 23, 2025
 * @project RL1 - Final Project - IFT6757 - UdeM
 *
 * @description: The core of this file is the episode() function at the end. It loops, computes the actions given a policy,
 *               calls the step() function, and update the Q_table until termination. 
"""


def physics_policy(observation):
    pass


def human_policy():
    pass

##################################### Q-POLICY & Q-POLICY UTILS ################################################
# (utils should be elsewhere)

def Q_policy(Q_table, state, action_bins):
    i, j = np.unravel_index(np.argmax(Q_table[state]), action_bins)

    i_max = Q_table[state].shape[0] - 1
    j_max = Q_table[state].shape[1] - 1

    while i < i_max and j < j_max:  # speeeeed (this basically reduce the action space from 9D to 5D in the 3x3 case)
        i += 1
        j += 1
    assert i == i_max or j == j_max

    middle_left = (i + 1) / action_bins[0]    # middle of the bin
    middle_right = (j + 1) / action_bins[1]
    return [i, j], [middle_left, middle_right]


def new_Q_value(Q_table, reward, state_new, terminated=False, discount_factor=0.995):
    if terminated:
        return reward
    return reward + discount_factor * np.max(Q_table[state_new])


def update_slower_state(Q, state, bin_idx, target):
    i_max, j_max = Q[state].shape[0] - 1, Q[state].shape[1] - 1
    
    i, j = bin_idx
    while i < i_max and j < j_max:
        Q[state][i+1, j+1] = target
        i+=1
        j+=1
    
    i, j = bin_idx
    while i > 0 and j > 0:
        Q[state][i-1, j-1] = target
        i-=1
        j-=1

    return Q


def learning_rate(n, min_rate=0.01):
    return min_rate    # I loaded a pre-trained policy. I don't want to overwrite everything
    return max(min_rate, min(1., 1. - math.log10((n + 1) / 75)))


def exploration_rate(n, min_rate=0.1):
    return min_rate    # I loaded a pre-trained policy. I don't want to overwrite everything
    return max(min_rate, min(1., 1.0 - math.log10((n + 1) / 150)))


def print_policy(Q, dims):
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                print(f"---------- (i={i}, j={j}, k={k}) ----------")
                print(np.around(Q[i, j, k], 2))


def discretizer(observation, est):
    d, theta, in_curve = observation[:]
    return tuple(map(int, est.transform([[d, theta, in_curve]])[0]))
####################################### END OF Q-POLICY UTILS ##########################################################

def episode(args, env, Q_table, est, episode, action_bins, testing=False):
    sum_of_reward = 0
    counter = 0
    
    if args.policy == "Q":
        if testing:
            current_state, terminated, truncated = discretizer(env.reset(position=[3 * 0.585 / 2, 0.585/4, 0])[0], est), False, False
        else:
            obs = env.reset()[0]
            current_state, terminated, truncated = discretizer(obs, est), False, False

    else: 
        current_state, terminated, truncated = env.reset(), False, False
    sleep(0.1)

    while not terminated and not truncated and counter < 300:

        if args.policy == "Q":
            if np.random.random() >= exploration_rate(episode) or testing: 
                bin_idx, action = Q_policy(Q_table, current_state, action_bins)
            else: 
                bin_idx = np.random.randint(low=0, high=action_bins[0], size=2)
                action = (bin_idx + 1) / action_bins
        elif args.policy == "physics":
            action = physics_policy(current_state)
        elif args.policy == "human":
            action = human_policy()
        else: raise Exception

        observation, reward, terminated, truncated, info = env.step(action)
        sleep(0.1)

        new_state = discretizer(observation, est)

        sum_of_reward += reward
        counter += 1

        if args.policy == "Q":
            if not testing and not truncated:
                idx = current_state + tuple(bin_idx)
                lr = learning_rate(episode)
                learnt_value = new_Q_value(Q_table, reward, new_state, terminated=terminated)
                old_value = Q_table[idx]
                Q_table[idx] = (1 - lr) * old_value + lr * learnt_value
                Q_table = update_slower_state(Q_table, current_state, bin_idx, Q_table[idx])
            current_state = new_state
        else:
            current_state = observation
    return Q_table, sum_of_reward