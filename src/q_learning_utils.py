import numpy as np
import math
from time import sleep, time
from q_policy_utils import *

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


def Q_policy(Q_table, state):
    action_idx = int(np.argmax(Q_table[state]))
    return action_idx, action_from_index(action_idx)


def learning_rate(n, min_rate=0.01):
    return min_rate    # I loaded a pre-trained policy. I don't want to overwrite everything
    #return max(min_rate, min(1., 1. - math.log10((n + 1) / 75)))


def exploration_rate(n, min_rate=0.1):
    return min_rate    # I loaded a pre-trained policy. I don't want to overwrite everything
    #return max(min_rate, min(1., 1.0 - math.log10((n + 1) / 150)))


def episode(args, env, Q_table, bin_finder, episode, n_actions, testing=False):
    sum_of_reward = 0
    distance = 0
    counter = 0
    
    if args.policy == "Q":
        if testing:
            current_state, terminated, truncated = discretizer(env.reset(position=[3 * 0.585 / 2, 0.585/4, 0])[0], bin_finder), False, False
        else:
            obs = env.reset()[0]
            current_state, terminated, truncated = discretizer(obs, bin_finder), False, False

    else: 
        current_state, terminated, truncated = env.reset(), False, False
    sleep(0.1)

    while not terminated and not truncated and counter < 300:

        if args.policy == "Q":
            if np.random.random() >= exploration_rate(episode) or testing: 
                action_idx, action = Q_policy(Q_table, current_state)   # 3e-5
            else: 
                action_idx = np.random.randint(low=0, high=n_actions)
                action = action_from_index(action_idx)
        elif args.policy == "physics":
            action = physics_policy(current_state)
        elif args.policy == "human":
            action = human_policy()
        else: raise Exception

        observation, reward, terminated, truncated, info = env.step(action) # 1e-4
        sleep(0.1)  # 1e-1

        new_state = discretizer(observation, bin_finder)

        sum_of_reward += reward
        distance += info["distance"]
        counter += 1

        if args.policy == "Q":
            if not testing and not truncated:
                idx = current_state + (action_idx,)
                lr = learning_rate(episode)
                learnt_value = new_Q_value(Q_table, reward, new_state, terminated=terminated)
                old_value = Q_table[idx]
                Q_table[idx] = (1 - lr) * old_value + lr * learnt_value
            current_state = new_state
        else:
            current_state = observation
    return Q_table, sum_of_reward, distance
