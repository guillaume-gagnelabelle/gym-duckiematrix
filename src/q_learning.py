from sklearn.preprocessing import KBinsDiscretizer
import argparse
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
import matplotlib.pyplot as plt
from q_learning_utils import *
from q_policy_utils import *

"""
 * @author Guillaume Gagné-Labelle, Gabriel Sasseville, Nico Bosteels
 * @date Dec 23, 2025
 * @project RL1 - Final Project - IFT6757 - UdeM
 *
 * @description: This file is the main loop of the algorithm. It's mostly there to gather training and testing data
 *              rather than implementing any RL logic.
"""

DELAY = 0

# This should be elsewhere
tile_size = 0.585
lane_width = 0.585 / 2

# Observation space
low_state =  [-lane_width/2, -np.pi / 2, 0]
high_state = [lane_width/2, np.pi/2, 1]

args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument('--policy', type=str, choices=["physics","human","Q"], default="Q")
args_form.add_argument('--test', action="store_true", default=False)
args_form.add_argument('--load_policy', type=str)
#args_form.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4,5,6,7,8,9])
args_form.add_argument("--seeds", type=int, nargs='+', default=[0])
args = args_form.parse_args()

for seed in args.seeds:

    experiment_starting_time = time()
    env = DuckiematrixDB21JEnv()
    np.random.seed(seed)
    observation, info = env.reset()

    n_bins = (7, 7, 2)
    n_actions = N_ACTIONS
    bin_finder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    bin_finder.fit([low_state, high_state])
    
    Q_table = np.zeros(n_bins + (n_actions,))
    # euristics. Those states are near failure. Thus, set their expectation to out_of_road penalty
    Q_table[:2, :2, :] = env.out_of_road_penalty
    Q_table[-2:, -2:, :] = env.out_of_road_penalty

    if args.load_policy is not None:
        Q_table = np.load(args.load_policy)
        expected_shape = n_bins + (n_actions,)
        if Q_table.shape != expected_shape:
            raise ValueError(f"Loaded Q-table shape {Q_table.shape} does not match expected {expected_shape}.")
    
    # Reward
    y_train, y_test, y_train_mean, y_train_std = [], [], [], []
    # Distance
    dist_train, dist_test, dist_train_mean, dist_train_std = [], [], [], []
    # Time
    time_test = []
    x_train, x_test = [], []

    print("---------------------- SEED %d BEGINNING -------------------" % seed)

    n_episodes = 2500
    for e in range(n_episodes + 1):
        info_msg = "\nEpisode: %d\n" % e
        if not args.test:
            Q_table, train_reward, train_distance = episode(args=args, env=env, Q_table=Q_table, episode=e, bin_finder=bin_finder, n_actions=n_actions, testing=False)
            y_train.append(train_reward)
            dist_train.append(train_distance)
            if e % 5 == 0:
                x_train.append(e)
                y_train_mean.append(np.array(y_train).mean())
                y_train_std.append(np.array(y_train).std())
                dist_train_mean.append(np.array(dist_train).mean())
                dist_train_std.append(np.array(dist_train).std())

                info_msg += "Avg training reward: %.2f" % (np.array(y_train_mean)[-1])
                info_msg += " | Avg training distance: %.2f" % (np.array(dist_train_mean)[-1])

            if e % 100 == 0:
                np.save("Q_intermediate", Q_table)
                print_policy(Q_table, n_bins)
            if e == n_episodes:
                np.save("Q_advanced", Q_table)
        else:
            y_train = [0]
            dist_train = [0]

        if e % 25 == 0:
            episode_start_time = time()
            _, test_reward, test_distance = episode(args=args, env=env, Q_table=Q_table, episode=e, bin_finder=bin_finder, n_actions=n_actions, testing=True)
            episode_end_time = time()
            x_test.append(e)
            y_test.append(test_reward)
            dist_test.append(test_distance)
            time_test.append(episode_end_time - episode_start_time)

            info_msg += "\nTesting reward: %.2f" % test_reward
            info_msg += " | Testing distance: %.2f" % test_distance

        if e % 5 == 0:
            print(info_msg)
            y_train = []
            dist_train = []


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

    np.save("experiment_%.2f_info"%(DELAY), info)

'''
    if not args.test:
        y_train_mean = np.array(y_train_mean)
        y_train_std = np.array(y_train_std)
        plt.plot(x_train, y_train_mean, label="Entraînement")
        plt.fill_between(x_train, y_train_mean-y_train_std, y_train_mean+y_train_std, alpha=0.3)

    plt.plot(x_test, y_test, label="Évaluation")

    plt.title("Performance d'un agent")
    plt.xlabel("Épisode")
    plt.ylabel("Récompense")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    if not args.test:
        dist_train_mean = np.array(dist_train_mean)
        dist_train_std = np.array(dist_train_std)
        plt.plot(x_train, dist_train_mean, label="Entraînement")
        plt.fill_between(x_train, dist_train_mean-dist_train_std, dist_train_mean+dist_train_std, alpha=0.3)

    plt.plot(x_test, dist_test, label="Évaluation")

    plt.title("Performance d'un agent")
    plt.xlabel("Épisode")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid()
    plt.show()
    print("----------------------- SEED %d END ------------------------" % seed)
'''