from sklearn.preprocessing import KBinsDiscretizer
import argparse
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
import matplotlib.pyplot as plt
from q_learning_utils import *

"""
 * @author Guillaume Gagné-Labelle, Gabriel Sasseville, Nico Bosteels
 * @date Dec 23, 2025
 * @project RL1 - Final Project - IFT6757 - UdeM
 *
 * @description: This file is the main loop of the algorithm. It's mostly there to gather training and testing data
 *              rather than implementing any RL logic.
"""

# This should be elsewhere
tile_size = 0.585
lane_width = 0.585 / 2

# Observation space
low_state =  [-lane_width/2, -np.pi / 2, 0]
high_state = [lane_width/2, np.pi/2, 1]

args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument('--policy', type=str, choices=["physics","human","Q"], default="Q")
args_form.add_argument('--render_mode', type=str, choices=["rgb_array","human","quick_human"], default="rgb_array")
args_form.add_argument('--test', action="store_true", default=False)
args_form.add_argument('--load_policy', type=str)
#args_form.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4,5,6,7,8,9])
args_form.add_argument("--seeds", type=int, nargs='+', default=[0])
args = args_form.parse_args()

for seed in args.seeds:

    env = DuckiematrixDB21JEnv()
    np.random.seed(seed)

    # This should be elsewhere
    start_x = lane_width / 2
    start_y = lane_width/2
    start_yaw = 0.
    desired_reset = (start_x, start_y, start_yaw)

    observation, info = env.reset()
    n_bins = (7, 7, 2)
    action_bins = (3, 3)
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([low_state, high_state])
    Q_table = np.zeros(n_bins + action_bins)
    if args.load_policy is not None:
        Q_table = np.load(args.load_policy)
    y_train, y_test, y_train_mean, y_test_mean, y_train_std, y_test_std = [], [], [], [], [], []
    x_train, x_test = [], []

    print("---------------------- SEED %d BEGINNING -------------------" % seed)

    n_episodes = 3500
    for e in range(n_episodes + 1):
        info_msg = "Episode: %d" % e
        if not args.test:
            Q_table, train_reward = episode(args=args, env=env, Q_table=Q_table, episode=e, est=est, action_bins=action_bins, testing=False)
            y_train.append(train_reward)
            if e % 5 == 0:
                x_train.append(e)
                y_train_mean.append(np.array(y_train).mean())
                y_train_std.append(np.array(y_train).std())
                info_msg += " | Avg training reward: %.2f" % (np.array(y_train_mean)[-1])

            if e % 100 == 0:
                np.save("Q_intermediate", Q_table)
                print_policy(Q_table, n_bins)
            if e == n_episodes:
                np.save("Q_advanced", Q_table)
        else:
            y_train = [0]

        if e % 25 == 0:
            _, test_reward = episode(args=args, env=env, Q_table=Q_table, episode=e, est=est, action_bins=action_bins, testing=True)
            y_test.append(test_reward)
            x_test.append(e)
            info_msg += " | testing reward: %.2f" % test_reward

        if e % 5 == 0:
            print(info_msg)
            y_train = []


    if not args.test:
        y_train_mean = np.array(y_train_mean)
        y_train_std = np.array(y_train_std)
        plt.plot(x_train, y_train_mean, label="Entraînement")
        plt.fill_between(x_train, y_train_mean-y_train_std, y_train_mean+y_train_std, alpha=0.3)

    plt.plot(x_test, y_test, label="Évaluation")

    plt.title("Performance d'un agent")
    plt.xlabel("Épisode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.show()
    print("----------------------- SEED %d END ------------------------" % seed)