import argparse
import numpy as np
import matplotlib.pyplot as plt

args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument("--delay", type=str, default='0')
args = args_form.parse_args()

delay = ["0", "05", "1", "15", "2", "25", "3"]
n = 6       # keep one point out of n
step = n

for d in delay:
    info = np.load(f"experiment_delay{d}_info.npy", allow_pickle=True).item()

    x_test = info["x_test"][::step]
    dist_test = info["dist_test"][::step]

    plt.plot(x_test, dist_test, label=f"Q{d}")

plt.title("Training Evolution")
plt.xlabel("Number of Episodes")
plt.ylabel("Distance [m]")
plt.legend()
plt.grid()
plt.show()
