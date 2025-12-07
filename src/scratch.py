import numpy as np

q = np.load("Q_advanced.npy")


for i in range(8):
    for j in range(8):
        print(f"------------------------------ (i={i}, j={j}) ------------------------------")
        print(np.around(q[i, j], 2))
