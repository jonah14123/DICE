import numpy as np
import matplotlib.pyplot as plt

L = np.load("mu3/losses_mu3.npy")
plt.plot(np.convolve(L, np.ones(50)/50, mode='valid'))
plt.xlabel("iteration"); plt.ylabel("DICE loss (smoothed)")
plt.show()