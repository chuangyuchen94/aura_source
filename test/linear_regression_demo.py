import numpy as np
import os
# %matplotlib inline
import matplotlib.pyplot as plt

plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

#  创建模拟数据
X = np.random.rand(100, 1) * 2
y = 4 + X * 3 + np.random.randn(100, 1)

plt.plot(X, y, "b.")
plt.xlabel("X_1")
plt.ylabel("y")
plt.axis([0, 2, 0, 15])
plt.show()
