import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(200)
y = 2 * x + 3

# ax+by+c=0
# y = 2x + 3; -2x + y -3 = 0，其中，a=-2, b=1
# 2x - y + 3 = 0
# 方向向量：[b, -a] = [1, 2]
# 法向量：[a, b] = [-2, 1]

plt.plot(x, y)
plt.plot([0, 0], [min(y), max(y)])
plt.plot([min(x), max(x)], [0, 0])
plt.scatter([-2,], [1,], c="r") # 法向量
plt.scatter([1,], [2,]) # 方向向量
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
