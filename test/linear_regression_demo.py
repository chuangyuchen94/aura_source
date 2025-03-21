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

X_train = np.insert(X, 0, 1, axis=1)
theta = np.random.randn(2, 1)
print(f"X_train.shape = {X_train.shape}")
print(f"theta = {theta.shape}")

# 批量梯度下降
def batch_descend():
    learning_rate = 0.1
    n_estimators = 1000
    count_of_sample = X_train.shape[0]
    theta = np.random.randn(2, 1)
    for _ in range(n_estimators):
        gradient_value = (1/count_of_sample) * X_train.T.dot(X_train.dot(theta) - y)
        theta -= learning_rate * gradient_value

    print(f"best theta of batch_descend: {theta}")

# 随机梯度下降: 在迭代过程中，随机选择一个样本
def randoom_descend():
    learning_rate = 0.01
    n_interators = 1000
    count_of_sample = X_train.shape[0]
    theta = np.random.randn(2, 1)

    for _ in range(n_interators):
        random_index = np.random.randint(0, count_of_sample)
        random_X = np.array([X_train[random_index]])
        random_y = np.array([y[random_index]])

        grandient_value = random_X.T.dot(random_X.dot(theta) - random_y)
        theta -= learning_rate * grandient_value

    print(f"best theta of random_descend: {theta}")

if __name__ == "__main__":
    batch_descend()
    randoom_descend()