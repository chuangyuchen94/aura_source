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
# theta = np.random.randn(2, 1)
print(f"X_train.shape = {X_train.shape}")
# print(f"theta = {theta.shape}")

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


# 学习率衰减函数
def decay_learning_rate(learning_rate, n_iterations, current_iteration):
    return learning_rate * (1 / (n_iterations * current_iteration + 1))

# 随机梯度下降: 在迭代过程中，随机选择一个样本
def randoom_descend():
    learning_rate = 0.01
    n_interators = 1000
    count_of_sample = X_train.shape[0]
    theta = np.random.randn(2, 1)

    for _ in range(n_interators):
        for i in range(count_of_sample):
            random_index = np.random.randint(0, count_of_sample)
            random_X = np.array([X_train[random_index]])
            random_y = np.array([y[random_index]])

            grandient_value = random_X.T.dot(random_X.dot(theta) - random_y)
            theta -= learning_rate * grandient_value

    print(f"best theta of random_descend: {theta}")

def mini_batch_descend():
    learning_rate = 0.01
    mini_batch_size = 16
    sample_size = X_train.shape[0]
    mini_iterators = sample_size // mini_batch_size

    n_interators = 500
    theta = np.random.randn(2, 1)

    for curr_iter in range(n_interators):
        for i in range(mini_iterators):
            mini_index = np.random.randint(0, sample_size, size=mini_batch_size)
            mini_X = X_train[mini_index]
            mini_y = y[mini_index]

            mini_gradient_value = 1/mini_batch_size *mini_X.T.dot(mini_X.dot(theta) - mini_y)
            theta -= learning_rate * mini_gradient_value

            #learning_rate = decay_learning_rate(learning_rate, n_interators, curr_iter)


    print(f"best theta of mini_batch_descend: {theta}")

if __name__ == "__main__":
    batch_descend()
    randoom_descend()
    mini_batch_descend()