from linear_regression import MYLinearRregression
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据（标准化后）
np.random.seed(42)
m = 100  # 样本数量
X = 2 * np.random.rand(m, 1)  # 特征
y = 4 + 3 * X + np.random.randn(m, 1)  # 标签（含噪声）

# 标准化特征
X_mean = np.mean(X)
X_std = np.std(X)
X_norm = (X - X_mean) / X_std

# 添加截距项（全1列）
X_b = np.c_[np.ones((m, 1)), X_norm]

# 不同初始化方法对比
theta_init_zero = np.zeros((2, 1))          # 零初始化
theta_init_random = np.random.randn(2, 1) * 0.01  # 小随机数初始化

# 训练参数
learning_rate = 0.1
n_epochs = 1000

linear_regression_random = MYLinearRregression(learning_rate=learning_rate,
                                               iteration_num=n_epochs,
                                               init_params=theta_init_random)

# 运行梯度下降
# theta_zero, losses_zero = gradient_descent(X_b, y, theta_init_zero, learning_rate, n_epochs)
linear_regression_random.fit(X, y)
pred = linear_regression_random.predict(X)
loss_values = linear_regression_random.get_loss_values()

# 绘制损失曲线
# plt.plot(losses_zero, label="Zero Initialization")
def show_loss_convergence(losses_random):
    plt.plot(losses_random, label="Random Initialization")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.title("Loss Convergence with Different Initializations")
    plt.show()

def show_linear_convergence(X, y, pred):
    plt.scatter(X, y, label="Random Initialization")
    plt.plot(X, pred, 'r')
    plt.xlabel("X")
    plt.ylabel("predict-y")
    plt.legend()
    plt.title("Loss Convergence with Different Initializations")
    plt.show()

def show_loss_values(x, loss_values):
    plt.plot(x, loss_values, 'r')
    plt.xlabel("Training times")
    plt.ylabel("Loss value")
    plt.legend()
    plt.title("loss values of gradient descend")
    plt.show()

# 输出最终参数
# print("零初始化最终参数：", theta_zero.ravel())
print("随机初始化最终参数：", linear_regression_random.get_best_params())

show_linear_convergence(X, y, pred)
show_loss_values(range(len(loss_values)), loss_values)
