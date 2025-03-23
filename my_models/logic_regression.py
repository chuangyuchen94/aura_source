import numpy as np
from scipy.optimize import minimize

class MyLogisticRegression:
    """
    对于逻辑回归的理解：
    1、特征值X（[m, n]，m行样本值，n列特征），标签值y（有限的枚举值，[m, 1]，对应m行样本值的分类，分类只有1列）==> 对应二元分类;
    2、逻辑回归的本质：找到特征值X与标签值y的映射关系（线性组合）; 接着，再通过sigmoid函数，将X到y的线性组合的输出a，转换为[0, 1]之间的值b;
        1）X与y之间的线性组合，表现为θ的转置与X的点乘积：y=X@T ;
        2）将y值通过sigmoid函数进行映射：z=1/(1+e^(-y)) ;
        3）通过阈值，来将值z分为类别0或1
        4）定义损失函数
        5）通过梯度下降来优化参数θ：用scipy.optimize.minimize方法，实现梯度下降的效果
    3、其他
        1）计算梯度值
    """
    def __init__(self):
        self.theta = None
        pass

    def fit(self, X_train, y_train):
        """
        训练方法
        :return:
        """
        size = X_train.shape[0]
        X_train_extend = np.hstack((np.ones((size, 1)), X_train))
        theta = np.random.randn(size)

        res = minimize(
            fun = MyLogisticRegression.loss_function, # 进行最小化的目标函数：损失函数
            x0 = theta, # 初始化参数θ
            args = (X_train_extend, y_train), # 训练集数据
            method = 'CG', # 优化方法
            jac = MyLogisticRegression.gradient_value, # 用于计算梯度向量的方法
        )

        if not res.success:
            print(f"minimize failed, cause: {res.message}")
        else:
            print(f"minimize success, theta: {res.x}")
            self.theta = res.x

    def predict(self, X):
        """
        预测方法
        :return:
        """
        pass

    @staticmethod
    def sigmoid(X, theta):
        """
        sigmoid方法
        :return:
        """
        sigmoid_value = 1 / (1 + np.exp(-X.dot(theta)))
        return sigmoid_value

    @staticmethod
    def loss_function(theta, X, y):
        """
        损失函数
        :return:
        """
        sigmoid_value = MyLogisticRegression.sigmoid(X, theta)
        size = X.shape[0]
        loss_value = -1 / size * (y.T.dot(np.log(sigmoid_value)) + (1 - y).T.dot(np.log(1 - sigmoid_value)))

        return loss_value

    @staticmethod
    def gradient_value(theta, X, y):
        """
        计算梯度值
        :return:
        """
        sigmoid_value = MyLogisticRegression.sigmoid(X, theta)
        size = X.shape[0]
        gradient_value = 1 / size * X.T.dot(sigmoid_value - y)

        return gradient_value
