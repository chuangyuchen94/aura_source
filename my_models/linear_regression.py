import numpy as np

class MYLinearRregression:
    """
    初始化方法
    """
    def __init__(self, learning_rate=0.001, iteration_num=1000):
        self.learning_rate = learning_rate
        self.iteration_num = iteration_num

    def fit(self, X, y):
        """
            训练方法：使用梯度下降，计算出最优参数θ，以此构建模型
            1、得到参数的初始值
            2、计算梯度值
            3、沿负梯度方向更新参数值
            4、重复2~3步，直到梯度值为0，或小于指定值
        """
        num_of_features = X.shape[1]
        params = np.random.randn(num_of_features, 1) # (n, 1)数组


        for _ in range(self.iteration_num):
            gradient_value = MYLinearRregression.calc_gradient(X, y, params)

            if MYLinearRregression.is_equal_target(gradient_value):
                self.best_param = params

            params = params - self.learning_rate * gradient_value

    @staticmethod
    def is_equal_target(grandient_value, target=0):
        return abs(grandient_value - target) < 1e-5

    @staticmethod
    def calc_gradient(X, y, params_theta):
        """
        计算梯度值
        """
        num_of_sample = X.shape[0] # 样本个数
        return X.T.dot(X.dot(params_theta) - y) / num_of_sample


    def predict(self, X):
        """
        预测方法
        """
        return X.dot(self.best_param)
