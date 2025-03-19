import numpy as np

class MYLinearRregression:
    """
    初始化方法
    """
    def __init__(self, learning_rate=0.001, iteration_num=1000, init_params=None):
        self.loss_values = None
        self.best_param = None
        self.learning_rate = learning_rate
        self.iteration_num = iteration_num
        self.init_params = init_params

    def fit(self, X, y):
        """
            训练方法：使用梯度下降，计算出最优参数θ，以此构建模型
            1、得到参数的初始值
            2、计算梯度值
            3、沿负梯度方向更新参数值
            4、重复2~3步，直到梯度值为0，或小于指定值
        """
        num_of_features = X.shape[0]
        if self.init_params is None:
            params = np.random.randn(X.shape[1]+1, 1) * 0.01  # 小随机数初始化(n+1, 1)数组
        else:
            params = self.init_params

        extend_X = np.insert(X, 0, 1, axis=1) # 在X第1列前面增加一列，作为截距项的处理
        loss_values = []

        for i in range(self.iteration_num):
            gradient_value = MYLinearRregression.calc_gradient(extend_X, y, params)
            loss_value = MYLinearRregression.calc_loss_value(extend_X, y, params)

            print(f"第{i}次迭代 ==> 参数值：{params} | 梯度值：{gradient_value} | 梯度范数：{np.linalg.norm(gradient_value)} | 损失值：{loss_value}")
            loss_values.append(loss_value[0][0])

            if MYLinearRregression.is_equal_target(gradient_value):
                self.best_param = params.copy()
                print("梯度值为0，迭代结束")
                break

            self.best_param = params.copy()
            params -= self.learning_rate * gradient_value

        self.loss_values = loss_values

    @staticmethod
    def is_equal_target(grandient_value, target=0):
        """
        计算梯度范数，再计算与目标值的差异
        :param grandient_value:
        :param target:
        :return:
        """
        norm_value_of_gradient = np.linalg.norm(grandient_value)
        return abs(norm_value_of_gradient - target) < 1e-5

    @staticmethod
    def calc_loss_value(X, y, params_theta):
        """
        计算损失值
        :return:
        """
        return 0.5 * (X.dot(params_theta)-y).T.dot(X.dot(params_theta)-y) / X.shape[0]

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
        extend_X = np.insert(X, 0, 1, axis=1)
        return extend_X.dot(self.best_param)

    def get_best_params(self):
        """
        返回最优参数
        :return:
        """
        return self.best_param

    def get_loss_values(self):
        """
        获取训练过程中的损失值
        :return:
        """
        return self.loss_values

    @staticmethod
    def calc_r2(y_true, y_pred):
        """
        计算R^2分数
        :param y_true: 真实值
        :param y_pred: 预测值
        :return: R^2值
        """
        ss_res = (y_true - y_pred).T.dot(y_true - y_pred)
        y_true_mean = y_true.mean()
        ss_tot = (y_true - y_true_mean).T.dot(y_true - y_true_mean)

        return 1 - ss_res / ss_tot
