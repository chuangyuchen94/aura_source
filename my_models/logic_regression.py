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
        2）对于多分类问题，使用softmax函数会更好，不过，这里暂且考虑用多个二分类来实现
    4、数据预处理
        1）标签值y，如果不为int值，要将其映射为int
    """
    def __init__(self, max_iter=1000, reg_strength=0.001):
        self.std = None
        self.mean = None
        self.theta = None # 参数θ
        self.y_dict = None # 标签值y的映射字典 {"枚举值": 索引值}
        self.theta_dict = {} # 参数θ的映射字典 {"枚举值": θ}
        self.max_iter = max_iter # 最大迭代次数
        self.reg_strength = reg_strength # 正则化强度

    def fit(self, X_train, y_train):
        """
        训练方法
        :return:
        """
        # 特征标准化
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0) + 1e-8  # 防止除零
        X_train_normalized = (X_train - self.mean) / self.std

        size = X_train.shape[0]
        X_train_extend = MyLogisticRegression.scalar_X(X_train_normalized)
        y_train_scaler, self.y_dict = MyLogisticRegression.scaler_y(y_train)

        # 将多元分类切割成多个二元分类的逻辑：每次的二元逻辑回归，对于当前枚举值而言，将其当做1，其他当做0，总共要切割枚举值个数的次数
        for enum_value in self.y_dict.keys():
            index = self.y_dict[enum_value]
            y_train_extend = np.array(y_train_scaler.copy() == index).reshape(size, )

            print(f"fit for enum_value: {enum_value}, index: {index}")

            theta = self.two_dim_classification(X_train=X_train_extend.copy(), y_train=y_train_extend)
            self.theta_dict[enum_value] = theta.copy()

    def predict(self, X):
        """
        预测方法
        :return:
        """
        # 旧的做法：取出所有预测值中，概率大于0.5的枚举值，然而，现实情况是，可能有多个枚举值的概率都小于0.5
        #length_of_label = max([len(key) for key in self.y_dict.keys()])
        #y_predict_all = np.empty((X.shape[0], 1), dtype=f"U{length_of_label}")

        # for enum_value in self.y_dict.keys():
        #    theta = self.theta_dict[enum_value]
        #    X_normalized = (X - self.mean) / self.std
        #    X_extend = MyLogisticRegression.scalar_X(X_normalized)
        #    # y_predict = MyLogisticRegression.sigmoid(X_extend, theta)
        #    # y_predict_all[y_predict > self.predict_precision] = enum_value

        X_normalized = (X - self.mean) / self.std
        X_extend = MyLogisticRegression.scalar_X(X_normalized)

        # 正确的做法，取多个标签值中，概率最大的那个作为最有可能的枚举值
        prob_matrix = np.column_stack([MyLogisticRegression.sigmoid(X_extend, theta)
                                       for theta in self.theta_dict.values()])
        y_predict_all = np.array([list(self.theta_dict.keys())[i]
                                  for i in np.argmax(prob_matrix, axis=1)])
        return y_predict_all

    def two_dim_classification(self, X_train, y_train):
        """
        实现二元的逻辑回归分类
        :return:
        """
        num_of_features = X_train.shape[1]
        theta = np.random.randn(num_of_features) * np.sqrt(2.9 / (num_of_features + 1))
        n_features = X_train.shape[1]

        res = minimize(
            fun=self.loss_function,  # 进行最小化的目标函数：损失函数
            x0=theta,  # 初始化参数θ
            args=(X_train, y_train),  # 训练集数据
            method='L-BFGS-B',  # 优化方法
            jac=self.gradient_value,  # 用于计算梯度向量的方法
            bounds=[(None, None)] * n_features,  # 无边界时可省略
            options={
                    "maxiter": self.max_iter,
                    "gtol": 1e-6,
                    "disp": True},
            callback=self.callback_fun
        )

        if not res.success:
            print(f"minimize failed, cause: {res.message}")
            raise Exception(f"minimize failed: {res.message}")
        else:
            print(f"***** minimize success, \n    theta: {res.x}\n    loss value: {res.fun}\n    目标函数调用次数:{res.nfev}")
            return res.x


    @staticmethod
    def sigmoid(X, theta):
        """
        sigmoid方法
        :return:
        """
        sigmoid_value = 1 / (1 + np.exp(-X.dot(theta)))
        return sigmoid_value

    def loss_function(self, theta, X, y):
        """
        损失函数
        :return:
        """
        sigmoid_value = MyLogisticRegression.sigmoid(X, theta)
        size = X.shape[0]

        reg_value = 0.5 * self.reg_strength * np.sum(theta[1:] ** 2) # 不惩罚截距项
        loss_value = -1 / size * (y.T.dot(np.log(sigmoid_value)) + (1 - y).T.dot(np.log(1 - sigmoid_value))) + reg_value

        return loss_value

    def gradient_value(self, theta, X, y):
        """
        计算梯度值
        :return:
        """
        print(f"gradient_value is calling!!!")
        # print(f"theta: {theta}")
        # print(f"X: {X}")
        # print(f"y: {y}")
        sigmoid_value = MyLogisticRegression.sigmoid(X, theta)
        size = X.shape[0]
        gradient_value = 1 / size * X.T.dot(sigmoid_value - y)
        gradient_value[1:] += self.reg_strength * theta[1:]

        grad_norm = np.linalg.norm(gradient_value)
        print(f"当前梯度范数: {grad_norm:.2e}")

        return gradient_value

    @staticmethod
    def scaler_y(y):
        """
        将y映射为int枚举值
        :return:
        """
        scaler_y = y.copy()
        unique_value = np.unique(y)
        y_dict = {}

        for index, value in enumerate(unique_value):
            scaler_y[scaler_y == value] = index
            y_dict[value] = index

        return scaler_y, y_dict

    @staticmethod
    def scalar_X(X):
        """
        在X的最前面（第0列之前）加上一列，值全为1，作为截距项
        :param X:
        :return:
        """
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def callback_fun(self, xk):
        """
        定义回调函数
        :param xk:
        :return:
        """
        print(f"当前迭代：{type(xk)}|值：{xk}")