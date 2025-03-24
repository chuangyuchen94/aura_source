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
    def __init__(self):
        self.theta = None # 参数θ
        self.y_dict = None # 标签值y的映射字典 {"枚举值": 索引值}
        self.theta_dict = None # 参数θ的映射字典 {"枚举值": θ}

    def fit(self, X_train, y_train):
        """
        训练方法
        :return:
        """
        size = X_train.shape[0]
        X_train_extend = np.hstack((np.ones((size, 1)), X_train))
        y_train_scaler, self.y_dict = MyLogisticRegression.scaler_y(y_train)

        # 将多元分类切割成多个二元分类的逻辑：每次的二元逻辑回归，对于当前枚举值而言，将其当做1，其他当做0，总共要切割枚举值个数的次数
        for enum_value in self.y_dict.keys():
            index = self.y_dict[enum_value]
            y_train_extend = np.array(y_train_scaler.copy() == index).reshape(size, 1)
            theta = self.two_dim_classification(X_train=X_train_extend, y_train=y_train_extend)
            self.theta_dict[enum_value] = theta

    def predict(self, X):
        """
        预测方法
        :return:
        """
        pass

    def two_dim_classification(self, X_train, y_train):
        """
        实现二元的逻辑回归分类
        :return:
        """
        theta = np.random.randn(X_train.shape[1])

        res = minimize(
            fun=MyLogisticRegression.loss_function,  # 进行最小化的目标函数：损失函数
            x0=theta,  # 初始化参数θ
            args=(X_train, y_train),  # 训练集数据
            method='CG',  # 优化方法
            jac=MyLogisticRegression.gradient_value,  # 用于计算梯度向量的方法
        )

        if not res.success:
            print(f"minimize failed, cause: {res.message}")
            raise Exception(f"minimize failed: {res.message}")
        else:
            print(f"minimize success, theta: {res.x}")
            return res.x


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