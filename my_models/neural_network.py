import numpy as np
from sklearn.preprocessing import StandardScaler

class MyNeuralNetwork:
    """
    实现神经网络算法
    1、 层与层之间的数据处理：线性变换、激活函数
    """
    def __init__(self, max_iter=1000, tol=1e-5, learning_rate=0.01, layer=None, y_to_one_hot=True):
        if layer is None:
            layer = [25, ]
        self.max_iter = max_iter # 最大迭代次数
        self.tol = tol # 迭代终止条件
        self.learning_rate = learning_rate # 学习率
        self.layer = layer # 层的架构定义：每一层的神经元个数
        self.standard_transformer = None
        self.y_to_one_hot = y_to_one_hot # 是否需要将标签转换为one-hot编码
        self.value_of_layer = {} # 每一层的输出值
        self.theta_in_layer = None # 每一层的参数值

    def fit(self, X, y):
        X_std = self.standardize(X)
        y_std = MyNeuralNetwork.one_hotize(y) if self.y_to_one_hot else y
        label_num = len(np.unique(y))
        theta_in_layer = self.init_thetas(X.shape[1], label_num)
        X_input = np.insert(X_std, 0, 1, axis=1)


        for _ in range(self.max_iter):
            y_result = self.predict_forword(X_std, theta_in_layer) # 前向传播，计算结果
            theta_in_layer = self.update_thetas_backward(y_result, y_std, X_input, theta_in_layer) # 反向传播，更新参数

        self.theta_in_layer = theta_in_layer

    def standardize(self, X):
        """
        对特征数据进行标准化处理
        :param X:
        :return:
        """
        standard_transformer = StandardScaler()
        X_new = standard_transformer.fit_transform(X)

        self.standard_transformer = standard_transformer # 保存标准化处理实例，用于后续预测时对预测的特征数据进行预处理
        return X_new

    @staticmethod
    def one_hotize(y):
        """
        独热编码
        :return:
        """
        num_classes = 10  # 明确指定类别总数
        y_std = np.eye(num_classes)[y.reshape(-1)]
        return y_std

    def init_thetas(self, num_feature, num_label):
        """
        初始化每一层的参数值
        :return:
        """
        theta_num = len(self.layer) + 1 # 1为输入层到隐藏层的参数
        layer_thetas = {}
        layer_neural_num = [num_feature, ]
        layer_neural_num.extend(self.layer)
        layer_neural_num.append(num_label) # 每一层的神经元个数（特征个数）

        for theta_index in range(theta_num):
            in_nueral_num = layer_neural_num[theta_index]+ 1 # 1代表偏置项
            out_nueral_num = layer_neural_num[theta_index + 1]
            layer_thetas[theta_index] = np.random.randn(in_nueral_num, out_nueral_num) * 0.01

        return layer_thetas


    def predict_forword(self, X, theta_in_layer):
        """
        前向传播，计算结果
        1、 线性变换
        2、激活函数
        :return:预测结果值
        """
        y_result = None
        input_features = X
        num_of_layer = len(self.layer)

        for theta_index in range(num_of_layer):
            theta = theta_in_layer[theta_index]
            input_features = np.insert(input_features, 0, 1, axis=1)
            y_result_linear = input_features.dot(theta) # 线性变换
            y_result_activate = MyNeuralNetwork.sigmoid(y_result_linear)

            self.value_of_layer[theta_index] = y_result_activate.copy()

            input_features = y_result_activate

        # 最后一层隐藏层到输出层，单独处理，因为涉及分类问题，需要对其应用softmax
        theta = theta_in_layer[num_of_layer]
        input_features = np.insert(input_features, 0, 1, axis=1)
        y_result_linear = input_features.dot(theta)  # 线性变换
        y_result = MyNeuralNetwork.softmax(y_result_linear) # softmax变换
        self.value_of_layer[num_of_layer] = y_result.copy()

        return y_result

    def update_thetas_backward(self, y_result, y_std, X_std, theta_in_layer):
        """
        反向传播：计算每一层的梯度值，并更新每一层的参数
        :return:
        """
        # 计算损失值，计算梯度值，更新参数
        # 先处理最后一层隐藏层到输出层的梯度值、更新参数（因为最后一层到输出层的激活函数用的是softmax，与其他层不同）
        # 计算损失值
        num_of_layer = len(self.layer) + 1
        theta_in_layer_new = theta_in_layer.copy()

        outer_layer = num_of_layer - 1
        outer_loss = self.calc_ouput_loss(y_result, y_std)
        input_of_outer = self.value_of_layer[outer_layer - 1]
        input_of_outer = np.insert(input_of_outer, 0, 1, axis=1)
        outer_gradient = input_of_outer.T.dot(outer_loss)
        theta_in_layer_new[outer_layer] = theta_in_layer_new[outer_layer] - self.learning_rate * outer_gradient

        # 处理每一层隐藏层之间：计算损失值->计算梯度值->更新参数
        loss_of_next_layer = outer_loss
        for layer_index in range(outer_layer - 1, -1, -1):
            sigmoid_value = self.value_of_layer[layer_index]
            theta_of_current_layer = theta_in_layer_new.get(layer_index)
            loss_of_current_layer = (theta_of_current_layer.T.dot(loss_of_next_layer)) * (
                sigmoid_value * (1 - sigmoid_value))
            input_of_current_layer = X_std if 0 == layer_index else self.value_of_layer[layer_index - 1]
            gradient_of_current_layer = input_of_current_layer.T.dot(loss_of_current_layer)
            theta_in_layer_new[layer_index] -= self.learning_rate * gradient_of_current_layer

            loss_of_next_layer = loss_of_current_layer.copy()

        return theta_in_layer_new

    @staticmethod
    def sigmoid(z):
        """
        sigmoid方法
        :return:
        """
        sigmoid_value = 1 / (1 + np.exp(-z))
        return sigmoid_value

    @staticmethod
    def softmax(z):
        # 减去最大值防止溢出
        z = z - np.max(z, axis=1, keepdims=True)  # 按行归一化
        z_exp = np.exp(z)
        return z_exp / np.sum(z_exp, axis=1, keepdims=True) + 1e-8  # 添加epsilon

    @staticmethod
    def softmax_derivative(softmax_z):
        """
        softmax作为激活函数的反向传播处理
        :return:
        """
        jacobian = np.diag(softmax_z)
        jacobian -= np.outer(softmax_z, softmax_z)

        return jacobian

    def calc_ouput_loss(self, y_result, y_std):
        """
        计算最后一层隐层到输出层的损失值
        :param y_result:
        :param y_std:
        :return:
        """
        return abs(y_result - y_std)

    def calc_loss(self, y_pred, y_true, theta_l, loss_l_p1):
        """
        计算损失值
        :return:
        """
        size = y_pred.shape[0]
        loss_value = -1 / size * (y_true.T.dot(np.log(y_pred)) + (1 - y_true).T.dot(np.log(1 - y_pred)))

        return loss_value

    def calc_gradient(self):
        """
        计算梯度值
        :return:
        """


    def predict(self, X):
        pass