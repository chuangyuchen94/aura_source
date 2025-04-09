import numpy as np
from sklearn.preprocessing import StandardScaler

class NeuralLayer:
    """
    定义每一层的网络层
    1、属性：
    1）输入a
    2）参数θ
    3）输出y
    2、方法
    1）fit：训练方法。包括线性转换+激活函数。
    2）reverse_brocast：反向传播
    3）update_theta：更新参数
    """
    def __init__(self, input_num, output_num, activation_func=None, derivative_func=None, is_input=False, is_final_outcome=False):
        self.input_num = input_num # 输入的维度
        self.output_num = output_num # 输出的维度
        self.activation_func = activation_func # 激活函数
        self.derivative_func = derivative_func # 激活函数的导数，用于反向传播时计算损失值
        self.is_input = is_input # 是否为输入层
        self.a = None # 输入值
        self.y = None # 输出值
        self.theta = None # 参数
        self.is_final_outcome = is_final_outcome # 激活层计算的结果，是否就是最终结果

    def init_theta(self):
        """
        初始化参数值
        :return:
        """
        scale = np.sqrt(2.0 / self.input_num)  # He初始化
        self.theta = np.random.randn(self.input_num, self.output_num) * scale

    def fit(self, X):
        """
        训练方法
        :param input_value:
        :return:
        """
        input_features = np.insert(X, 0, 1, axis=1) if self.is_input else X

        y_result_linear = input_features.dot(self.theta)  # 线性变换
        y_result_activate = self.activation_func(y_result_linear)

        self.a = input_features.copy()  # 输入值
        self.y = y_result_activate.copy()  # 输出值
        activate_mean = y_result_activate.mean()

        return y_result_activate, activate_mean

    def reverse_broacast(self, layer_index, loss_of_next_layer, theta_of_next_layer, learning_rate, print_log=False):
        """
        反向传播
        :return:
        """
        if not self.is_final_outcome:
            deriv_value = self.derivative_func(self.y)
            loss_of_next_layer = (loss_of_next_layer.dot(theta_of_next_layer.T)) * deriv_value

        input_of_current_layer = self.a # 本层的输入值
        gradient_value = input_of_current_layer.T.dot(loss_of_next_layer) / \
                                        input_of_current_layer.shape[0]
        self.theta -= learning_rate * gradient_value

        # 正常范围应在1e-5到1e-3之间
        if print_log and (
                    np.abs(gradient_value).max() < 1e-5 or np.abs(gradient_value).max() > 1e-3):
            print(f"Layer {layer_index} gradient range:",
                      np.abs(gradient_value).min(),
                      np.abs(gradient_value).max())

        grad_norm = np.linalg.norm(gradient_value)
        if print_log:
            print(f"Layer {layer_index} gradient norm:", grad_norm)

        return grad_norm

    def set_final_outcome_flag(self, flag=True):
        """
        设置该隐藏层的下一层是否为输出层
        :param flag:
        :return:
        """
        self.is_final_outcome = flag

    def get_input_value(self):
        """
        获取输入值
        :return:
        """
        return self.a

    def get_output_value(self):
        """
        获取输出值
        :return:
        """
        return self.y

    def get_theta(self):
        """
        获取参数值
        :return:
        """
        return self.theta

class MyNeuralNetwork:
    """
    实现神经网络算法
    1、 层与层之间的数据处理：线性变换、激活函数
    """
    def __init__(self, max_iter=1000, tol=1e-5, learning_rate=0.01, layer=None, y_to_one_hot=True, theta_init_rate =0.01, print_log=False, random_state=42):
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
        if random_state is not None:
            np.random.seed(random_state)
        self.theta_init_rate = theta_init_rate
        self.grad_norm = []
        self.activation_mean = []
        self.print_log = print_log

    def fit(self, X, y):
        X_std = self.standardize(X)
        y_std = MyNeuralNetwork.one_hotize(y) if self.y_to_one_hot else y
        X_input = np.insert(X_std, 0, 1, axis=1)
        y_label_num = len(np.unique(y))

        self.init_layer(self.layer, X_input, y_label_num)

        loss_of_iter = []

        for _ in range(self.max_iter):
            y_result = self.predict_forword(X_std, theta_in_layer) # 前向传播，计算结果
            theta_in_layer = self.update_thetas_backward(y_result, y_std, X_input, theta_in_layer) # 反向传播，更新参数

            loss_value = MyNeuralNetwork.calc_loss(y_result, y_std)
            loss_of_iter.append(loss_value)

        self.theta_in_layer = theta_in_layer

        return loss_of_iter

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

    def init_layer(self, layer, X_input, y_label_num):
        """
        初始化每一层及其参数值
        :return:
        """
        layer_num = len(layer) + 1
        layer_of_all = {}

        neural_num_of_layer = []
        neural_num_of_layer.append(X_input.shape[1])
        neural_num_of_layer.extend(layer)
        neural_num_of_layer.append(y_label_num)

        for layer_index in range(layer_num):
            input_num = neural_num_of_layer[layer_index]
            output_num = neural_num_of_layer[layer_index + 1]
            layer_of_each = NeuralLayer(
                input_num=input_num,
                output_num=output_num,
                activation_func=MyNeuralNetwork.relu,
                derivative_func=MyNeuralNetwork.relu_derivative,
                is_input=layer_index == 0,
                is_final_outcome=layer_index == layer_num - 1)
            layer_of_all[layer_index] = layer_of_each

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
            scale = np.sqrt(2.0 / layer_neural_num[theta_index])  # He初始化
            layer_thetas[theta_index] = np.random.randn(in_nueral_num, out_nueral_num) * scale

        return layer_thetas


    def predict_forword(self, X, theta_in_layer):
        """
        前向传播，计算结果
        1、 线性变换
        2、激活函数
        :return:预测结果值
        """
        y_result = None
        # input_features = X / 255.0  # 添加归一化
        input_features = X
        num_of_layer = len(self.layer)

        for theta_index in range(num_of_layer):
            theta = theta_in_layer[theta_index]
            input_features = np.insert(input_features, 0, 1, axis=1)
            y_result_linear = input_features.dot(theta) # 线性变换
            y_result_activate = MyNeuralNetwork.relu(y_result_linear)

            self.value_of_layer[theta_index] = y_result_activate.copy()

            input_features = y_result_activate

            self.activation_mean.append(y_result_activate.mean())
            if self.print_log:
                print(f"Layer {theta_index} activation mean:", y_result_activate.mean())

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
        input_of_outer_extend = np.insert(input_of_outer, 0, 1, axis=1)
        outer_gradient = input_of_outer_extend.T.dot(outer_loss)
        theta_in_layer_new[outer_layer] = theta_in_layer[outer_layer] - self.learning_rate * outer_gradient

        # 处理每一层隐藏层之间：计算损失值->计算梯度值->更新参数
        loss_of_next_layer = outer_loss
        for layer_index in range(outer_layer - 1, -1, -1):
            y_result_activate = self.value_of_layer[layer_index]
            relu_deriv = MyNeuralNetwork.relu_derivative(y_result_activate)
            theta_of_next_layer = theta_in_layer.get(layer_index + 1)
            loss_of_current_layer = (loss_of_next_layer.dot(theta_of_next_layer.T[:, 1:])) * relu_deriv # * 10
            input_of_current_layer = X_std if 0 == layer_index else self.value_of_layer[layer_index - 1]
            gradient_of_current_layer = input_of_current_layer.T.dot(loss_of_current_layer) / input_of_current_layer.shape[0]
            theta_in_layer_new[layer_index] = theta_in_layer[layer_index] - self.learning_rate * gradient_of_current_layer

            loss_of_next_layer = loss_of_current_layer.copy()

            # 正常范围应在1e-5到1e-3之间
            if self.print_log and (np.abs(gradient_of_current_layer).max() < 1e-5 or np.abs(gradient_of_current_layer).max() > 1e-3):
                print(f"Layer {layer_index} gradient range:",
                  np.abs(gradient_of_current_layer).min(),
                  np.abs(gradient_of_current_layer).max())

            grad_norm = np.linalg.norm(gradient_of_current_layer)
            self.grad_norm.append(grad_norm)
            if self.print_log:
                print(f"Layer {layer_index} gradient norm:", grad_norm)

        return theta_in_layer_new

    @staticmethod
    def relu_derivative(a):
        return (a > 0).astype(float)  # 基础ReLU导数

    @staticmethod
    def sigmoid(z):
        """
        sigmoid方法
        :return:
        """
        z = np.clip(z, -10, 10)  # 限制输入范围
        sigmoid_value = 1 / (1 + np.exp(-z))
        return sigmoid_value

    @staticmethod
    def relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def softmax(z):
        # 减去最大值防止溢出
        z = z - np.max(z, axis=1, keepdims=True)  # 按行归一化
        z_exp = np.exp(z)
        return z_exp / (np.sum(z_exp, axis=1, keepdims=True) + 1e-8)  # 添加epsilon

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
        return (y_result - y_std) / y_std.shape[0]

    @staticmethod
    def calc_loss(y_pred, y_true):
        """
        计算损失值
        :return:
        """
        epsilon = 1e-8
        loss_matrix = -y_true * np.log(y_pred + epsilon)  # 形状 (1000,10)
        total_loss = np.mean(loss_matrix)  # 标量损失

        return total_loss

    def calc_gradient(self):
        """
        计算梯度值
        :return:
        """


    def predict(self, X):
        """
        进行预测
        :param X:
        :return:
        """
        # input_features = X / 255.0  # 添加归一化
        X_std = self.standard_transformer.transform(X)
        y_softmax = self.predict_forword(X_std, self.theta_in_layer)
        y_max_index = np.argmax(y_softmax, axis=1).reshape(-1, 1)

        return y_max_index
