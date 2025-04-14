import numpy as np

"""
实现线性判别分析算法
"""

class MyLDA:
    def __init__(self, n_features=None):
        """
        初始化方法
        """
        self.n_features= n_features # 降维后的维数，若不指定，则将取标签类别数-1
        self.scatter_matrix_w = None # 类内散布矩阵
        self.scatter_matrix_b = None # 类间散布矩阵

    def fit(self, X, y):
        """
        训练方法
        :param X: (m, n)矩阵
        :param y: (m, 1)矩阵
        :return:
        """
        # 统计标签类别
        y = y.reshape(-1)
        label_all = np.unique(y)
        target_matrix = np.zeros((X.shape[0], len(label_all)))
        for label_index, label in enumerate(label_all):
            target_matrix[:, label_index] = (y == label)
        label_count_matrix = np.sum(target_matrix, axis=0)  # 计算每个类别的样本数
        type_means = X.T.dot(target_matrix) / label_count_matrix

        # 计算类内散布矩阵 SW=sum(Sk), Sk=sum((x-u)*(x-u)^T)，用矩阵表示为：Sk=(Xk-Uk)^T@(Xk-Uk)
        means = target_matrix.dot(type_means.T)
        diff_w = X - means
        scatter_matrix_w = diff_w.T.dot(diff_w).astype(np.float64)
        self.scatter_matrix_w = scatter_matrix_w

        # 计算类间散布矩阵 SB = D⋅W⋅D^T
        mean_of_all = np.mean(X)
        diff_b = type_means - mean_of_all
        w_matrix = np.diag(label_count_matrix)
        scatter_matrix_b = diff_b.dot(w_matrix).dot(diff_b.T).astype(np.float64)

        self.scatter_matrix_b = scatter_matrix_b

        # 计算特征向量及特征值
        scatter_sw_inv_sb = np.linalg.inv(scatter_matrix_w) @ scatter_matrix_b
        reject_value, reject_vector = np.linalg.eig(scatter_sw_inv_sb)

        # 降维
        reject_value = reject_value.reshape(-1)
        order_index = np.argsort(reject_value)

        return

    def fit_transform(self, X, y):
        """
        训练，并返回投影后的X
        :param X:
        :param y:
        :return:
        """
        pass

    def transform(self, X):
        """
        对X进行投影
        :param X:
        :return:
        """
        pass

