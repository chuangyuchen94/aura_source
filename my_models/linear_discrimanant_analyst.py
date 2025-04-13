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

    def fit(self, X, y):
        """
        训练方法
        :param X: (m, n)矩阵
        :param y: (m, 1)矩阵
        :return:
        """
        # 统计标签类别
        label_all = np.unique(y)

        # 计算投影前每个标签类别的均值：对列求均值Uk；结果为(1, n)矩阵
        mean_of_label = {}
        for label in label_all:
            mean_of_label[label] = np.mean(X[y==label], axis=0)

        # 计算类内散布矩阵 SW=sum(Sk), Sk=sum((x-u)*(x-u)^T)，用矩阵表示为：Sk=(Xk-Uk)^T@(Xk-Uk)
        scatter_matrix_w = {}
        for label in label_all:
            scatter_matrix_w[label] = (X[y==label]-mean_of_label[label]).T.dot(X[y==label]-mean_of_label[label])

        # 计算类间散布矩阵
        mean_of_all = np.sum(X)
        scatter_matrix_b = {}

        # 计算特征向量及特征值

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

