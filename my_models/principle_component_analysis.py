import numpy as np
from sklearn.preprocessing import StandardScaler

class MyPCATransformer:
    """
    实现PCA（主成分分析）算法
    """
    def __init__(self, min_threshold=0.95, is_standardize=True):
        """
        初始化方法
        :param min_threshold:
        """
        self.min_threshold = min_threshold # 阈值，用于计算留下多少个特征向量
        self.is_standardize = is_standardize # 是否需要对数据集进行标准化处理
        self.standardize_transformer = None # 标准化转换器
        self.best_feature_value = [] # 最优特征值
        self.best_feature_vector = None # 最优特征向量

    def fit(self, X, y=None):
        """
        用训练集数据，计算得到特征值及特征向量
        :param X:
        :param y: 因为是无监督算法，所以，实际上不需要传入y
        :return:
        """
        if self.is_standardize:
            X = self.standardize_input(X)

        sample_num = X.shape[0]

        # 计算协方差矩阵：C=1/m * X^T @ X
        cov_matrix = X.T.dot(X) / sample_num

        # 计算特征值及特征向量：特征值, 特征向量矩阵 = np.linalg.eigh(C)
        candidate_feature_value, candidate_feature_vector = np.linalg.eigh(cov_matrix)

        # 累计特征值比例，并求出最佳特征值及特征向量组合
        sorted_indexes = np.argsort(candidate_feature_value)[::-1]
        total_value = np.sum(candidate_feature_value)
        accumulate_ratio = 0
        selected_index = []
        for index in sorted_indexes:
            accumulate_ratio += candidate_feature_value[index] / total_value
            selected_index.append(index)
            if accumulate_ratio >= self.min_threshold:
                break
        self.best_feature_value = candidate_feature_value[selected_index]
        self.best_feature_vector = candidate_feature_vector[:, selected_index]


    def transform(self, X):
        """
        对特征数据进行降维处理
        :param X:
        :return:
        """
        if self.is_standardize:
            X = self.standardize_input(X)

        transformed_X = X.dot(self.best_feature_vector)

        return transformed_X

    def fit_transform(self, X, y=None):
        """
        运用PCA算法求得最优特征值及特征向量，并对特征数据集X进行降维处理
        :param X:
        :param y: 因为是无监督算法，所以，实际上不需要传入y
        :return:
        """
        pass

    def standardize_input(self, X):
        """
        对数据集进行标准化处理
        :param X:
        :return:
        """
        standardize_transformer = StandardScaler()
        X_std = standardize_transformer.fit_transform(X)

        self.standardize_transformer = standardize_transformer

        return X_std
