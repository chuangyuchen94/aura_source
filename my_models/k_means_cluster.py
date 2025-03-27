import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class MyKMeansCluster:
    """
    实现K-MEANS聚类算法
    """
    def __init__(self, k_estimator):
        """
        K-MEANS算法的实现
        :param k_estimator:预先指定的簇个数
        """
        self.k_estimator = k_estimator
        self.standard_tranformer = None
        self.min_max_transformer = None


    def fit(self, X):
        # 对数据进行预处理，包括：标准化、归一化
        X_train = self.train_data_preprocess(X)

        # 初始化质心
        center_point = self.init_cluster_center(X_train)
        cluster()

    def predict(self, X):
        pass

    def cluster(self, X, init_center_point):
        """
        步骤1：分配数据点到其最近的质心
        步骤2：更新质心
        步骤3：判断收敛
        :param X:
        :param init_center_point:
        :return:
        """
        k_cluster_dict = {}


    def get_belong_cluster(self, X, center_point_all):
        """
        找出指定数据点所属的簇
        :param X:
        :param center_point_all:
        :return:
        """

    def train_data_preprocess(self, X):
        """
        对训练集数据进行预处理
        :param X: 训练集数据
        :return:
        """
        # 标准化
        standard_tranformer = StandardScaler()
        X_standard = standard_tranformer.fit_transform(X)

        # 归一化
        min_max_transformer = MinMaxScaler(feature_range=(0, 1))
        X_standard_min_max = min_max_transformer.fit_transform(X_standard)

        self.standard_tranformer = standard_tranformer
        self.min_max_transformer = min_max_transformer

        return X_standard_min_max

    def init_cluster_center(self, X):
        """
        初始化质心
        :param X:
        :return:
        """
        center_point_all = []
        index_all = []

        # 第一个质心
        first_index = np.random.randint(0, X.shape[0])
        first_center = X[first_index]
        center_point_all.append(first_center)
        index_all.append(first_index)

        # 初始化其他质心
        for _ in range(self.k_estimator - 1):
            mask = np.ones(X.shape[0], dtype=bool)
            mask[index_all] = False
            available_X = X[mask]

            # 找出除了质心数据点之外的数据点，离当前质心最近的数据点，并计算器距离，找出其中的最大值，作为新的质心
            distance_to_center = {}
            for data_index in range(available_X.shape[0]):
                distance = MyKMeansCluster.calc_distance_to_center(available_X.iloc[data_index], center_point_all)
                distance_to_center[data_index] = distance
            new_center_index = max(distance_to_center, key=distance_to_center.get)
            index_all.append(new_center_index)
            center_point_all.append(available_X.iloc[new_center_index])

        return center_point_all

    @staticmethod
    def get_center_point(X, center_point_all):
        """
        根据已有的质心，找出新的质心
        :param X:除去了已有质心的数据点
        :param center_point_all:已有的质心
        :return:
        """
        for data_index in range(X.shape[0]):
            MyKMeansCluster.calc_distance_to_center(X.iloc[data_index], center_point_all)

    @staticmethod
    def calc_distance_to_center(point_a, center_point_all):
        """
        计算指定的数据点到质心的距离，并选出最小距离
        :param point_a:
        :param center_point_all:
        :return:
        """
        return min([MyKMeansCluster.calc_distance(point_a, center_point) for center_point in center_point_all])
    @staticmethod
    def calc_distance(point_a, point_b):
        """
        计算距离：欧几里得距离
        :param point_a: a点
        :param point_b: b点
        :return: 一个元组，(距离的平方, 距离)
        """
        distance_double = np.sum((point_a - point_b) ** 2, axis=1)
        distance = np.sqrt(distance_double)

        return distance_double, distance