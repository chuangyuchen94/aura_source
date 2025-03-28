import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class MyKMeansCluster:
    """
    实现K-MEANS聚类算法
    """
    def __init__(self, k_estimator, max_iter=100, minimal_distance=0.001, callback=None):
        """
        K-MEANS算法的实现
        :param k_estimator:预先指定的簇个数
        """
        self.data_labels = None # 数据点的标签
        self.k_estimator = k_estimator
        self.center_points = None  # 用于保存质心
        self.minimal_distance = minimal_distance
        self.max_iter = max_iter
        self.callback = callback # 回调函数，在每次迭代后调用

    def fit(self, X):
        # 对数据进行预处理，包括：标准化、归一化
        X_train = self.train_data_preprocess(X)

        # 初始化质心
        center_point = self.init_cluster_center(X)

        for _ in range(self.max_iter):
            # 分配数据点到质心
            data_labels = self.cluster(X, center_point)

            # 更新质心：按照标签找到簇，计算簇所包含的数据点的均值，作为新的质心
            new_center = self.update_center(X, data_labels)

            # 判断收敛：计算新旧质心的距离，若小于指定阈值，则认为已经收敛
            distance_changed = self.distance_to_center(new_center, self.center_points)[0]
            min_distance = np.min(distance_changed, axis=1)

            self.call_back(center_point, data_labels, min_distance)

            self.center_points = new_center.copy()
            self.data_labels = data_labels.copy()
            center_point = new_center

            mean_distance = np.mean(min_distance)
            if mean_distance < self.minimal_distance:
                print("K-MEANS算法收敛")
                self.call_back(center_point, data_labels, min_distance)
                return
            else:
                print("K-MEANS算法未收敛")


    def call_back(self, new_center, data_labels, min_distance):
        if self.callback is not None:
            self.callback(new_center, data_labels, min_distance)

    def predict(self, X):
        pass

    def update_center(self, X, data_labels):
        """
        更新质心
        :param X:
        :param data_labels:
        :return:
        """
        new_centers = []

        labels = np.unique(data_labels)
        for label in labels:
            cluster_data = X[data_labels == label]
            new_center_point = np.mean(cluster_data, axis=0)
            new_centers.append(new_center_point.copy())

        return np.array(new_centers)

    def cluster(self, X, center_point):
        """
        步骤1：分配数据点到其最近的质心
            方法：计算每个数据点到每个质心的距离，距离最小的那个，就是最近的质心，同时，将这个数据点的簇的标识更新为新的质心
        步骤2：更新质心
        步骤3：判断收敛
        :param X:
        :param center_point:
        :return:
        """
        # 计算每个数据点到质心的距离
        distance = self.distance_to_center(X, center_point)[0]
        # min_distance = np.min(distance, axis=1)
        cluster_labels = np.argmin(distance, axis=1)

        return cluster_labels

    def distance_to_center(self, X, center_points):
        """
        计算数据点到质心的距离: ‖x - c‖² = ‖x‖² - 2x·c + ‖c‖²
        :param X:
        :param center_points:
        :return:
        """
        all_one_k = np.ones((self.k_estimator, 1))
        all_one_features = np.ones((X.shape[0], 1))

        diag_X = np.diag(X.dot(X.T)).reshape(X.shape[0], 1)
        diag_C = np.diag(center_points.dot(center_points.T)).reshape(center_points.shape[0], 1)

        distance_double = np.sum(X ** 2, axis=1, keepdims=True) - 2 * X @ center_points.T + np.sum(center_points ** 2, axis=1)
        distance = np.sqrt(distance_double)

        return distance, distance_double

    def get_belong_cluster(self, X, center_point_all):
        """
        找出指定数据点所属的簇
        :param X:
        :param center_point_all:
        :return:
        """
        k_cluster_dict = {}

        for center_point in center_point_all:
            distance_double, distance = MyKMeansCluster.calc_distance_to_center(X, center_point)
            k_cluster_dict[center_point] = distance_double, distance

        return max(k_cluster_dict, key=lambda x: k_cluster_dict.get(x)[1])

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

        return X_standard_min_max

    def init_cluster_center(self, X_all):
        """
        初始化质心
        :param X:
        :return:
        """
        # 第一个质心
        first_index = np.random.randint(0, X_all.shape[0])
        first_center = X_all[first_index]
        center_point_all = np.array([first_center])

        # 初始化其他质心
        for center_index in range(self.k_estimator - 1):
            # 找出除了质心数据点之外的数据点，离当前质心最近的数据点，并计算其距离，找出其中的最大值，作为新的质心
            distance_matrix = self.distance_to_center(X_all, center_point_all)[0]
            min_distance = np.min(distance_matrix, axis=1)
            max_of_mins = max(min_distance)

            # 概率选择（距离平方的概率分布）
            probs = min_distance ** 2 / np.sum(min_distance ** 2)
            chosen_idx = np.random.choice(X_all.shape[0], p=probs)

            new_center_index = np.where(min_distance == max_of_mins)[0]
            new_center_point = X_all[new_center_index].reshape(1, -1)
            center_point_all = np.concatenate([center_point_all.copy(), new_center_point])

        self.center_points = center_point_all

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