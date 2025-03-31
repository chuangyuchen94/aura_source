import numpy as np
def create_dataset():
    data_set = [
        [0, 0, 0, 0, "no"],
        [0, 0, 0, 1, "no"],
        [0, 1, 0, 1, "yes"],
        [0, 1, 1, 0, "yes"],
        [0, 0, 0, 0, "no"],
        [1, 0, 0, 0, "no"],
        [1, 0, 0, 1, "no"],
        [1, 1, 1, 1, "yes"],
        [1, 0, 1, 2, "yes"],
        [1, 0, 1, 2, "yes"],
        [2, 0, 1, 2, "yes"],
        [2, 0, 1, 1, "yes"],
        [2, 1, 0, 1, "yes"],
        [2, 1, 0, 2, "yes"],
        [2, 0, 0, 0, "no"],
    ]

    labels = ["F1-AGE", "F2-WORK", "F3-HOME", "F4-LOAN"]

    return np.array(data_set), np.array(labels)


class MyDecisionTree:
    """
    决策树模型
    """
    def __init__(self):
        self.features_used = [] # 已经使用的特征
        pass

    def fit(self, X, y):
        """
        训练方法
        1、每次分割的操作：要按照分割条件将数据分割
        :param X:
        :param y:
        :return:
        """
        # 找到没被使用的特征，并计算其熵值
        feature_all = range(X.shape[1])
        featured_unused = set(feature_all) - set(self.features_used)

        for feature in featured_unused:
            self.calc_feature_entropy(X, y, feature)


    def calc_feature_entropy(self, X, y, feature_index):
        """
        计算节点的熵值
        :param X:
        :param y:
        :return:
        """
        # 按照枚举值或连续值来切割
        entropy_value = 0

        feature_value = np.unique(X[:, feature_index])
        for feature in feature_value:
            y_slice = y[X[:, feature_index] == feature]
            y_slice_entropy = MyDecisionTree.calc_target_entropy(y_slice)
            entropy_value += y_slice_entropy

    @staticmethod
    def calc_target_entropy(y):
        """
        计算目标数据集的熵值
        :param y:
        :return:
        """
        y_value = np.unique(y)
        y_entropy_value = sum([(sum(y == y_label)/len(y) * np.log2(sum(y == y_label)/len(y))) for y_label in y_value])
        return y_entropy_value

    def predict(self, X):
        """
        预测方法
        :param X:
        :return:
        """

if "__main__" == __name__:
    data, label = create_dataset()
    print(f"data: {data.shape}")
