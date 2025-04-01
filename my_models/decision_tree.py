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


class DecisionNode:
    def __init__(self, feature_index=None, slice_method=None, enum_values=None, slice_value=None, target_value=None):
        self.feature_index = feature_index # 节点的索引
        self.slice_method = slice_method   # 切割下级的方法：E-按枚举，C-按连续值
        self.enum_values = enum_values     # 枚举值列表
        self.slice_value = slice_value     # 切割值
        self.next_node = {}                # 切割逻辑下对应的下级节点
        self.target_value = target_value           # 当前节点的标签值

    def set_sub_node(self, slice_value, sub_node):
        """
        根据切分规则，设置下级节点
        :param slice_value:
        :param node:
        :return:
        """
        self.next_node[slice_value] = sub_node


class MyDecisionTree:
    """
    决策树模型
    """
    def __init__(self):
        self.root_node = None # 已经使用的特征
        self.feature_index_all = None # 所有的特征


    def fit(self, X, y):
        """
        训练方法
        1、每次分割的操作：要按照分割条件将数据分割
        :param X:
        :param y:
        :return:
        """
        self.feature_index_all = range(X.shape[1])

        # 找到根节点
        root_feature_index, root_entropy_value = self.find_best_feature(X, y, [])
        root_feature_value = np.unique(X[:, root_feature_index])
        self.root_node = DecisionNode(feature_index=root_feature_index, slice_method="E", enum_values=root_feature_value, target_value=y)

        # 递归找到下级节点
        # 上级节点的切分值，上级节点的列索引，上级节点
        for feature_label in root_feature_value:
            slice_index = X[:, root_feature_index] == feature_label
            X_new = X[slice_index]
            y_new = y[slice_index]
            self.build_next_node(root_feature_index, feature_label, [root_feature_index], self.root_node, X_new, y_new)

    def build_next_node(self, parent_feature_index, parent_feature_value, feature_index_used, parent_node, X, y):
        """
        根据父节点，构建当前节点
        :param parent_feature_index:
        :param parent_feature_value:
        :param parent_node:
        :return:
        """
        # 先计算当前节点的熵值
        entropy_value_orig = abs(self.calc_target_entropy(y))
        if entropy_value_orig < 1e-6: # 如果当前局部的初始熵值已经小于阈值，则创建叶子节点，并返回
            current_node = DecisionNode(feature_index=None, slice_method="E",
                                        enum_values=None, target_value=y)
            parent_node.set_sub_node(parent_feature_value, current_node)
            return


        # 找到最优特征，构建分支节点
        best_node_current, entropy_value_current = self.find_best_feature(X, y, feature_index_used)
        current_node = DecisionNode(feature_index=best_node_current, slice_method="E", enum_values=np.unique(X[:, best_node_current]), target_value=y)
        parent_node.set_sub_node(parent_feature_value, current_node)

        feature_value_current = np.unique(X[:, best_node_current])
        for label in feature_value_current:
            slice_index = X[:, best_node_current] == label
            X_new = X[slice_index]
            y_new = y[slice_index]
            self.build_next_node(best_node_current, label, feature_index_used + [best_node_current], current_node, X_new, y_new)

    def find_best_feature(self, X, y, feature_used):
        """
        找出当前节点的最优特征
        :param X:
        :param y:
        :param feather_used:
        :return:
        """
        # 找到没被使用的特征，并计算其熵值
        featured_unused = set(self.feature_index_all) - set(feature_used)
        entropy_dict = {}

        for feature in featured_unused:
            entropy_value = self.calc_feature_entropy(X, y, feature)
            entropy_dict[feature] = entropy_value

        # 熵值最小，相对来说，信息增益值最大
        best_feature = min(entropy_dict, key=entropy_dict.get)
        best_entropy_value = entropy_dict[best_feature]

        return best_feature, best_entropy_value

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
            entropy_value -= y_slice_entropy

        return entropy_value

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
        predict_result = []

        sample_num = X.shape[0]
        for sample_index in range(sample_num):
            sample = X[sample_index, :]
            sample_result = self.get_predict_result(self.root_node, sample)
            predict_result.append(sample_result)

        return np.array(predict_result)

    def get_predict_result(self, node, sample_one_row):
        """
        预测一行数据的结果
        :param sample_one_row:
        :return:
        """
        if node.feature_index is None:
            unique_elements, counts = np.unique(node.target_value, return_counts=True)
            max_count_idx = np.argmax(counts)
            predict_value = unique_elements[max_count_idx]
            predict_percent = counts[max_count_idx] / len(node.target_value)
            return predict_value, predict_percent

        next_node = node.next_node[sample_one_row[node.feature_index]]
        return self.get_predict_result(next_node, sample_one_row)


if "__main__" == __name__:
    data, label = create_dataset()
    print(f"data: {data.shape}")
