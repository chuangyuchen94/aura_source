import pandas as pd

class RelatedRule:
    """
    关联规则的类
    1、组成：
    1）元素的组合：由元组表示
    2）指标的值
    """
    def __init__(self, item_tuple, pre_set=None, post_set=None):
        """
        初始化方法
        :param item_tuple:
        """
        self.item_tuple = item_tuple # 项集
        self.support = None # 支持度
        self.confidence = None # 置信度
        self.lift = None # 提升度
        self.pre_set = pre_set # 前置项集
        self.post_set = post_set # 后置项集

    def set_metric(self, metric, value):
        """
        设置指标的值
        :param metric:
        :param value:
        :return:
        """
        if "support" == metric:
            self.support = value
        elif "confidence" == metric:
            self.confidence = value
        elif "lift" == metric:
            self.lift = value
        else:
            raise ValueError("metric is not support")

    def get_set(self):
        return self.item_tuple

    def get_metric(self, metric):
        """
        获取指标的值
        :param metric:
        :return:
        """
        if "support" == metric:
            return self.support
        elif "confidence" == metric:
            return self.confidence
        elif "lift" == metric:
            return self.lift
        else:
            raise ValueError("metric is not support")

    def set_pre_set(self, pre_set):
        """
        设置前置项集
        :param pre_set:
        :return:
        """
        self.pre_set = pre_set

    def set_post_set(self, post_set):
        """
        设置后置项集
        :param post_set:
        :return:
        """
        self.post_set = post_set

    def print(self, title=None):
        """
        打印关联规则
        :param title:
        :return:
        """
        if title is not None:
            print(title)

        print(f"item_tuple:{self.item_tuple} | pre_set:{self.pre_set} | post_set:{self.post_set} | support:{self.support} | confidence:{self.confidence} | lift:{self.lift} | ")

class MyRelatedRuleModel:
    def __init__(self):
        """
        初始化方法
        :return:
        """
        self.related_rule_map = {}

    def build_related_rule(self, X, metric="support", min_threshold=0.5):
        """
        构建关联规则
        :param X:
        :return:
        """
        X_std = pd.DataFrame(X)

        # 构建第一层的关联规则
        self.init_rule(X_std, metric, min_threshold)

        # 逐层构建关联规则
        self.iter_rule(X_std, metric, min_threshold)

    def init_rule(self, X_std, metric, min_threshold):
        """
        构建第一层的关联规则
        :param X_std:
        :return:
        """
        feature_list = X_std.stack().unique().tolist()
        feature_list.sort()
        related_rule_list = []

        for feature in feature_list:
            item_tuple = (feature, )
            related_rule = self.create_related_rule(X_std, item_tuple, metric, min_threshold)

            if related_rule is not None:
                related_rule_list.append(related_rule)

        self.related_rule_map[0] = related_rule_list


    def create_related_rule(self, X_std, item_tuple, metric, min_threshold):
        value_of_metric = self.cal_set_metric(X_std, item_tuple, metric)
        if value_of_metric < min_threshold:
            return None

        related_rule = RelatedRule(item_tuple)
        related_rule.set_metric(metric, value_of_metric)

        return related_rule

    def cal_set_metric(self, X_std, item_tuple, metric):
        """
        计算项集的指标
        :param item_tuple:
        :param metric:
        :return:
        """
        if "support" == metric:
            return MyRelatedRuleModel.calc_set_support(X_std, item_tuple)

    @staticmethod
    def calc_set_support(X_std, item_tuple):
        """
        计算项集的支持度
        :param X_std:
        :param item_tuple:
        :return:
        """
        num_of_sample = X_std.shape[0]

        count_of_exists = 0
        for sample in X_std.itertuples():
            tuple_of_row = tuple(sample[1:])
            if set(item_tuple).issubset(set(tuple_of_row)):
                count_of_exists += 1

        return count_of_exists / num_of_sample

    def get_related_rule(self, metric, min_threshold):
        """
        获取关联规则
        :param metric:
        :param min_threshold:
        :return:
        """
        pass

    def iter_rule(self, X_std, metric, min_threshold):
        """
        迭代的建立关联关系
        :return:
        """
        layer_of_current = 0

        while True:
            num_of_rules_built = self.build_rules_of_next_layer(layer_of_current, X_std, metric, min_threshold)
            if num_of_rules_built == 0:
                break
            layer_of_current += 1

    def build_rules_of_next_layer(self, layer_of_current, X_std, metric, min_threshold):
        """
        逐层创建规则
        :return:
        """
        related_rule_list = []

        rules_to_combine = self.related_rule_map[layer_of_current]
        len_of_rules_to_combine = len(rules_to_combine)

        if 1 == len_of_rules_to_combine:
            return 0

        for index in range(len_of_rules_to_combine - 1):
            current_index = index
            next_index = index + 1
            rule_of_current = rules_to_combine[current_index]
            rule_of_next = rules_to_combine[next_index]

            set_of_current = rule_of_current.get_set()
            set_of_next = rule_of_next.get_set()
            diff = set(set_of_next) - set(set_of_current)  # 找到两个当前层级的项集的差值

            for item in diff:
                new_set = set_of_current + (item,)
                related_rule = self.create_related_rule(X_std, new_set, metric, min_threshold)

                if related_rule is None:
                    continue

                related_rule.set_pre_set(set_of_current)
                related_rule.set_post_set((item,))

                # 计算置信度 X ==> Y，p(Y|X)=p(XY)/p(X)
                confidence = related_rule.get_metric("support") / rule_of_current.get_metric("support")
                related_rule.set_metric("confidence", confidence)
                # 计算提升度 lift(A==>B) = p(B|A)/p(B)
                item_support_value = self.get_first_rule_support(item)
                lift = related_rule.get_metric("support") / item_support_value
                related_rule.set_metric("lift", lift)

                related_rule_list.append(related_rule)

        self.related_rule_map[layer_of_current + 1] = related_rule_list

        return len(related_rule_list)

    def get_first_rule_support(self, item_set):
        """
        为了计算提升度，需要获得单元素项集的支持度
        """
        rule_of_first_layer = self.related_rule_map[0]
        for rule in rule_of_first_layer:
            if set(rule.get_set()) == set(item_set):
                return rule.get_metric("support")

    def print(self):
        """
        打印关联规则
        :return:
        """
        for layer_index in self.related_rule_map:
            rule_in_layer = self.related_rule_map[layer_index]
            for rule in rule_in_layer:
                rule.print()
