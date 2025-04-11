import numpy as np

"""
实现贝叶斯算法：求，给定数据的前提下，某个场景出现的概率
    P(h+|D)：表示垃圾邮件的概率
    其中：
        D是邮件（内容），
        h+表示垃圾邮件
        h-表示正常邮件
    按照贝叶斯算法
        P(h+|D) = P(D|h+) * P(h+) / P(D)，
        P(h-|D) = P(D|h-) * P(h-) / P(D)，
        P(D)可以不用管，毕竟最终比较的是P(h+|D)和P(h-|D)的大小即可，并不需要得出准确值
        P(h+)、P(h-)是先验概率，只要统计现有的即可
        D是一封邮件，可以拆成一个个的单词，即，邮件由单词组成，w1, w2, w3..., wn
            进一步的，P(D|h+) = P(w1,w2...wn|h+) = P(w1|h+)*P(w2|h+)...P(wn|h+)
            P(wn|h+)表示：垃圾邮件中，出现单词wn的概率，这里是利用了朴素贝叶斯，假设单词之间的出现的独立的，不存在依赖关系
    具体的算法实现
        P(h+)、P(h-)：是基于对训练数据的统计得出——训练数据中，垃圾邮件、正常邮件的比例
        P(wn|h+)：垃圾邮件中，单词wn出现的比例——统计训练的垃圾邮件中，单词wn出现的比例
        P(wn|h-)：正常邮件中，单词wn出现的比例——统计训练集的正常邮件中，单词wn出现的比例
        P(h+|D) = P(D|h+) * P(h+) / P(D) 近似 P(w1|h+)*P(w2|h+)...P(wn|h+) * P(h+)
        P(h-|D) = P(D|h-) * P(h-) / P(D) 近似 P(w1|h-)*P(w2|h-)...P(wn|h-) * P(h-)
        综合P(h+|D)与P(h-|D)：
            ln(P(hi|D)) = X @ ln(P(Wj|hi)) + ln(P(hi))，
                i代表+和-；
                Wj的汇总就是D
        例如：
            X = (50, 844), y = (50,) , 
            P(Wj|hi) = (844, 1) ==> 垃圾邮件中，词库中的单词出现的概率（频率） 
            P(hi)为标量(1,)
"""
class MyNaiveBayes:
    def __init__(self, alpha=1.0, output_func=None):
        """
        初始化方法
        """
        self.pre_of_label = {} # 每个标签值的先验概率
        self.label_p_of_label = {} # 每个类别的概率
        self.alpha = alpha
        self.output_func = output_func

    def fit(self, X, y):
        """
        训练方法
        1、计算各个标签的类别下，不同的特征值出现的比例
        :param X:
        :param y:
        :return:
        """
        # 计算各个标签的类别下，不同的特征值出现的比例
        label_unique = np.unique(y)
        nun_of_features = X.shape[1]

        for label in label_unique:
            label_index = np.where(y == label)[0]
            X_of_label = X[label_index]
            count_of_feature = np.sum(X_of_label, axis=0)
            total_of_feature = np.sum(X_of_label)
            p_of_label = (count_of_feature + self.alpha) / (total_of_feature + self.alpha * nun_of_features)
            self.label_p_of_label[label] = p_of_label.copy()

        # 计算每个标签的先验概率
        for label in label_unique:
            self.pre_of_label[label] = np.sum(y==label) / len(y)


    def predict(self, X_test):
        """
        预测方法
        公式：ln(P(hi|D)) = X @ ln(P(Wj|hi)) + ln(P(hi))
        :param X_test:
        :return:
        """
        label_p_list = []
        pre_b_list = []

        for label in self.label_p_of_label:
            label_p_list.append(self.label_p_of_label[label])
            pre_b_list.append(self.pre_of_label[label])

        label_p = np.array(label_p_list).T
        pre_b = np.array(pre_b_list).reshape(-1, 2)

        ln_pred = X_test.dot(np.log(label_p)) + np.log(pre_b)

        return np.argmax(ln_pred, axis=1)

