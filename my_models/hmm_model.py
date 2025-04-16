
"""
实现隐马尔科夫模型
"""

class MyHMM:
    def __init__(self, n_states):
        """
        初始化方法
        """
        self.n_states = n_states # 隐藏状态的数量
        self.init_states = None # 隐藏状态的初始值π
        self.transition_matrix = None # 隐藏状态的转移矩阵A
        self.production_matrix = None # 隐藏状态生成观测序列的矩阵B

    def fit(self, X, y=None):
        """
        实现HMM的学习问题：已知观测序列，求模型参数
        """
        pass

    def predict_p(self, observed_sequence):
        """
        已知模型参数及观测序列，求观测序列的概率
        """
        pass

    def predict_i(self, observed_sequence):
        """
        已知模型及观测序列，找到最可能的隐藏状态序列
        """