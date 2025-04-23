from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

class ScoringBySVD：
    """
    通过SVD矩阵分解，求得“用户-物品”的评分
    """
    def __init__(self):
        """
        初始化方法
        """
        pass

    def fit(self, data):
        """
        将传入的数据，分解为U、S、V三个矩阵
        """
        pass

    def predict(self, feature):
        """
        预测评分
        """
        pass