from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import numpy as np

class ScoringBySVD:
    """
    通过SVD矩阵分解，求得“用户-物品”的评分
    """
    def __init__(self, n_components):
        """
        初始化方法
        """
        self.n_components = n_components
        self.U = None # 用户潜在特征矩阵
        self.S = None # 特征重要性矩阵
        self.Vt = None # 物品潜在特征矩阵的转置

    def fit(self, X, y=None):
        """
        将传入的数据，分解为U、S、V三个矩阵
        """
        U, s, Vt = svds(X, self.n_components)
        S = np.diag(S)

        self.U = U
        self.S = S
        self.Vt = Vt

    def predict(self, X):
        """
        预测评分
        """
        scores = self.U[X] @ self.S @ self.Vt

        return scores
