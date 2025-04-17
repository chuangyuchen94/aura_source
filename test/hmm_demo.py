import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 直接添加 test 的父目录（aura_source）

from my_models.hmm_model import MyHMM
import numpy as np
from hmmlearn import hmm

def load_data():
    """
    加载数据
    """
    np.random.seed(42)

    model = hmm.GaussianHMM(n_components=3, covariance_type="full")
    model.startprob_ = np.array([0.6, 0.3, 0.1])
    model.transmat_ = np.array([[0.7, 0.2, 0.1],
                                [0.3, 0.5, 0.2],
                                [0.3, 0.3, 0.4]])
    model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
    model.covars_ = np.tile(np.identity(2), (3, 1, 1))
    X, Z = model.sample(100)

    return X, Z

if "__main__" == __name__:
    hmm_model = MyHMM(n_states=3)
    
    X, z = load_data()
    print(f"X shape: {X.shape}")
    print(f"z shape: {z.shape}")

    hmm_model.fit(z)
    hmm_model.print()
    