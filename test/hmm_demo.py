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

    print(f"\n--------评估问题（已知 λ 和 O，求 P(O|λ) ）--------")
    observed_sequence = np.array([0, 1, 0]).reshape(-1)
    print(f"observed_sequence 1: {observed_sequence}")
    alpha_1, observed_p_1 = hmm_model.predict_p(observed_sequence)
    print(f"观测序列概率：{observed_p_1} | alpha：{alpha_1}")

    observed_sequence_2 = np.array([1, 0, 1]).reshape(-1)
    print(f"observed_sequence 2: {observed_sequence_2}")
    alpha_2, observed_p_2 = hmm_model.predict_p(observed_sequence_2)
    print(f"观测序列概率：{observed_p_2} | alpha：{alpha_2}")

