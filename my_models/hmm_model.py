import numpy as np

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
        observed_state = np.unique(X)

        # 初始化模型参数值
        self.init_model_params(observed_state)

        # 计算前向概率
        alpha, p = self.calc_forward_p(self.init_states, self.transition_matrix, self.production_matrix, X)

        # 计算后向概率
        beta = self.calc_backward_p(X)

        # 计算状态占用概率
        gamma = self.calc_state_used_p(alpha, beta)

        # 计算状态转移期望
        xi = self.calc_state_transfered_q(X, alpha, beta)

        # 更新模型参数
        self.update_model_params(X, observed_state, gamma, xi)

    def predict_p(self, observed_sequence, pi=None, A=None, B=None):
        """
        已知模型参数及观测序列，求观测序列的概率
        """
        if pi is None:
            pi = self.init_states
        if A is None:
            A = self.transition_matrix
        if B is None:
            B = self.production_matrix

        alpha, p = self.calc_forward_p(pi, A, B, observed_sequence)

        return alpha, p


    def predict_i(self, observed_sequence, pi=None, A=None, B=None):
        """
        已知模型及观测序列，找到隐藏状态的最可能序列
        """
        if pi is None:
            pi = self.init_states
        if A is None:
            A = self.transition_matrix
        if B is None:
            B = self.production_matrix

        num_of_state = self.n_states
        num_of_sequence = len(observed_sequence)

        # 初始化delta和psi矩阵
        delta = np.zeros((num_of_sequence, num_of_state))
        psi = np.zeros((num_of_sequence, num_of_state), dtype=np.int8)

        # 初始化t=0
        delta[0] = pi * B[:, observed_sequence[0]]
        psi[0] = 0 # 无前驱状态

        # 递推计算
        for t in range(1, num_of_sequence):
            max_delta_a = 0
            for j in range(num_of_state):
                trans_prob = delta[t - 1] * A[:, j]
                max_trans = np.max(trans_prob)
                delta[t, j] = max_trans * B[j, observed_sequence[t]]
                psi[t, j] = np.argmax(trans_prob)

        # 终止
        max_prob = np.max(delta[-1])
        best_path = [np.argmax(delta[-1])]

        # 回溯路径
        for t in range(num_of_sequence - 2, -1, -1):
            best_path.insert(0, psi[t + 1, best_path[0]])

        return best_path, max_prob

    def init_model_params(self, observed_state):
        """
        初始化模型参数值
        """
        # 初始化隐藏状态的初始值π
        init_states = np.abs(np.random.rand(1, self.n_states))
        init_states = MyHMM.onilize(init_states)

        self.init_states = init_states

        # 初始化隐藏状态的转移概率矩阵A
        transition_matrix = np.abs(np.random.rand(self.n_states, self.n_states))
        transition_matrix = MyHMM.onilize(transition_matrix)

        self.transition_matrix = transition_matrix

        # 初始化隐藏状态的生成观测序列的矩阵B
        production_matrix = np.abs(np.random.rand(self.n_states, len(observed_state)))
        self.production_matrix = MyHMM.onilize(production_matrix)

    @staticmethod
    def onilize(data):
        """
        对矩阵进行归一化
        """
        total = np.sum(data, axis=1).reshape(-1, 1)

        onilized_data = data / total
        
        return onilized_data

    def calc_forward_p(self, init_states, transition_matrix, production_matrix, observed_sequence):
        """
        计算前向概率
        """
        alpha = np.zeros((len(observed_sequence), self.n_states))

        alpha[0] = init_states * production_matrix[:, observed_sequence[0]]
        for index in range(1, len(observed_sequence)):
            alpha[index] = (alpha[index - 1] @ transition_matrix) * production_matrix[:, observed_sequence[index]]

        sequence_p = np.sum(alpha[-1])

        return alpha, sequence_p

    def calc_backward_p(self, observed_state):
        """
        计算后向概率
        """
        beta = np.zeros((len(observed_state), self.n_states))

        beta[-1] = 1
        for index in range(len(observed_state) -2, -1, -1):
            beta[index] = self.transition_matrix.dot(self.production_matrix[:, observed_state[index + 1]] * beta[index + 1])
        
        return beta

    def calc_state_used_p(self, alpha, beta):
        """
        计算状态占用概率
        """
        p_0 = np.sum(alpha[-1, :])
        gamma = alpha * beta / p_0

        return gamma
    
    def calc_state_transfered_q(self, observed_state, alpha, beta):
        """
        计算状态转移期望
        """
        # 计算总概率 P(O|λ)
        P_O = np.sum(alpha[-1, :])

        num_of_observed_state = len(observed_state)
        Xi = np.zeros((num_of_observed_state - 1, self.n_states, self.n_states))

        for index in range(num_of_observed_state - 1):
            observed_state_t = observed_state[index + 1]

            # 计算分子部分：α_t[i] * A[i,j] * B[j,obs] * β_{t+1}(j)
            numerator = alpha[index, :, None] * self.transition_matrix * self.production_matrix[:, observed_state_t][None, :] * beta[index + 1, :][None, :]

            # 归一化
            Xi[index] = numerator / P_O
        
        return Xi

    def update_model_params(self, X, observed_state, gamma, xi):
        """
        更新模型参数
        """
        # 更新π
        new_init_states = gamma[0, :]

        # 更新转移矩阵
        new_transition_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            sum_gamma_i = np.sum(gamma[:-1, i])  # 分母：状态i的总占用次数（除最后时刻）
            for j in range(self.n_states):
                new_transition_matrix[i, j] = np.sum(xi[:, i, j]) / sum_gamma_i
        
        # 更新生成观测序列矩阵
        new_production_matrix = np.zeros((self.n_states, len(observed_state)))
        for i in range(self.n_states):
            sum_gamma_i = np.sum(gamma[:, i])  # 分母：状态i的总占用次数
            for k in range(len(observed_state)):
                # 分子：在状态i下观测到k的期望次数
                mask = (X == k)
                new_production_matrix[i, k] = np.sum(gamma[mask, i]) / sum_gamma_i

        self.init_states = new_init_states
        self.transition_matrix = new_transition_matrix
        self.production_matrix = new_production_matrix

    def print(self):
        """
        打印模型的参数信息
        :return:
        """
        print(f"init_state:\n{self.init_states}")
        print(f"transition_matrix:\n{self.transition_matrix}")
        print(f"production_matrix:\n{self.production_matrix}")
