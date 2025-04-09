from my_models.neural_network import MyNeuralNetwork
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data = np.array(pd.read_csv("../data/mnist-demo.csv"))
X_all = data[:, 1:]
y_all = data[:, :1]

print(f"X_all shape: {X_all.shape}")
print(f"y_all shape: {y_all.shape}")

print(f"X_all head: {X_all[:5,:]}")
print(f"y_all head: {y_all[:5]}")

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)

# 初始化神经网络模型
neural_model = MyNeuralNetwork(max_iter=1000, learning_rate=1)
losses = neural_model.fit(X_train, y_train)
iter_num = range(len(losses))

# plt.plot(iter_num, losses)
# plt.show()

y_pred = neural_model.predict(X_test)
score = accuracy_score(y_test, y_pred) * 100
print(f"end of predict, score: {score}")