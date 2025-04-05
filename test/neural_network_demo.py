from my_models.neural_network import MyNeuralNetwork
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = np.array(pd.read_csv("../data/mnist-demo.csv"))[:1000,:]
X_all = data[:, 1:]
y_all = data[:, :1]

print(f"X_all shape: {X_all.shape}")
print(f"y_all shape: {y_all.shape}")

print(f"X_all head: {X_all[:5,:]}")
print(f"y_all head: {y_all[:5]}")

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)

# 初始化神经网络模型
neural_model = MyNeuralNetwork()
neural_model.fit(X_train, y_train)
neural_model.predict(X_test)
