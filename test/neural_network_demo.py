from my_models.neural_network import MyNeuralNetwork
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from datetime import datetime

data = np.array(pd.read_csv("../data/mnist-demo.csv"))
X_all = data[:, 1:]
y_all = data[:, :1]

print(f"X_all shape: {X_all.shape}")
print(f"y_all shape: {y_all.shape}")

print(f"X_all head: {X_all[:5,:]}")
print(f"y_all head: {y_all[:5]}")

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)

# 初始化神经网络模型
neural_model = MyNeuralNetwork(max_iter=10, layer=[256, 128], learning_rate=0.1)
start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"begin to fit(train): {start_time}")

losses = neural_model.fit(X_train, y_train)

stop_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"end of fit(train): {stop_time}; on-going: {stop_time - start_time}")

iter_num = range(len(losses))

# plt.plot(iter_num, losses)
# plt.show()

y_pred = neural_model.predict(X_test)
score = accuracy_score(y_test, y_pred) * 100
print(f"end of predict, score: {score}")

grandient_norm = neural_model.grad_norm
plt.plot(range(len(grandient_norm)), grandient_norm)
plt.title("grandient_norm")
plt.show()

activation_mean = neural_model.activation_mean
plt.plot(range(len(activation_mean)), activation_mean)
plt.title("activation_mean")
plt.show()