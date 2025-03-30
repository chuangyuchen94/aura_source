from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import mean_absolute_error

X_all, y_all = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")

# 先找到典型的值，作为训练用的数据集
k_means_model = KMeans(n_clusters=50, random_state=42)
X_for_trained_distance = k_means_model.fit_transform(X_train)
standard_index = np.argmin(X_for_trained_distance, axis=0)

# 用典型值进行训练
logistic_model = LogisticRegression(random_state=0)
logistic_model.fit(X_train[standard_index], y_train[standard_index])
y_pred = logistic_model.predict(X_test)
print(f"Accuracy: {logistic_model.score(X_test, y_test)}")