from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = np.array(mnist["data"]), np.array(mnist["target"])

print(f"X shape: {X.shape}, y shape: {y.shape}")

X_part, X_val, y_part, y_val = train_test_split(X, y, test_size=10000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_part, y_part, test_size=10000, random_state=42)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

random_forest_model = RandomForestClassifier(random_state=42)
extra_tree_model = ExtraTreesClassifier(random_state=42)
linear_svc_model = LinearSVC(random_state=42)
mlp_model = MLPClassifier(random_state=42)

# 第一次训练：多个分类模型
estimators = [random_forest_model, extra_tree_model, linear_svc_model, mlp_model]
for estimator in estimators:
    print(f"training model: {estimator}")
    estimator.fit(X_train, y_train)

# 用第一次训练后的模型进行预测，将预测结果组装成新的测试数据
X_val_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)
for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)

print(f"X_val_predictions: {X_val_predictions}")

# 第二次训练：组装的第一次预测结果
second_level_estimator = RandomForestClassifier(n_estimators=1000, random_state=42)
second_level_estimator.fit(X_val_predictions, y_val)

# 用第二次训练的结果进行预测
X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)
for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)
y_pred = second_level_estimator.predict(X_test_predictions).reshape(-1, 1)
print(f"second level score: {mean_squared_error(y_pred, y_test)}")