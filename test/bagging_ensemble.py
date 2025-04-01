from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 创建测试数据
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'yo')
plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'bs')
plt.show()


bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=100,
    bootstrap=True,
    random_state=42
)

decision_model = DecisionTreeClassifier(random_state=42)

bagging_model.fit(X_train, y_train)
y_pred_bagging = bagging_model.predict(X_test)
print(f"bagging_model: {accuracy_score(y_test, y_pred_bagging)}")

decision_model.fit(X_train, y_train)
y_pred_decision = decision_model.predict(X_test)
print(f"decision_model: {accuracy_score(y_test, y_pred_decision)}")

# OOB策略
bagging_model_2 = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
    oob_score=True
)
bagging_model_2.fit(X_train, y_train)
y_pred_oob_score = bagging_model_2.predict(X_test)
print(f"oob_score: {accuracy_score(y_test, y_pred_oob_score)}")

from sklearn.ensemble import RandomForestClassifier
random_model = RandomForestClassifier(
    n_estimators=500,
    n_jobs=-1,
    random_state=42
)
random_model.fit(X_train, y_train)
for importance in random_model.feature_importances_:
    print(importance)


# mnist数据集输入
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = np.array(mnist["data"]), np.array(mnist["target"])
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print(f"Begin to train random_forest_model..")
random_forest_model_3 = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
random_forest_model_3.fit(X, y)
print(f"After trainning: {random_forest_model_3.feature_importances_.shape}")

def plot_digit(importances):
    image = importances.reshape(28, 28)
    im = plt.imshow(image, cmap=matplotlib.cm.hot)
    plt.axis("off")
    charbar = plt.colorbar(im)

plot_digit(random_forest_model_3.feature_importances_)
