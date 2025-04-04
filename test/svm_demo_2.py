import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt

x0 = np.linspace(0, 5.5, 200)
pred_1 = 5 * x0 - 20
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5

iris_data = datasets.load_iris()
X, y = iris_data["data"][:, (2, 3)], iris_data["target"]
sov_or_vet = (y==0) | (y==1)
X_train, y_train = X[sov_or_vet], y[sov_or_vet]

svc_model = SVC(kernel="linear", C=float("inf"))
svc_model.fit(X_train, y_train)
def plot_svc_decision_boundary(svm_clf, x_min, x_max , sv=True):
    w = svm_clf.coef_[0] # 权重参数
    b = svm_clf.intercept_[0] # 偏置项

    x0_new = np.linspace(x_min, x_max, 200)
    decision_boundary = -w[0]/w[1] * x0_new - b/w[1]
    margin = 1/w[1]

    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    # 显示支持向量
    if sv:
        svs = svm_clf.support_vectors_
        plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')

    # 画决策边界
    plt.plot(x0_new, decision_boundary, "k-", linewidth=2)

    # 画决策边界
    plt.plot(x0_new, gutter_up, "k--", linewidth=2)
    plt.plot(x0_new, gutter_down, "k--", linewidth=2)

plt.figure(figsize=(14, 4))
plt.subplot(121)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "ys")
plt.plot(x0, pred_1, "g--", linewidth=2)
plt.plot(x0, pred_2, "m-", linewidth=2)
plt.plot(x0, pred_3, "r-", linewidth=2)
plt.axis((0.0, 5.5, 0.0, 2))

plt.subplot(122)
plot_svc_decision_boundary(svc_model, 0, 5.5)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "ys")
plt.axis((0.0, 5.5, 0.0, 2))

plt.show()

