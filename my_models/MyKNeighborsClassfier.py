from collections import Counter
import numpy as np

class MyKNeighborsClassfier(object):
    """
        自定义KNN分类器
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("训练集必须是二维数组，结果集必须是一维数组")
        if X.shape[0] != y.shape[0]:
            raise ValueError("训练集样本数必须与结果集样本数一致")

        self.X = X
        self.y = y

    def predict(self, X):
        # 找出与X距离最相近的n_neighbors个样本
        X_pred = np.array(X)
        result = []

        for x in X_pred:
            distance = ((self.X - x) ** 2).sum(axis=1)
            indices = np.argsort(distance)[: self.n_neighbors]
            lables = self.y[indices]
            lable = Counter(lables).most_common(1)[0][0]
            result.append(lable)

        return np.array(result)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import numpy as np

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    knn_model = MyKNeighborsClassfier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    print(y_pred)
    print((y_test == y_pred).mean())
