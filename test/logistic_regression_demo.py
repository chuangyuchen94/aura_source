from my_models.logic_regression import MyLogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 鸢尾花数据集的多分类任务
def load_iris_data():
    data = pd.read_csv("../data/iris.csv")
    data_size = data.shape[0]
    print(f"data.shape: {data.shape}")

    X_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    y_columns = ["class"]

    X_all = np.array(data[X_columns]).reshape(data_size, len(X_columns))
    y_all = np.array(data[y_columns]).reshape(data_size, len(y_columns))
    print(f"X_all shape: {X_all.shape}")
    print(f"y_all shape: {y_all.shape}")

    return X_all, y_all


if "__main__" == __name__:
    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    logistic_model = MyLogisticRegression()
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)

    mea = sum(y_test.reshape(y_test.shape[0]) == y_pred) / len(y_test) * 100
    print(f"predict precision: {mea}")