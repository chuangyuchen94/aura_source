from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data = np.array(pd.read_csv("../data/iris.csv"))
print(f"data: {data.shape}")

X_all = data[:, :-1]
y_all = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
print(f"result: {y_pred}")

mea = sum(y_pred == y_test) / len(y_test) * 100
print(f"predict precision: {mea}")