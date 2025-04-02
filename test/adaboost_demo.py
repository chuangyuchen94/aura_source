from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = np.array(pd.read_csv("../data/iris.csv"))
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=100,
    learning_rate=0.05,
    random_state=42
)

ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_test)
print(f"ada score: {accuracy_score(y_test, y_pred)}")

decision_model = DecisionTreeClassifier(max_depth=2)
decision_model.fit(X_train, y_train)
y_pred = decision_model.predict(X_test)
print(f"decision_model score: {accuracy_score(y_test, y_pred)}")