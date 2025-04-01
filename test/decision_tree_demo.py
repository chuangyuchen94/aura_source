from my_models.decision_tree import create_dataset, MyDecisionTree
from sklearn.model_selection import train_test_split

data, label = create_dataset()
print(f"data: {data.shape}")

X_all = data[:, :-1]
y_all = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

tree_model = MyDecisionTree()
tree_model.fit(X_train, y_train)
result = tree_model.predict(X_test)
print(f"result: {result}")
y_pred = result[:, 0]
mea = sum(y_pred == y_test) / len(y_test) * 100
print(f"predict precision: {mea}")