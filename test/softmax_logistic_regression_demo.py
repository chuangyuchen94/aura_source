from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

X, y = load_iris(return_X_y=True)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# 初始化多分类的逻辑分类实例
multi_logistic_model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
multi_logistic_model.fit(X_train, y_train)
y_pred = multi_logistic_model.predict(X_test)
mea = mean_absolute_error(y_pred, y_test)
print(f"mea : {mea}")
print(f"precision: {accuracy_score(y_test, y_pred)}")

precision = sum(y_pred == y_test)/len(y_test) * 100
print(f"precision 2: {accuracy_score(y_test, y_pred)}")
print(f"precision 2: {precision}")