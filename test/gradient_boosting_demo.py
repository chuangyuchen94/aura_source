from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gradient_model = GradientBoostingClassifier(
    max_depth=2,
    random_state=42,
    warm_start=True
)

min_error = float("inf")
stop_counting = 0

for n_estimator in range(1, 120):
    gradient_model.n_estimators = n_estimator
    gradient_model.fit(X_train, y_train)
    y_pred = gradient_model.predict(X_test)
    error_value = mean_squared_error(y_test, y_pred)

    if error_value < min_error:
        min_error = error_value
        stop_counting = 0
    else:
        stop_counting += 1
        if stop_counting >= 5:
            break

print(f"best number of estimator: {gradient_model.n_estimators}")
