from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np

iris_data = datasets.load_iris()
X = iris_data["data"][:, (2, 3)]
y = (iris_data["target"] == 2).astype(int)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

svm_pipeline = Pipeline(
    [
        ("std", StandardScaler()),
        ("linear svc", LinearSVC(C=1)),
    ]
)

svm_pipeline.fit(X, y)
svm_pipeline.predict([[5.5, 1.7]])