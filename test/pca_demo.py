import pandas as pd
import numpy as np
from my_models.principle_component_analysis import MyPCATransformer

def load_data():
    data = pd.read_csv(filepath_or_buffer="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
    data_all = np.array(data)

    X = data_all[:, :-1]
    y = data_all[:, -1]

    return X, y

if "__main__" == __name__:
    X, y = load_data()

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    pca_transformer = MyPCATransformer()
    pca_transformer.fit(X)
    X_pca = pca_transformer.transform(X)
    print(f"X_pca shape: {X_pca.shape}")

