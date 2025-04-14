import pandas as pd
import numpy as np
from my_models.linear_discrimanant_analyst import MyLDA

def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    data = np.array(pd.read_csv(url))

    column_name_of_feature = ["sepal_length", "sepal_width", "petal_length", "petal_width", ]
    column_name_of_target = ["class", ]

    X = data[:, :-1]
    y = data[:, -1:]

    return X, y

if "__main__" == __name__:
    X, y = load_data()
    print(f"data: \nX\n{X}\ny\n{y}")
    print(f"data's shape: {X.shape}, {y.shape}")

    lda_transformer = MyLDA(n_features=2)
    lda_transformer.fit(X, y)
