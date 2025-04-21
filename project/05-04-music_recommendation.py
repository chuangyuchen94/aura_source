import pandas as pd
import numpy as np

def load_data(path=None):
    if path is None:
        path = "../data/train_triplets.txt"

    data = pd.read_csv(path)
    print(data.head(5))

    return data

if "__main__" == __name__:
    load_data()
