import pandas as pd
from my_models.linear_discrimanant_analyst import MyLDA

def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    data = pd.read_csv(url)

    return data

if "__main__" == __name__:
    data = load_data()
    print(f"data: {data}")
    print(f"data's shape: {data.shape}")
