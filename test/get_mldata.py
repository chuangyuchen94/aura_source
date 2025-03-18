import pandas as pd

mldata = pd.read_csv("../data/mnist-demo.csv")

print(mldata.columns)
print(mldata.shape)