from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

plt.scatter(X[:, 0], X[:, 1], c="blue", s=3)
plt.title("uncluttered data")
plt.show()

dbscan_model = DBSCAN(eps=0.05, min_samples=5)
dbscan_model.fit(X)

label_of_sample = dbscan_model.labels_
label_of_kind = np.unique(label_of_sample)

colors = ["blue", "red", "green", "purple", "orange", "pink", "steelblue", "gold", "darkviolet", "limegreen"]
for label in label_of_kind:
    cluster_index = label_of_sample == label

    if -1 == label:
        plt.scatter(X[cluster_index, 0], X[cluster_index, 1], c="black", s=3)
    else:
        plt.scatter(X[cluster_index, 0], X[cluster_index, 1], c=colors[label], s=3)

plt.title("clustered data")
plt.show()
