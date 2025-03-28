import numpy as np
import matplotlib.pyplot as plt
from my_models.k_means_cluster import MyKMeansCluster

# 构造数据点：二维数组
size = 100
np.random.seed(0)
data_part1 = np.random.randn(size, 2) + np.array([2, 0])
data_part2 = np.random.randn(size, 2) + np.array([6, 4])
data_part3 = np.random.randn(size, 2) + np.array([2, 6])
data = np.concatenate((data_part1, data_part2, data_part3), axis=0)

# plt.figure(figsize=(5, 5))
plt.scatter(data[:,0], data[:,1], s=5)
plt.xlabel(xlabel="x")
plt.ylabel(ylabel="y")
plt.show()

def show_cluster(centers, cluster_labels, distance):
    plt.xlabel(xlabel="x")
    plt.ylabel(ylabel="y")

    for label in np.unique(cluster_labels):
        mask = (cluster_labels == label)
        plt.scatter(
            data[:, 0][mask], data[:, 1][mask],
            s=10,
            marker= ".", #['o', '^', 's'][label],  # 不同形状
            c=['red', 'blue', 'green'][label],  # 不同颜色
            label=f'Cluster {label}'
        )

    # 绘制质心
    for label in range(len(centers)):
        plt.scatter(
            centers[:, 0][label], centers[:, 1][label],
            s=100,  # 更大尺寸
            marker='^',  # 星形标记
            c=['red', 'blue', 'green'][label],  # 金色
            edgecolors='black',
            linewidths=1,
            label='Centers'
        )

    plt.title(f"distance={distance}|{centers}")
    plt.legend()
    plt.show()

k_means_model = MyKMeansCluster(k_estimator=3, callback=show_cluster)
# k_means_model = MyKMeansCluster(k_estimator=3)
k_means_model.fit(data)

