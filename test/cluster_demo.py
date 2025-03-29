from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

centers = [
    [0.2, 2.3],
    [-1.5, 2.3],
    [-2.8, 1.8],
    [-2.8, 2.8],
    [-2.8, 1.3],
]

blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=centers, cluster_std=blob_std, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=5)
plt.show()

def show_cluster(centers, cluster_labels, distance):
    plt.figure(figsize=(10, 6))
    plt.xlabel(xlabel="x1")
    plt.ylabel(ylabel="y2")

    for label in np.unique(cluster_labels):
        mask = (cluster_labels == label)
        plt.scatter(
            X[:, 0][mask], X[:, 1][mask],
            s=10,
            marker= ".", #['o', '^', 's'][label],  # 不同形状
            c=['red', 'blue', 'green', 'purple', 'pink'][label],  # 不同颜色
            label=f'Cluster {label}'
        )

    # 绘制质心
    for label in range(len(centers)):
        plt.scatter(
            centers[:, 0][label], centers[:, 1][label],
            s=100,  # 更大尺寸
            marker='^',  # 星形标记
            c=['red', 'blue', 'green', 'purple', 'pink'][label],  # 金色
            edgecolors='black',
            linewidths=1,
            label='Centers'
        )

    plt.title(f"distance={distance}|{centers}")
    plt.legend()
    plt.show()

k_means_model = KMeans(n_clusters=5, init="k-means++", n_init=5)
k_means_model.fit(X)
cluster_centers = k_means_model.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], s=5)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            s=100,
            marker='^',
            c="red",
            edgecolors='black',
            linewidths=1,
            label='Centers'
            )
plt.show()

# 测试不同k值对应的评估值
inertias = []
sil_scores = []
for k in range(2, 21):
    kmean_model = KMeans(n_clusters=k)
    kmean_model.fit(X)
    inertias.append(kmean_model.inertia_)
    sil_scores.append(silhouette_score(X, kmean_model.labels_))

plt.plot(range(2, 21), inertias)
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 示例用雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
plt.xlabel("k值")
plt.ylabel("评估值：距离")
plt.title("不同k值的评估值变化情况")
plt.show()

plt.plot(range(2, 21), sil_scores)
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 示例用雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
plt.xlabel("k值")
plt.ylabel("轮廓系数")
plt.title("不同k值的轮廓系数变化情况")
plt.show()

# K-MEANS算法存在的问题
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]

plt.scatter(X[:, 0], X[:, 1], s=2, c="black")
plt.show()
