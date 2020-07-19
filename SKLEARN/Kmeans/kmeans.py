#!/usr/bin/python

import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
x1 = [random.randint(0,100) for _ in range(100)]
x2 = [random.randint(0,100) for _ in range(100)]
tranning_data = [[x1, x2] for x1,x2 in zip(x1, x2)]
cluster = KMeans(n_clusters=6, max_iter=500)
cluster.fit(tranning_data)
centers = cluster.cluster_centers_#每个中心一个代号1，2，3，，，该列表为每个训练样本所属中心点
cluster.labels_
prediction = cluster.predict([[55,55]])
if __name__ == "__main__":
    plt.scatter(x1, x2, color = "red")
    plt.scatter(centers[:,0], centers[:,1], color="black")
    plt.scatter(centers[prediction[0]][0], centers[prediction[0]][1],color="blue")
    plt.show()
