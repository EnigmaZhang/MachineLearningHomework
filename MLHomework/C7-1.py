import pandas as pd
import numpy as np
import random
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SkKmeans


class KMeans:
    def __init__(self, data, k, centroids_choice=0, times=100):
        self.data = data
        self.k = k
        self.sample_numbers = data.shape[0]
        self.dim = data.shape[1]

        # Choose init ways.
        if centroids_choice == 1:
            self.centroids = self._init_uniform_centroids(self.data, self.k)
        elif centroids_choice == 2:
            self.centroids = self._init_best_centroids(self.data, self.k, times=times)
        elif centroids_choice == 4:
            self.centroids = self._init_kmeansplusplus_centroids(self.data, self.k)
        else:
            self.centroids = self._init_random_centroids(self.data, self.k)

    def fit(self):
        # Errors between sample and centroid.
        assessment = np.zeros(self.sample_numbers)
        label = np.zeros(self.sample_numbers, dtype=np.int)
        converged = False

        while not converged:
            old_centroids = np.copy(self.centroids)
            for i in range(self.sample_numbers):
                # Find the nearest centroid and track with the label.
                min_distance, min_index = np.inf, -1
                for j in range(self.k):
                    distance = KMeans.euclid_distance(self.data[i], self.centroids[j])
                    if distance < min_distance:
                        min_distance, min_index = distance, j
                        label[i] = j
                assessment[i] = KMeans.euclid_distance(data[i], self.centroids[label[i]]) ** 2

            # Update centroids.
            for i in range(self.k):
                # Avoid mean of empty slice.
                if len(data[label == i]) != 0:
                    self.centroids[i] = np.mean(data[label == i], axis=0)
            converged = KMeans._converged(old_centroids, self.centroids)

        return self.centroids, label, np.sum(assessment)

    @staticmethod
    def _converged(centroids1, centroids2):
        set1 = set([tuple(i) for i in centroids1])
        set2 = set([tuple(i) for i in centroids2])
        return set1 == set2

    def _init_random_centroids(self, data, k):
        centroids = np.zeros((self.sample_numbers, self.dim))
        for i in range(k):
            index = random.randint(0, self.sample_numbers - 1)
            centroids[i, :] = data[index, :]
        return centroids

    def _init_uniform_centroids(self, data, k):
        centroids = np.zeros((self.sample_numbers, self.dim))
        data = sorted(data, key=lambda x: x[0] ** 2 + x[1] ** 2)
        data = np.asarray(data)
        for index, i in enumerate(range(0, len(data), len(data) // (k - 1))):
            centroids[index, :] = data[i, :]
        return centroids

    # times is the parameter for find the best centroids repeatedly.
    def _init_best_centroids(self, data, k, times):
        best_total_assessment = np.inf
        best_centroids = np.zeros((self.sample_numbers, self.dim))
        for _ in range(times):
            self.centroids = self._init_random_centroids(self.data, self.k)
            centroids, label, total_assessment = self.fit()
            if total_assessment < best_total_assessment:
                best_centroids = centroids

        return best_centroids

    def _init_kmeansplusplus_centroids(self, data, k):
        centroids = np.zeros((self.sample_numbers, self.dim))
        index = random.randint(0, self.sample_numbers - 1)
        # Fisrt random
        centroids[0, :] = data[index, :]
        farthest_distance = 0.0
        farthest_point = np.zeros(data[0].shape)

        # Rest choose points farthest with all centroids.
        for i in range(k - 1):
            for points in data:
                distance = np.sum([KMeans.euclid_distance(points, centroid) for centroid in centroids])
                if distance > farthest_distance:
                    farthest_point = points
            centroids[i + 1, :] = farthest_point

        return centroids


    @staticmethod
    def euclid_distance(a, b):
        return sqrt(np.sum(np.power((a - b), 2)))


data = pd.read_csv(r"./data/watermelon4.csv")
data = np.asarray(data[["density", "sugar"]].values)

# Random centroids
# k = 2, 3, 4
ks = [2, 3, 4]
print("Random centroids:")
for k in ks:
    kmeans = KMeans(data, k, 0)
    centroids, label, total_assessment = kmeans.fit()
    for i in range(k):
        plt.scatter(data[label == i][:, 0], data[label == i][:, 1])
    plt.scatter(centroids[:k, 0], centroids[:k, 1], c="red")
    plt.show()
    print("k = {} total_assessment = {}".format(k, total_assessment))

# Uniform centroids
print("Uniform centroids:")
k = 3
kmeans = KMeans(data, k, 1)
centroids, label, total_assessment = kmeans.fit()
for i in range(3):
    plt.scatter(data[label == i][:, 0], data[label == i][:, 1])
plt.scatter(centroids[:k, 0], centroids[:k, 1], c="red")
plt.show()
print("k = {} total_assessment = {}".format(k, total_assessment))

# Best centroids
k = 3
kmeans = KMeans(data, k, 2)
centroids, label, total_assessment = kmeans.fit()
for i in range(3):
    plt.scatter(data[label == i][:, 0], data[label == i][:, 1])
plt.scatter(centroids[:k, 0], centroids[:k, 1], c="red")
plt.show()
print("Best centroids:")
print("k = {} total_assessment = {}".format(k, total_assessment))

# KMeans++
k = 3
kmeans = KMeans(data, k, 3)
centroids, label, total_assessment = kmeans.fit()
for i in range(3):
    plt.scatter(data[label == i][:, 0], data[label == i][:, 1])
plt.scatter(centroids[:k, 0], centroids[:k, 1], c="red")
plt.show()
print("KMeans++ centroids:")
print("k = {} total_assessment = {}".format(k, total_assessment))

kmeans = SkKmeans(n_clusters=3)
prediction = kmeans.fit_predict(data)
centroids = kmeans.cluster_centers_
plt.scatter(data[:, 0], data[:, 1], c=prediction)
plt.scatter(centroids[:, 0], centroids[:, 1], c="red")
plt.show()
print("Sklearn: {}".format(kmeans.inertia_))
