from random import uniform, choice
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


import keys
import duration

key_occurrence = {}
for key in keys.keyName:
    if key in key_occurrence:
        key_occurrence[key] += 1
    else:
        key_occurrence[key] = 1

duration_dict = {}

for key, dur in zip(keys.keyName, duration.duration_list):
    if key in duration_dict:
        duration_dict[key] += dur
    else:
        duration_dict[key] = dur

piano_keys = [
    "C0", "C#0", "D0", "D#0", "E0", "F0", "F#0", "G0", "G#0", "A0", "A#0", "B0",
    "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1", "A1", "A#1", "B1",
    "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2",
    "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
    "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6", "A6", "A#6", "B6",
    "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7", "A7", "A#7", "B7",
    "C8"
]


def sorting_keys(key):
    return piano_keys.index(key)


sorted_occurrence = dict(sorted(key_occurrence.items(), key=lambda x: sorting_keys(x[0])))

sorted_durations = dict(sorted(duration_dict.items(), key=lambda x: sorting_keys(x[0])))

sorted_occurrence = {'C#1': 2, 'G#1': 6, 'C#2': 4, 'D#2': 4, 'E2': 14, 'F2': 2, 'F#2': 16, 'G#2': 37, 'A2': 16, 'B2': 6,
                     'C3': 12, 'C#3': 40, 'D3': 2, 'D#3': 29, 'E3': 48, 'F3': 4, 'F#3': 58, 'G3': 3, 'G#3': 150,
                     'A3': 38, 'A#3': 34, 'B3': 15, 'C4': 67, 'C#4': 104, 'D4': 32, 'D#4': 67, 'E4': 108, 'F4': 20,
                     'F#4': 98, 'G4': 18, 'G#4': 130, 'A4': 60, 'A#4': 20, 'B4': 36, 'C5': 38, 'C#5': 103, 'D5': 14,
                     'D#5': 90, 'E5': 71, 'F5': 15, 'F#5': 70, 'G5': 12, 'G#5': 49, 'A5': 19, 'A#5': 3, 'B5': 6,
                     'C6': 3, 'C#6': 10, 'D6': 6, 'D#6': 6, 'E6': 6, 'F6': 6, 'F#6': 6, 'G6': 6, 'G#6': 6, 'C#7': 6}
sorted_durations = {'C#1': 0.5703124999999858, 'G#1': 0.7786458333333286, 'C#2': 0.734375, 'D#2': 0.830729166666675,
                    'E2': 3.0677083333333535, 'F2': 0.6276041666666643, 'F#2': 2.174479166666721,
                    'G#2': 5.588541666666713, 'A2': 3.3463541666666465, 'B2': 2.0078125000000036,
                    'C3': 1.4348958333333215, 'C#3': 13.257812500000027, 'D3': 0.8515625, 'D#3': 17.656250000000025,
                    'E3': 8.338541666666647, 'F3': 3.184895833333343, 'F#3': 8.380208333333375,
                    'G3': 0.7473958333333428, 'G#3': 24.749999999999908, 'A3': 8.62239583333329,
                    'A#3': 12.585937499999954, 'B3': 3.2552083333333464, 'C4': 14.265624999999998,
                    'C#4': 17.992187499999872, 'D4': 7.708333333333325, 'D#4': 17.835937500000064,
                    'E4': 20.979166666666757, 'F4': 8.76041666666665, 'F#4': 24.536458333333385,
                    'G4': 8.677083333333346, 'G#4': 26.822916666666735, 'A4': 15.749999999999917,
                    'A#4': 5.041666666666639, 'B4': 7.312499999999988, 'C5': 11.111979166666673,
                    'C#5': 26.052083333333183, 'D5': 1.6770833333333073, 'D#5': 23.841145833333307,
                    'E5': 15.059895833333428, 'F5': 8.869791666666686, 'F#5': 13.414062500000032,
                    'G5': 2.247395833333343, 'G#5': 9.203125000000053, 'A5': 3.005208333333286,
                    'A#5': 1.4348958333333428, 'B5': 0.6927083333333144, 'C6': 1.9453124999999858,
                    'C#6': 5.177083333333343, 'D6': 0.7552083333333286, 'D#6': 0.8020833333332718,
                    'E6': 0.8854166666666998, 'F6': 0.6953124999999787, 'F#6': 0.7369791666666572,
                    'G6': 0.9036458333333499, 'G#6': 1.2317708333333357, 'C#7': 37.328125}

occurrence_values = list(sorted_occurrence.values())
duration_values = list(sorted_durations.values())

data = np.array(list(zip(occurrence_values, duration_values)))
# print(data)

norms_to_try = ['manhattan', 'euclidean', 'infinity', 'mahalanobis']
n_clusters_range = [3, 4, 5, 6, 7]  # Define your desired range of cluster numbers
max_iter_range = [19, 111, 60, 261, 322]
random_norm = choice(norms_to_try)
random_n_clusters = choice(n_clusters_range)
random_max_iter = choice(max_iter_range)


class KMeans:
    def __init__(self, n_clusters=1, max_iter=19, norm="euclidean"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.norm = norm

    def manhattan(self, x, y):
        return np.linalg.norm(x - y, ord=1)

    def euclidean(self, x, y):
        return np.linalg.norm(x - y)

    def infinity(self, x, y):
        return np.max(np.abs(x - y))

    def mahalanobis_distance(self, x, y, covariance_matrix):
        diff = x - y
        mahalanobis_dist = np.sqrt(np.dot(np.dot(diff, np.linalg.inv(covariance_matrix)), diff))
        return mahalanobis_dist

    def fit(self, X_train, norm='euclidean'):
        min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        '''
        self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]'''
        self.centroids = [np.random.uniform(min_, max_) for _ in range(self.n_clusters)]

        if norm == 'euclidean':
            distance_func = self.euclidean
        elif norm == 'manhattan':
            distance_func = self.manhattan
        elif norm == 'infinity':
            distance_func = self.infinity
        elif norm == 'mahalanobis':
            covariance_matrix = np.cov(data, bias=True)
            covariance_matrix = np.cov(X_train.T)
            distance_func = lambda x, y: self.mahalanobis_distance(x, y, covariance_matrix)
        else:
            raise ValueError("Invalid norm")

        iteration = 200
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = [distance_func(x, centroid) for centroid in self.centroids]
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]

            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = [self.euclidean(x, centroid) for centroid in self.centroids]
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs


custom_kmeans = KMeans(n_clusters=random_n_clusters, max_iter=random_max_iter, norm=random_norm)
custom_kmeans.fit(data)
custom_centroids, custom_labels = custom_kmeans.evaluate(data)

plt.scatter(data[:, 0], data[:, 1], c=custom_labels, cmap='viridis', alpha=0.5)
plt.scatter(np.array(custom_centroids)[:, 0], np.array(custom_centroids)[:, 1], c='red', marker='x', s=200,
            label='Centroids')
plt.title('K-Means Clustering by random number')
plt.legend()
plt.show()

custom_kmeans = KMeans(n_clusters=6, max_iter=11, norm="mahalanobis")
custom_kmeans.fit(data)
custom_centroids, custom_labels = custom_kmeans.evaluate(data)

plt.scatter(data[:, 0], data[:, 1], c=custom_labels, cmap='viridis', alpha=0.5)
plt.scatter(np.array(custom_centroids)[:, 0], np.array(custom_centroids)[:, 1], c='red', marker='x', s=200,
            label='Centroids')
plt.title('K-Means Clustering (6 clusters, 11 iterations, mahalanobis norm)')
plt.legend()
plt.show()

custom_kmeans = KMeans(n_clusters=6, max_iter=199, norm="mahalanobis")
custom_kmeans.fit(data)
custom_centroids, custom_labels = custom_kmeans.evaluate(data)

plt.scatter(data[:, 0], data[:, 1], c=custom_labels, cmap='viridis', alpha=0.5)
plt.scatter(np.array(custom_centroids)[:, 0], np.array(custom_centroids)[:, 1], c='red', marker='x', s=200,
            label='Centroids')
plt.title('K-Means Clustering (6 clusters, 199 iterations, mahalanobis norm)')
plt.legend()
plt.show()

custom_kmeans = KMeans(n_clusters=3, max_iter=199, norm="mahalanobis")
custom_kmeans.fit(data)
custom_centroids, custom_labels = custom_kmeans.evaluate(data)

plt.scatter(data[:, 0], data[:, 1], c=custom_labels, cmap='viridis', alpha=0.5)
plt.scatter(np.array(custom_centroids)[:, 0], np.array(custom_centroids)[:, 1], c='red', marker='x', s=200,
            label='Centroids')
plt.title('K-Means Clustering (3 clusters, 199 iterations, mahalanobis norm)')
plt.legend()
plt.show()

custom_kmeans = KMeans(n_clusters=6, max_iter=119, norm="manhattan")
custom_kmeans.fit(data)
custom_centroids, custom_labels = custom_kmeans.evaluate(data)

plt.scatter(data[:, 0], data[:, 1], c=custom_labels, cmap='viridis', alpha=0.5)
plt.scatter(np.array(custom_centroids)[:, 0], np.array(custom_centroids)[:, 1], c='red', marker='x', s=200,
            label='Centroids')
plt.title('K-Means Clustering (6 clusters, 119 iterations, manhattan norm)')
plt.legend()
plt.show()