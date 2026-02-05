import numpy as np

def squared_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return np.dot(diff, diff)

def compute_centroids(X: np.ndarray, labels: np.ndarray) -> dict:
    centroids = {}
    for label in np.unique(labels):
        centroids[label] = X[labels == label].mean(axis=0)
    return centroids

def inertia_score(X: np.ndarray, labels: np.ndarray) -> float:
    centroids = compute_centroids(X, labels)
    inertia = 0.0

    for label, centroid in centroids.items():
        points = X[labels == label]
        for x in points:
            inertia += squared_euclidean(x, centroid)

    return inertia
