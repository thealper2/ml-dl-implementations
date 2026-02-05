import numpy as np

def squared_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return np.dot(diff, diff)

def compute_centroids(X: np.ndarray, labels: np.ndarray) -> dict:
    centroids = {}
    for label in np.unique(labels):
        centroids[label] = X[labels == label].mean(axis=0)
    return centroids

def within_cluster_dispersion(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: dict
) -> float:
    W = 0.0
    for label, centroid in centroids.items():
        points = X[labels == label]
        for x in points:
            W += squared_euclidean(x, centroid)
    return W

def between_cluster_dispersion(
    centroids: dict,
    labels: np.ndarray,
    global_centroid: np.ndarray
) -> float:
    B = 0.0
    for label, centroid in centroids.items():
        n_k = np.sum(labels == label)
        B += n_k * squared_euclidean(centroid, global_centroid)
    return B

def calinski_harabasz_index(X: np.ndarray, labels: np.ndarray) -> float:
    N = X.shape[0]
    unique_labels = np.unique(labels)
    K = len(unique_labels)

    if K == 1:
        raise ValueError("Calinski-Harabasz Index is undefined for K=1")

    global_centroid = X.mean(axis=0)
    centroids = compute_centroids(X, labels)

    W = within_cluster_dispersion(X, labels, centroids)
    B = between_cluster_dispersion(centroids, labels, global_centroid)

    return (B / (K - 1)) / (W / (N - K))
