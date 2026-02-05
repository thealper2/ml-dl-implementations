import numpy as np

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def compute_centroids(X: np.ndarray, labels: np.ndarray) -> dict:
    centroids = {}
    for label in np.unique(labels):
        centroids[label] = X[labels == label].mean(axis=0)
    return centroids

def compute_cluster_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: dict
) -> dict:
    scatter = {}
    for label, centroid in centroids.items():
        points = X[labels == label]
        distances = [euclidean_distance(x, centroid) for x in points]
        scatter[label] = np.mean(distances)
    return scatter

def davies_bouldin_index(X: np.ndarray, labels: np.ndarray) -> float:
    centroids = compute_centroids(X, labels)
    scatter = compute_cluster_scatter(X, labels, centroids)

    cluster_labels = list(centroids.keys())
    K = len(cluster_labels)
    R = []

    for i in cluster_labels:
        max_ratio = -np.inf
        for j in cluster_labels:
            if i == j:
                continue

            numerator = scatter[i] + scatter[j]
            denominator = euclidean_distance(centroids[i], centroids[j])
            ratio = numerator / denominator

            max_ratio = max(max_ratio, ratio)

        R.append(max_ratio)

    return np.mean(R)
