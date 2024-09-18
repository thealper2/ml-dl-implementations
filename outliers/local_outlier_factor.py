import numpy as np
from scipy.spatial import distance

def k_nearest_neighbors(data, point, k):
    distances = np.linalg.norm(data - point, axis=1)
    nearest_neighbors = np.argsort(distances)[1:k+1]
    return nearest_neighbors, distances[nearest_neighbors]

def local_reachability_distance(data, point, k, distances_to_neighbors):
    k_distances = []
    for neighbor_idx in k_nearest_neighbors(data, point, k)[0]:
        k_distances.append(np.linalg.norm(data[neighbor_idx] - data, axis=1))
    
    reachability_distances = np.maximum(distances_to_neighbors, np.mean(k_distances))
    return 1 / (np.mean(reachability_distances))

def lof_score(data, point, k):
    neighbors, distances_to_neighbors = k_nearest_neighbors(data, point, k)
    lrd_point = local_reachability_distance(data, point, k, distances_to_neighbors)
    
    lrd_neighbors = np.array([local_reachability_distance(data, data[neighbor], k, distances_to_neighbors)
                              for neighbor in neighbors])
    
    lof = np.mean(lrd_neighbors) / lrd_point
    return lof

data = np.array([[2, 3], [3, 5], [5, 8], [7, 10], [9, 12], [10, 15], [12, 16], [20, 30]])
point = np.array([12, 16])

lof = lof_score(data, point, k=3)
# LOF = 1 -> Normal
# LOF < 1 -> Normal
# LOF > 1 -> Outlier
print("LOF Score:", lof)
