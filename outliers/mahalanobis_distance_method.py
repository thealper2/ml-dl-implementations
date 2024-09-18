import numpy as np
from scipy.spatial.distance import mahalanobis

def mahalanobis_distance(target_point, data):
    mean_data = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    delta = target_point - mean_data
    distance = np.sqrt(np.dot(np.dot(delta.T, inv_cov_matrix), delta))
    return distance

data = np.array([[2, 3], [3, 5], [5, 8], [7, 10], [9, 12]])
target_point = np.array([4, 6])

dist = mahalanobis_distance(target_point, data)
print("Mahalanobis distance:", dist)