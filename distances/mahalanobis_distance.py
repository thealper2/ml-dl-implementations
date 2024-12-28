import numpy as np

def mahalanobis_distance(x, y, covariance_matrix):
    inv_cov_matrix = np.linalg.inv(covariance_matrix)
    diff = x - y
    distance = np.sqrt(np.dot(np.dot(diff.T, inv_cov_matrix), diff))
    return distance