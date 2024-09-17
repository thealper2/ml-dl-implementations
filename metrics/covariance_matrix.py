import numpy as np

def calculate_covariance_matrix(vectors):
    return np.cov(np.stack(vectors, axis=0))

vectors = [[1, 2, 3], [4, 5, 6]]
cov_mat = calculate_covariance_matrix(vectors)
print(cov_mat) # [[1., 1.], [1., 1.]]