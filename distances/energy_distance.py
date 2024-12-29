import numpy as np

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def energy_distance(X, Y):
    term1 = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            term1 += euclidean_distance(X[i], X[j])
    term1 /= (X.shape[0] * X.shape[0])

    term2 = 0
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            term2 += euclidean_distance(X[i], Y[j])
    term2 /= (X.shape[0] * Y.shape[0])

    term3 = 0
    for i in range(Y.shape[0]):
        for j in range(Y.shape[0]):
            term3 += euclidean_distance(Y[i], Y[j])
    term3 /= (Y.shape[0] * Y.shape[0])

    ed = 2 * term1 - term2 - term3
    return ed
