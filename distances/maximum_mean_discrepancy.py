import numpy as np

def rbf_kernel(x, y, gamma=1.0):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

def maximum_mean_discrepancy(X, Y, gamma=1.0):
    mmd_squared += np.mean([rbf_kernel(xi, xj, gamma) for xi in X for xj in X])
    mmd_squared += np.mean([rbf_kernel(yi, yj, gamma) for yi in Y for yj in Y])
    mmd_squared -= 2 * np.mean([rbf_kernel(xi, yi, gamma) for xi in X for yi in Y])
    return np.sqrt(mmd_squared)
