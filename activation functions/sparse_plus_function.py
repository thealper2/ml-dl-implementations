import numpy as np

def sparse_plus(x, alpha=1.0):
	return np.where(x < 0, 0, np.where(x < alpha, x - alpha, x))
