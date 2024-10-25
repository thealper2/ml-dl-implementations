import numpy as np

def sparse_sigmoid(x, epsilon=1e-6):
	return np.where(x < 0, 0, (1 / (1 + np.exp(-x))) * (1 - epsilon))
