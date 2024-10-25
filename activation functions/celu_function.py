import numpy as np

def celu(x, alpha=1.0):
	return np.where(x > 0, x, alpha * (np.exp(x) - 1))

