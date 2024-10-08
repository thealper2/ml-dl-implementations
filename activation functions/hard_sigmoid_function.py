import numpy as np

def hard_sigmoid(x):
	return np.maximum(0, np.minimum(1, (x + 1) / 2))
