import numpy as np

def relu6(x):
	return np.minimum(np.maximum(0, x), 6)
