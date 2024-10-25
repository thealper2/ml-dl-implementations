import numpy as np

def glu(x):
	x1, x2 = np.split(x, 2)
	return x1 * (1 / (1 + np.exp(-x2)))
