import numpy as np

def hard_tanh(x):
	return np.clip(x, -1, 1)
