import numpy as np

def log_sigmoid(x):
	return -np.log(1 + np.exp(-x))
