import numpy as np

def log_softmax(x):
	exp_x = np.exp(x - np.max(x))
	return x - np.log(np.sum(exp_x))
