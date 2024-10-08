import numpy as np

alpha = 1.67326
lambda_ = 1.0507

def selu(x):
	return lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))
