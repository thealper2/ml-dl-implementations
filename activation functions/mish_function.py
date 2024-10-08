import numpy as np

def mish(x):
	return x * np.tanh(np.log1p(np.exp(x)))
