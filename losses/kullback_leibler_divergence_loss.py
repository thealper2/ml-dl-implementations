import numpy as np

def kl_divergence(p, q):
	epsilon = 1e-10
	p = np.clip(p, epsilon, 1)
	q = np.clip(q, epsilon, 1)
	return np.sum(p * np.log(p / q))
