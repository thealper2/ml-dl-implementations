import numpy as np

def poisson_nll(y_true, y_pred):
	epsilon = 1e-15
	y_pred = np.clip(y_pred, epsilon, None)
	nll = y_pred - y_true * np.log(y_pred) + np.log(np.arange(1, y_true.max() + 1).prod())
	return np.mean(nll)
