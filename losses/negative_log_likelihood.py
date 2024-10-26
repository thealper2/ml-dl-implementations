import numpy as np

def negative_log_likelihood(y_true, y_pred):
	epsilon = 1e-15
	y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
	nll = -np.sum(y_true * np.log(y_pred))
	return nll / y_true.shape[0]
