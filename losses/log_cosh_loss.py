import numpy as np

def log_cosh(y_true, y_pred):
	error = y_pred - y_true
	loss = np.sum(np.log(np.cosh(error)))
	return loss
