import numpy as np

def poisson_loss(y_true, y_pred):
	y_pred = np.clip(y_pred, 1e-10, None)
	loss = np.sum(y_pred - y_true + y_true * np.log(y_true / y_pred))
	return loss
