import numpy as np

def mean_absolute_error(y_true, y_pred):
	error = np.abs(y_true - y_pred)
	mae = np.mean(error)
	return mae
