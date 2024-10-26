import numpy as np

def mean_squared_error(y_true, y_pred):
	errors = y_true - y_pred
	squared_errors = np.square(errors)
	mse = np.mean(squared_errors)
	return mse
