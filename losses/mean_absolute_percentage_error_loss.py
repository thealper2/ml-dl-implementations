import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
	percentage_error = np.abs((y_true, y_pred) / y_true) * 100
	mape = np.mean(percentage_error)
	return mape
