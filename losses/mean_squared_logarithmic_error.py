import numpy as np

def mean_squared_logarithmic_error(y_true, y_pred):
	log_y_true = np.log(y_true + 1)
	log_y_pred = np.log(y_pred + 1)
	errors = log_y_true - log_y_pred
	squared_errors = np.square(errors)
	msle = np.mean(squared_errors)
	return msle
