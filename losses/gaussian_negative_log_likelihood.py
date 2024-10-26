import numpy as np

def gaussian_nll(y_true, y_pred_mean, y_pred_var):
	epsilon = 1e-6
	y_pred_var = np.clip(y_pred_var, epsilon, None)
	nll = (y_true - y_pred_mean) ** 2 / (2 * y_pred_var) + np.log(y_pred_var) + 0.5 * np.log(2 * np.pi)
	return np.mean(nll)
