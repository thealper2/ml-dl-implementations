import numpy as np

def categorical_hinge(y_true, y_pred):
	y_true_pred = np.sum(y_true * y_pred, axis=-1)
	y_pred_wrong = np.max((1 - y_true) * y_pred, axis=-1)
	loss = np.maximum(0, 1 - y_true_pred + y_pred_wrong)
	return np.mean(loss)
