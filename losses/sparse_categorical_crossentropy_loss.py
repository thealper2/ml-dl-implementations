import numpy as np

def sparse_categorical_crossentropy(y_true, y_pred):
	y_pred = np.clip(y_pred, 1e-10, None)
	loss = -np.sum(np.log(y_pred[np.arange(len(y_true)), y_true])) / len(y_true)
	return loss
