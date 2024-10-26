import numpy as np

def soft_margin_loss(y_true, y_pred):
	loss = np.maximum(0, 1 - y_true * y_pred)
	return np.mean(loss)
