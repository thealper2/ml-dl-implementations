import numpy as np

def squared_hinge_loss(y_true, y_pred):
	hinge_loss = np.maximum(0, 1 - y_true * y_pred)
	squared_loss = np.mean(hinge_loss * 2)
	return squared_loss
