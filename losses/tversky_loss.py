import numpy as np

def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5):
	TP = np.sum(y_true * y_pred)
	FP = np.sum((1 - y_true) * y_pred)
	FN = np.sum(y_true * (1 - y_pred))

	tversky = TP / (TP + alpha * FP + beta * FN)
	return 1 - tversky
