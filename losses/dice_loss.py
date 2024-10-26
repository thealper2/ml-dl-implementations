import numpy as np

def dice(y_true, y_pred, epsilon=1e-6):
	y_true = y_true.flatten()
	y_pred = y_pred.flatten()
	intersection = np.sum(y_true * y_pred)
	dice_score = (2. * intersection + epsilon) / (np.sum(y_true) + np.sum(y_pred) + epsilon)
	dice_loss_value = 1 - dice_score
	return dice_loss_value
