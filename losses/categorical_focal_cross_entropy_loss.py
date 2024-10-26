import numpy as np

def categorical_focal_crossentropy(y_true, y_pred, gamma=2.0, alpha=0.25):
	epsilon = 1e-15
	y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
	cross_entropy_loss = -y_true * np.log(y_pred)
	focal_loss = alpha * (1 - y_pred) ** gamma * cross_entropy_loss
	return np.mean(np.sum(focal_loss, axis=1))
