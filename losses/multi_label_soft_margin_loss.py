import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def multi_label_soft_margin_loss(y_true, y_pred):
	pred_prob = sigmoid(y_pred)
	loss = - (y_true * np.log(pred_prob) + (1 - y_true) * np.log(1 - pred_prob))
	return np.mean(loss)
