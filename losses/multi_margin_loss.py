import numpy as np

def multi_margin_loss(y_true, y_pred, margin=1):
	correct_class_score = y_pred[y_true]
	loss = 0.0
	for i in range(len(y_pred)):
		if i != y_true:
			loss += max(0, margin - (correct_class_score - y_pred[i]))

	return loss
