import numpy as np

def multi_label_margin_loss(y_true, y_pred):
	positive_classes = np.where(y_true == 1)[0]
	negative_classes = np.where(y_true == 0)[0]
	
	loss = 0.0
	for i in positive_classes:
		for j in negative_classes:
			margin_loss = max(0, 1 - (y_pred[i] - y_pred[j]))
			loss += margin_loss

	if len(positive_classes) > 0:
		loss /= len(positive_classes)

	return loss
