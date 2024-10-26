import numpy as np

def margin_ranking_loss(x1, x2, y, margin=1.0):
	loss = np.maximum(0, -y * (x1 - x2) + margin)
	return np.mean(loss)
