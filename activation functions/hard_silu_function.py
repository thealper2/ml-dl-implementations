import numpy as np

def hard_silu(x):
	return x * np.maximum(0, np.minimum(1, (x + 1) / 2))
