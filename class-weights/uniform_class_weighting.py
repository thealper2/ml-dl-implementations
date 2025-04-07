import numpy as np

def compute_ucw_weights(y_true):
    y_true = np.array(y_true)
    classes = np.unique(y_true)
    weights = np.ones_like(classes, dtype=float)
    return dict(zip(classes, weights))