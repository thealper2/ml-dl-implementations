import numpy as np

def inverse_sqrt_class_weights(y, normalize=True):
    classes, counts = np.unique(y, return_counts=True)
    weights = 1.0 / np.sqrt(counts)

    if normalize:
        weights = weights / np.sum(weights) * len(classes)
    
    return dict(zip(classes, weights))