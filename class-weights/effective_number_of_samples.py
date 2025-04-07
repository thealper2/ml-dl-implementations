import numpy as np

def effective_number_weights(y, beta=0.999, normalize=True):
    classes, counts = np.unique(y, return_counts=True)
    effective_num = (1 - np.power(beta, counts)) / (1 - beta)
    weights = 1.0 / effective_num

    if normalize:
        weights = weights / np.sum(weights) * len(classes)
    
    return dict(zip(classes, weights))