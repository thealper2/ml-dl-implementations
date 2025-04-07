import numpy as np

def inverse_class_frequency_weights(y, normalize=True):
    classes, counts = np.unique(y, return_counts=True)
    total = np.sum(counts)
    weights = total / counts.astype(float)

    if normalize:
        weights = weights / np.sum(weights) * len(classes)

    return dict(zip(classes, weights))