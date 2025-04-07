import numpy as np

def compute_srif_weights(y_true, normalize=True):
    y_true = np.array(y_true)
    classes = np.unique(y_true)
    class_counts = np.array([np.sum(y_true == c) for c in classes])
    weights = 1.0 / np.sqrt(class_counts)

    if normalize:
        weights = weights / np.sum(weights) * len(classes)

    return dict(zip(classes, weights))