import numpy as np

def compute_lif_weights(y_true, normalize=True, epsilon=1e-8):
    y_true = np.array(y_true)
    classes = np.unique(y_true)
    class_counts = np.array([np.sum(y_true == c) for c in classes])
    weights = 1.0 / (np.log(class_counts + epsilon))

    if normalize:
        weights = weights / np.sum(weights) * len(classes)

    return dict(zip(classes, weights))