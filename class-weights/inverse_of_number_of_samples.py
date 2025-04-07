import numpy as np

def compute_class_weights(y):
    classes = np.unique(y)
    n_classes = len(classes)
    n_samples = len(y)
    
    class_counts = np.zeros(n_classes)
    for i, c in enumerate(classes):
        class_counts[i] = np.sum(y == c)
    
    weights = n_samples / (n_classes * class_counts)
    return weights