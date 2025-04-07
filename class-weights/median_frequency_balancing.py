import numpy as np

def median_frequency_balancing(y):
    classes, counts = np.unique(y, return_counts=True)
    total = np.sum(counts)
    freqs = counts / total
    median_freq = np.median(freqs)
    weights = median_freq / freqs
    return dict(zip(classes, weights))