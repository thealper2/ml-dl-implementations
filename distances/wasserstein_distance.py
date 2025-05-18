import numpy as np

def wasserstein_distance(p, q):
    """
    1D discrete Wasserstein (Earth Mover's) distance between two histograms.
    
    Parameters:
    - p: numpy array, distribution 1 (unnormalized counts or probabilities)
    - q: numpy array, distribution 2 (same shape as p)
    
    Returns:
    - Wasserstein distance (float)
    """
    p = p / np.sum(p)
    q = q / np.sum(q)

    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)

    return np.sum(np.abs(cdf_p - cdf_q))