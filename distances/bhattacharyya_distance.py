import numpy as np

def bhattacharyya_distance(p, q):
    """
    Bhattacharyya distance between two discrete probability distributions.
    
    Parameters:
    - p, q: numpy arrays (same shape), unnormalized or normalized histograms
    
    Returns:
    - Bhattacharyya distance (float)
    """
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    bc = np.sum(np.sqrt(p * q))
    
    bc = np.clip(bc, 1e-10, 1.0)
    
    return -np.log(bc)