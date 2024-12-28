import numpy as np

def chebyshev_distance(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("")
    
    abs_diff = np.abs(np.array(vec1) - np.array(vec2))
    return np.max(abs_diff)