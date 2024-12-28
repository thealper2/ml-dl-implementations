import numpy as np

def manhattan_distance(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("")
    
    distance = np.sum(np.abs(np.array(vec1) - np.array(vec2)))
    return distance