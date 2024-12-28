import numpy as np

def hamming_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("")
    
    distance = np.sum(np.array(vector1) != np.array(vector2))
    return distance