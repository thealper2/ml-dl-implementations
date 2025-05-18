import numpy as np

def chamfer_distance(set_A, set_B):
    distances_A_to_B = np.min(np.linalg.norm(set_A[:, np.newaxis] - set_B, axis=2), axis=1)
    sum_A_to_B = np.sum(distances_A_to_B)
    
    distances_B_to_A = np.min(np.linalg.norm(set_B[:, np.newaxis] - set_A, axis=2), axis=1)
    sum_B_to_A = np.sum(distances_B_to_A)
    
    return (sum_A_to_B + sum_B_to_A) / (len(set_A) + len(set_B))