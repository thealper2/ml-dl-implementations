import numpy as np

def hausdorff_distance(set_A, set_B):
    min_distances_A_to_B = np.array([np.min(np.linalg.norm(b - set_A, axis=1)) for b in set_B])
    h_A_to_B = np.max(min_distances_A_to_B)
    
    min_distances_B_to_A = np.array([np.min(np.linalg.norm(a - set_B, axis=1)) for a in set_A])
    h_B_to_A = np.max(min_distances_B_to_A)
    
    return max(h_A_to_B, h_B_to_A)