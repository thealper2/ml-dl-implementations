import numpy as np

def sorensen_dice_distance(set1, set2):
    set1 = np.array(set1)
    set2 = np.array(set2)
    intersection = np.intersect1d(set1, set2)
    similarity = (2 * len(intersection)) / (len(set1) + len(set2))
    distance = 1 - similarity
    return distance