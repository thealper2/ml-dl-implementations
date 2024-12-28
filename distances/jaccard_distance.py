import numpy as np

def jaccard_distance(set1, set2):
    set1 = np.array(set1)
    set2 = np.array(set2)
    intersection = np.intersect1d(set1, set2)
    union = np.union1d(set1, set2)
    jaccard_similarity = len(intersection) / len(union)
    jaccard_distance = 1 - jaccard_similarity
    return jaccard_distance