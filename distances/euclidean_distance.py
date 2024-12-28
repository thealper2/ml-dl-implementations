import numpy as np

def euclidean_distance(point1, point2):
    diff = point1 - point2
    squared_diff = np.square(diff)
    distance = np.sqrt(np.sum(squared_diff))
    return distance