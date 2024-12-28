import numpy as np

def canberra_distance(x, y):
    distance = np.sum(np.abs(x - y) / (np.abs(x) + np.abs(y)))
    return distance