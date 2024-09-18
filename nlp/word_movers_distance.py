import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment

def word_movers_distance(vectors1, vectors2):
    vectors1 = normalize(vectors1)
    vectors2 = normalize(vectors2)
    distance_matrix = cdist(vectors1, vectors2, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    return distance_matrix[row_ind, col_ind].sum()

vectors1 = np.array([[0.5, 0.1], [0.3, 0.2]])
vectors2 = np.array([[0.4, 0.2], [0.2, 0.3]])
wmd_value = word_movers_distance(vectors1, vectors2)
print(f"Word Mover's Distance: {wmd_value:.4f}")
