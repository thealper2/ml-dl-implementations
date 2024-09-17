import numpy as np
from itertools import combinations

def rand_score(labels_true, labels_pred):
    n = len(labels_true)
    
    pairs = list(combinations(range(n), 2))
    
    a = b = c = d = 0
    
    for i, j in pairs:
        if labels_true[i] == labels_true[j] and labels_pred[i] == labels_pred[j]:
            a += 1
        elif labels_true[i] != labels_true[j] and labels_pred[i] != labels_pred[j]:
            b += 1
        elif labels_true[i] == labels_true[j] and labels_pred[i] != labels_pred[j]:
            c += 1
        elif labels_true[i] != labels_true[j] and labels_pred[i] == labels_pred[j]:
            d += 1
    
    rand_score_value = (a + b) / (a + b + c + d)
    return rand_score_value

labels_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
labels_pred = np.array([1, 1, 0, 0, 0, 1, 0, 1])

rand_score_value = rand_score(labels_true, labels_pred)
print(f"Rand Score: {rand_score_value:.2f}")
