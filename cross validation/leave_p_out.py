import numpy as np
from collections import Counter
from itertools import combinations

def leave_p_out_cv(data, labels, p):
    num_samples = len(data)
    folds = []
    
    for test_indices in combinations(range(num_samples), p):
        test_indices = list(test_indices)
        train_indices = list(set(range(num_samples)) - set(test_indices))
        folds.append([train_indices, test_indices])
    
    return folds

data = np.arange(20)
labels = np.concatenate([np.repeat(i, 4) for i in range(5)])

p = 3

folds = leave_p_out_cv(data, labels, p)

for i, (train_indices, test_indices) in enumerate(folds):
    print(f"Iteration {i+1}")
    print(f"Train indices: {train_indices}")
    print(f"Test indices: {test_indices}")
    print("Train class distribution:", Counter(labels[train_indices]))
    print("Test class distribution:", Counter(labels[test_indices]))
    print()
