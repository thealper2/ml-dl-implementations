import numpy as np
from collections import Counter

def leave_one_out_cv(data, labels):
    num_samples = len(data)
    folds = []
    
    for i in range(num_samples):
        test_indices = [i]
        train_indices = list(set(range(num_samples)) - set(test_indices))
        folds.append([train_indices, test_indices])
    
    return folds

data = np.arange(20)
labels = np.concatenate([np.repeat(i, 4) for i in range(5)])

folds = leave_one_out_cv(data, labels)

for i, (train_indices, test_indices) in enumerate(folds):
    print(f"Iteration {i+1}")
    print(f"Train indices: {train_indices}")
    print(f"Test indices: {test_indices}")
    print("Train class distribution:", Counter(labels[train_indices]))
    print("Test class distribution:", Counter(labels[test_indices]))
    print()
