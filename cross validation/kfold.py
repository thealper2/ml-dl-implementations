import numpy as np
from collections import Counter

def kfold_cv(data, k):
    fold_size = len(data) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k-1 else len(data)
        test_indices = data[start:end]
        train_indices = np.concatenate([data[:start], data[end:]])
        folds.append([train_indices.tolist(), test_indices.tolist()])
    
    return folds

data = np.arange(20)
labels = np.concatenate([np.repeat(i, 4) for i in range(5)])

k = 5

folds = kfold_cv(data, k)

for i, (train_indices, test_indices) in enumerate(folds):
    print(f"Fold {i+1}")
    print(f"Train indices: {train_indices}")
    print(f"Test indices: {test_indices}")
    print("Train class distribution:", Counter(labels[train_indices]))
    print("Test class distribution:", Counter(labels[test_indices]))
    print()