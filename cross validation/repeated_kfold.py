import numpy as np
from collections import Counter

def repeated_kfold_cv(data, labels, k, n_repeats) -> list:
    folds = []
    num_samples = len(data)
    indices = np.arange(num_samples)
    
    for _ in range(n_repeats):
        np.random.shuffle(indices)
        
        fold_size = num_samples // k
        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size if i != k-1 else num_samples
            test_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            folds.append([train_indices.tolist(), test_indices.tolist()])
    
    return folds

data = np.arange(20)
labels = np.concatenate([np.repeat(i, 4) for i in range(5)])

k = 5
n_repeats = 3

folds = repeated_kfold_cv(data, labels, k, n_repeats)

for i, (train_indices, test_indices) in enumerate(folds):
    print(f"Fold {i+1}")
    print(f"Train indices: {train_indices}")
    print(f"Test indices: {test_indices}")
    print("Train class distribution:", Counter(labels[train_indices]))
    print("Test class distribution:", Counter(labels[test_indices]))
    print()
