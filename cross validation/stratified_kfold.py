import numpy as np
from collections import Counter

def stratified_kfold_cv(data, labels, k) -> list:
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    
    folds = []
    indices = np.arange(len(data))
    
    np.random.shuffle(indices)
    
    fold_size = len(data) // k
    for i in range(k):
        test_indices = []
        train_indices = []
        
        for label in unique_labels:
            label_indices_for_class = label_indices[label]
            np.random.shuffle(label_indices_for_class)
            
            start = i * fold_size // len(unique_labels)
            end = (i + 1) * fold_size // len(unique_labels) if i != k-1 else len(label_indices_for_class)
            test_indices_for_class = label_indices_for_class[start:end]
            
            test_indices.extend(test_indices_for_class)
            train_indices_for_class = np.setdiff1d(label_indices_for_class, test_indices_for_class)
            train_indices.extend(train_indices_for_class)
        
        folds.append([np.array(train_indices).tolist(), np.array(test_indices).tolist()])
    
    return folds

data = np.arange(20)
labels = np.concatenate([np.repeat(i, 4) for i in range(5)])

k = 5

folds = stratified_kfold_cv(data, labels, k)

for i, (train_indices, test_indices) in enumerate(folds):
    print(f"Fold {i+1}")
    print(f"Train indices: {train_indices}")
    print(f"Test indices: {test_indices}")
    print("Train class distribution:", Counter(labels[train_indices]))
    print("Test class distribution:", Counter(labels[test_indices]))
    print()
