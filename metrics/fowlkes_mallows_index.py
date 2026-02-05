import numpy as np

def pairwise_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple[int, int, int]:
    n = len(y_true)
    TP = FP = FN = 0

    for i in range(n):
        for j in range(i + 1, n):
            same_true = y_true[i] == y_true[j]
            same_pred = y_pred[i] == y_pred[j]

            if same_true and same_pred:
                TP += 1
            elif not same_true and same_pred:
                FP += 1
            elif same_true and not same_pred:
                FN += 1

    return TP, FP, FN

def fowlkes_mallows_score(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    TP, FP, FN = pairwise_counts(y_true, y_pred)

    if TP == 0:
        return 0.0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return np.sqrt(precision * recall)
