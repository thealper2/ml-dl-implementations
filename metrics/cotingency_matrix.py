import numpy as np

def contingency_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple[np.ndarray, dict, dict]:
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)

    class_index = {c: i for i, c in enumerate(classes)}
    cluster_index = {k: j for j, k in enumerate(clusters)}

    matrix = np.zeros((len(classes), len(clusters)), dtype=int)

    for t, p in zip(y_true, y_pred):
        i = class_index[t]
        j = cluster_index[p]
        matrix[i, j] += 1

    return matrix, class_index, cluster_index
