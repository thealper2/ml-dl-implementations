def precision_at_k(y_true, y_pred, k):
    y_pred_k = y_pred[:k]
    hits = len(set(y_pred_k) & set(y_true))
    precision = hits / k
    return precision
