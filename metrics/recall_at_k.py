def recall_at_k(y_true, y_pred, k):
    y_pred_k = y_pred[:k]
    hits = len(set(y_pred_k) & set(y_true))
    recall = hits / len(y_true) if len(y_true) > 0 else 0.0
    return recall
