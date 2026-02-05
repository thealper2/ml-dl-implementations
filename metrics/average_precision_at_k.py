def average_precision_at_k(y_true, y_pred, k):
    if len(y_true) == 0:
        return 0.0

    score = 0.0
    hits = 0

    for i, item in enumerate(y_pred[:k]):
        if item in y_true:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(y_true), k)
