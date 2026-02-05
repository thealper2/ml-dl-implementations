def mean_reciprocal_rank(y_true, y_pred):
    rr_sum = 0.0
    for true in y_true:
        rr = 0.0
        for idx, rel in enumerate(true):
            if rel == 1:
                rr = 1.0 / (idx + 1)
                break
        rr_sum += rr

    return rr_sum / len(y_true)
