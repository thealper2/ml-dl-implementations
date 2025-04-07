import numpy as np

def cb_focal_loss(y_true, y_pred, beta=0.999, gamma=2.0):
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 1e-7, 1. - 1e-7)
    N, C = y_pred.shape
    class_counts = np.array([np.sum(y_true == c) for c in range(C)])
    eff_num = (1 - np.power(beta, class_counts)) / (1 - beta)
    weights = 1.0 / eff_num
    weights = weights / np.sum(weights) * C
    losses = []
    for i in range(N):
        true_class = y_true[i]
        pt = y_pred[i, true_class]
        alpha = weights[true_class]
        focal_term = (1 - pt) ** gamma
        loss = -alpha * focal_term * np.log(pt)
        losses.append(loss)

    return np.mean(losses)