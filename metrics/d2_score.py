import numpy as np

def d2_mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    baseline = np.mean(y_true)
    model_loss = np.sum((y_true - y_pred) ** 2)
    baseline_loss = np.sum((y_true - baseline) ** 2)
    return 1 - model_loss / baseline_loss
