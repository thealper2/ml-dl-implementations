import numpy as np

def pinball_loss(y_true, y_pred, tau=0.5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    diff = y_true - y_pred
    return np.mean(np.maximum(tau * diff, (tau - 1) * diff))
