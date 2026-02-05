import numpy as np

def msle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("MSLE only works for non-negative targets")
    
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    return np.mean((log_true - log_pred) ** 2)
