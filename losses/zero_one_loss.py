import numpy as np

def zero_one_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return np.mean(y_true != y_pred)
