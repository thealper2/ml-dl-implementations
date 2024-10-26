import numpy as np

def binary_focal_crossentropy(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    cross_entropy_loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    focal_loss = alpha * ((1 - y_pred) ** gamma) * y_true * np.log(y_pred) + \
                 alpha * (y_pred ** gamma) * (1 - y_true) * np.log(1 - y_pred)
    
    loss = -focal_loss
    return np.mean(loss)
