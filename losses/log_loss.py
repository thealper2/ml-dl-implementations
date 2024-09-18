import numpy as np

def log_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return loss

y_true = np.array([1, 0, 1, 1, 0, 1, 0])
y_pred = np.array([0.8, 0.2, 0.7, 0.6, 0.4, 0.9, 0.3])

log_loss_value = log_loss(y_true, y_pred)
print(f"Log Loss: {100 * log_loss_value:.2f}%")
