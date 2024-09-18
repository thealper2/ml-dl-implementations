import numpy as np

def binary_cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) 
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.7])

loss_value = binary_cross_entropy_loss(y_true, y_pred)
print(f"Binary Cross-Entropy Loss: {loss_value:.4f}")
