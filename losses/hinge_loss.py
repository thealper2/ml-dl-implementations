import numpy as np

def hinge_loss(y_true, y_pred):
    loss = np.maximum(0, 1 - y_true * y_pred)
    return np.mean(loss)

y_true = np.array([1, -1, 1, -1])
y_pred = np.array([0.8, -0.6, 1.2, -1.0])

loss_value = hinge_loss(y_true, y_pred)
print(f"Hinge Loss: {loss_value:.4f}")
