import numpy as np

def categorical_cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = - np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(loss)

y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

loss_value = categorical_cross_entropy_loss(y_true, y_pred)
print(f"Categorical Cross-Entropy Loss: {loss_value:.4f}")
