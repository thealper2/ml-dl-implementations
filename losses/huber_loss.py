import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    error = np.abs(y_true - y_pred)
    
    loss = np.where(error <= delta,
                    0.5 * (error ** 2),
                    delta * error - 0.5 * delta ** 2)
    
    return np.mean(loss)

y_true = np.array([2.5, 0.0, 2.1, 1.7])
y_pred = np.array([3.0, -0.5, 2.0, 1.5])

loss_value = huber_loss(y_true, y_pred, delta=1.0)
print(f"Huber Loss: {loss_value:.4f}")