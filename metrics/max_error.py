import numpy as np

def max_error(y_true, y_pred):
    absolute_errors = np.abs(y_true - y_pred)
    max_error_value = np.max(absolute_errors)
    return max_error_value

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

max_error_value = max_error(y_true, y_pred)
print(f"Max Error: {max_error_value:.2f}")
