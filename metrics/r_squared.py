import numpy as np

def r_squared(y_true, y_pred):
    y_mean = np.mean(y_true)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    r2 = 1 - (ss_res / ss_tot)
    return r2

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

r2_value = r_squared(y_true, y_pred)
print(f"R-Squared: {r2_value:.2f}")
