import numpy as np

def explained_variance_score(y_true, y_pred):
    variance_y = np.var(y_true)
    variance_residuals = np.var(y_true - y_pred)
    evs = 1 - (variance_residuals / variance_y)
    return evs

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

evs_value = explained_variance_score(y_true, y_pred)
print(f"Explained Variance Score: {evs_value:.2f}")
