import numpy as np

def log_likelihood(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    log_likelihood_value = np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return log_likelihood_value

def mcfadden_pseudo_r2(y_true, y_pred, y_null_pred):
    log_likelihood_model = log_likelihood(y_true, y_pred)
    log_likelihood_null = log_likelihood(y_true, y_null_pred)
    
    r2_mcfadden = 1 - (log_likelihood_model / log_likelihood_null)
    
    return r2_mcfadden

y_true = np.array([1, 0, 1, 1, 0, 1, 0])
y_pred = np.array([0.8, 0.2, 0.7, 0.6, 0.4, 0.9, 0.3])
y_null_pred = np.array([0.5] * len(y_true))

mcfadden_r2 = mcfadden_pseudo_r2(y_true, y_pred, y_null_pred)
print(f"McFadden’s Pseudo R²: {mcfadden_r2:.2f}")
