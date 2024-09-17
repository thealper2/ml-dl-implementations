import numpy as np

def log_likelihood(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    log_likelihood_value = np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return log_likelihood_value

def aic(log_likelihood, num_params):
    return -2 * log_likelihood + 2 * num_params

y_true = np.array([1, 0, 1, 1, 0, 1, 0])
y_pred = np.array([0.8, 0.2, 0.7, 0.6, 0.4, 0.9, 0.3])
num_params = 5

log_likelihood_value = log_likelihood(y_true, y_pred)
aic_value = aic(log_likelihood_value, num_params)

print(f"AIC: {aic_value:.2f}")
