import numpy as np

def log_likelihood(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    log_likelihood_value = np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return log_likelihood_value

def bic(log_likelihood, num_params, num_observations):
    return -2 * log_likelihood + num_params * np.log(num_observations)

y_true = np.array([1, 0, 1, 1, 0, 1, 0])
y_pred = np.array([0.8, 0.2, 0.7, 0.6, 0.4, 0.9, 0.3])
num_params = 5
num_observations = len(y_true)

log_likelihood_value = log_likelihood(y_true, y_pred)
bic_value = bic(log_likelihood_value, num_params, num_observations)

print(f"BIC: {bic_value:.2f}")
