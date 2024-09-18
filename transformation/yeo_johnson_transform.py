import numpy as np
from scipy.optimize import minimize

def yeo_johnson_transform(data: np.ndarray, lambda_: float) -> np.ndarray:
    epsilon = 1e-8
    data = data + epsilon
    if lambda_ == 0:
        return np.log1p(data)
    elif lambda_ > 0:
        return ((data ** lambda_ - 1) / lambda_) * (data > 0) + ((-((-data) ** (2 - lambda_)) + 1) / (2 - lambda_)) * (data <= 0)
    else:
        return ((data ** lambda_ - 1) / lambda_) * (data > 0) + ((-((-data) ** (2 - lambda_)) + 1) / (2 - lambda_)) * (data <= 0)

def yeo_johnson_lambda(data: np.ndarray) -> float:
    def log_likelihood(lambda_):
        transformed_data = yeo_johnson_transform(data, lambda_)
        return -np.sum(np.log(transformed_data + 1e-8))

    result = minimize(log_likelihood, x0=0, bounds=[(-5, 5)])
    return result.x[0]

data = np.array([1, 2, -3, -4, 5], dtype=float)

lambda_ = yeo_johnson_lambda(data)

yeo_johnson_data = yeo_johnson_transform(data, lambda_)

print("Original Data:", data)
print("Lambda:", lambda_)
print("Yeo-Johnson Transform:", yeo_johnson_data)
