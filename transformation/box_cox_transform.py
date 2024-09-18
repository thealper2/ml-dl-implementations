import numpy as np
from scipy.optimize import minimize

def box_cox_transform(data: np.ndarray, lambda_: float) -> np.ndarray:
    epsilon = 1e-8
    data = data + epsilon
    if lambda_ == 0:
        return np.log(data)
    else:
        return (data ** lambda_ - 1) / lambda_

def box_cox_lambda(data: np.ndarray) -> float:
    def log_likelihood(lambda_):
        transformed_data = box_cox_transform(data, lambda_)
        return -np.sum(np.log(transformed_data))
    
    result = minimize(log_likelihood, x0=0, bounds=[(-5, 5)])
    return result.x[0]

data = np.array([1, 2, 3, 4, 5], dtype=float)

lambda_ = box_cox_lambda(data)

box_cox_data = box_cox_transform(data, lambda_)

print("Original Data:", data)
print("Lambda:", lambda_)
print("Box-Cox Transform:", box_cox_data)
