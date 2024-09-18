import numpy as np

def sqrt_transform(data: np.ndarray) -> np.ndarray:
    epsilon = 1e-8
    return np.sqrt(data + epsilon)

data = np.array([1, 4, 9, 16, 25], dtype=float)

sqrt_data = sqrt_transform(data)

print("Original Data:", data)
print("Square Root Transform:", sqrt_data)
