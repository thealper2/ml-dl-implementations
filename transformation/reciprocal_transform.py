import numpy as np

def reciprocal_transform(data: np.ndarray) -> np.ndarray:
    epsilon = 1e-8
    return 1 / (data + epsilon)

data = np.array([1, 2, 3, 4, 5], dtype=float)
reciprocal_data = reciprocal_transform(data)

print("Original Veri:", data)
print("Reciprocal Transform:", reciprocal_data)
