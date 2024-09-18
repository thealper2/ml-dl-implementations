import numpy as np

def power_transform(data: np.ndarray, power: float) -> np.ndarray:
    if power == 0:
        return np.log(data + 1e-8)
    else:
        return np.power(data, power)

data = np.array([1, 2, 3, 4, 5], dtype=float)

power = 2
transformed_data = power_transform(data, power)

print("Original Data:", data)
print(f"Power Transform (power={power}):", transformed_data)
