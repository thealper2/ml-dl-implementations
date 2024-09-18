import numpy as np

def cube_root_transform(data: np.ndarray) -> np.ndarray:
    return np.cbrt(data)

data = np.array([1, 8, 27, 64, 125], dtype=float)

cube_root_data = cube_root_transform(data)

print("Original Data:", data)
print("Cube Root Transform:", cube_root_data)
