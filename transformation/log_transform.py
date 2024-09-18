import numpy as np

def log_transform(data: np.ndarray, base: int = 10) -> np.ndarray:
    if base == 10:
        return np.log10(data + 1e-8)
    elif base == np.e:
        return np.log(data + 1e-8)

data = np.array([1, 10, 100, 1000, 10000], dtype=float)

log_data_10 = log_transform(data, base=10)
log_data_e = log_transform(data, base=np.e)

print("Original Data:", data)
print("Logaritmic Transform (Base 10):", log_data_10)
print("Logaritmic Transform:", log_data_e)
