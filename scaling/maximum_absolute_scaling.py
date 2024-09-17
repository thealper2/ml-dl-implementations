import numpy as np

def max_abs_scaling(data):
    data_max = np.max(data, axis=0)
    scaled_data = data / data_max
    return scaled_data

data = np.array([[1, 2], [3, 4], [5, 6]])
normalized_data = max_abs_scaling(data)
print(normalized_data)