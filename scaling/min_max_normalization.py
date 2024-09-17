import numpy as np

def normalize(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data

data = np.array([[1, 2], [3, 4], [5, 6]])
normalized_data = normalize(data)
print(normalized_data)