import numpy as np

def standardize(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    standardized_data = (data - data_mean) / data_std
    return standardized_data

data = np.array([[1, 2], [3, 4], [5, 6]])
standardized_data = standardize(data)
print(standardized_data)