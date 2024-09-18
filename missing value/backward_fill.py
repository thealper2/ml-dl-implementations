import numpy as np

def backward_fill(data):
    filled_data = data.copy()
    nan_indices = np.isnan(filled_data)
    valid_indices = ~nan_indices
    next_valid_value = None

    for i in range(len(filled_data) - 1, -1, -1):
        if valid_indices[i]:
            next_valid_value = filled_data[i]

        elif next_valid_value is not None:
            filled_data[i] = next_valid_value

    return filled_data

data = np.array([1, np.nan, np.nan, 8, 9, np.nan, 15])
filled_data = backward_fill(data)
print("Raw Data:", data)
print("Filled Data:", filled_data)