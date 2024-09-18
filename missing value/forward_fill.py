import numpy as np
import pandas as pd

def forward_fill(data):
    filled_data = data.copy()
    nan_indices = np.isnan(filled_data)
    valid_indices = ~nan_indices
    last_valid_value = None

    for i in range(len(filled_data)):
        if valid_indices[i]:
            last_valid_value = filled_data[i]

        elif last_valid_value is not None:
            filled_data[i] = last_valid_value

    return filled_data

data = np.array([1, np.nan, np.nan, 8, 9, np.nan, 15])
filled_data = forward_fill(data)
print("Raw Data:", data)
print("Filled Data:", filled_data)