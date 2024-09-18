import numpy as np

def arbitrary_number_imputation(data, fill_value):
    filled_data = data.copy()
    nan_indices = np.isnan(filled_data)
    filled_data[nan_indices] = fill_value
    return filled_data

data = np.array([1, np.nan, np.nan, 8, 9, np.nan, 15])
filled_data = arbitrary_number_imputation(data, fill_value=3)
print("Raw Data:", data)
print("Filled Data:", filled_data)