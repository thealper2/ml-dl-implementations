import numpy as np

def median_imputation(data):
    filled_data = data.copy()
    nan_indices = np.isnan(filled_data)
    median_value = np.nanmedian(filled_data)
    filled_data[nan_indices] = median_value
    return filled_data

data = np.array([1, np.nan, np.nan, 8, 9, np.nan, 15])
filled_data = median_imputation(data)
print("Raw Data:", data)
print("Filled Data:", filled_data)