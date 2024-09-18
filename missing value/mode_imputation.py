import numpy as np
from scipy import stats

def mode_imputation(data):
    filled_data = data.copy()
    nan_indices = np.isnan(filled_data)
    mode_value = stats.mode(filled_data[~nan_indices])[0][0]
    filled_data[nan_indices] = mode_value
    return filled_data

data = np.array([1, np.nan, 8, 9, np.nan, 15])
filled_data = mode_imputation(data)
print("Raw Data:", data)
print("Filled Data:", filled_data)