import numpy as np
from scipy import stats
from collections import Counter

def frequent_category_imputation(data):
    filled_data = data.copy()
    nan_indices = [_ for _ in data if data == np.nan]
    mode_value = stats.mode([_ for _ in data if _ != np.nan])[0][0]
    filled_data[nan_indices] = mode_value
    return filled_data

data = np.array(['A', 'B', 'A', np.nan, 'B', 'A', np.nan, 'C'])
filled_data = frequent_category_imputation(data)
print("Raw Data:", data)
print("Filled Data:", filled_data)