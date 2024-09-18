import numpy as np

def tukeys_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    cleaned_data = data[~((data < lower_bound) | (data > upper_bound))]
    return outliers, cleaned_data

data = np.array([15, 18, 19, 20, 21, 22, 24, 28, 29, 30, 35, 40, 200])
outliers, cleaned_data = tukeys_iqr(data)
print(f"raw data ({len(data)}) :", data)
print(f"cleaned data ({len(cleaned_data)}) :", cleaned_data)