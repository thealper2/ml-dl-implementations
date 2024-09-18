import numpy as np

def winsorization(data, k=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    cleaned_data = data.copy()
    cleaned_data[cleaned_data < lower_bound] = lower_bound
    cleaned_data[cleaned_data > upper_bound] = upper_bound
    return cleaned_data

data = np.array([15, 18, 19, 20, 21, 22, 24, 28, 29, 30, 35, 40, 200])
cleaned_data = winsorization(data)
print(f"raw data ({len(data)}) :", data)
print(f"cleaned data ({len(cleaned_data)}) :", cleaned_data)