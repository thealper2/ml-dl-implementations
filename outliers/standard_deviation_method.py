import numpy as np

def standard_deviation(data):
    mean = np.mean(data)
    std_dev = np.std(data)

    lower_bound = mean - 2 * std_dev
    upper_bound = mean + 2 * std_dev

    outliers = data[(data < lower_bound) | (data > upper_bound)]
    cleaned_data = data[~((data < lower_bound) | (data > upper_bound))]
    return outliers, cleaned_data

data = np.array([15, 18, 19, 20, 21, 22, 24, 28, 29, 30, 35, 40, 200])
outliers, cleaned_data = standard_deviation(data)
print(f"raw data ({len(data)}) :", data)
print(f"cleaned data ({len(cleaned_data)}) :", cleaned_data)