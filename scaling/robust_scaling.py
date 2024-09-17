import numpy as np

def robust_scaling(data):
    data_median = np.median(data, axis=0)

    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)

    iqr = q3 - q1

    iqr = np.where(iqr == 0, 1 , iqr)

    scaled_data = (data - data_median) / iqr
    return scaled_data

data = np.array([[1, 2], [3, 4], [5, 6]])
scaled_data = robust_scaling(data)
print(scaled_data)