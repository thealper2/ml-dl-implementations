import numpy as np

def z_score(data, threshold=1.5):
    mean = np.mean(data)
    std_dev = np.std(data)

    z_scores = (data - mean) / std_dev

    outliers = data[np.abs(z_scores) > threshold]
    cleaned_data = data[~(np.abs(z_scores) > threshold)]
    return outliers, cleaned_data

data = np.array([15, 18, 19, 20, 21, 22, 24, 28, 29, 30, 35, 40, 200])
outliers, cleaned_data = z_score(data)
print(f"raw data ({len(data)}) :", data)
print(f"cleaned data ({len(cleaned_data)}) :", cleaned_data)