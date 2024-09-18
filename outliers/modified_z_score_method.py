import numpy as np

def modified_z_score(data, threshold=3.5):
    median = np.median(data)

    mad = np.median(np.abs(data - median))
    
    z_scores = 0.6745 * (data - median) / mad

    outliers = data[np.abs(z_scores) > threshold]
    cleaned_data = data[~(np.abs(z_scores) > threshold)]
    return outliers, cleaned_data

data = np.array([15, 18, 19, 20, 21, 22, 24, 28, 29, 30, 35, 40, 200])
outliers, cleaned_data = modified_z_score(data)
print(f"raw data ({len(data)}) :", data)
print(f"cleaned data ({len(cleaned_data)}) :", cleaned_data)