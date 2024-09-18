import numpy as np
from scipy.stats import norm, rankdata

def quantile_transform(data: np.ndarray, target_distribution='normal') -> np.ndarray:
    sorted_data = np.sort(data)
    ranks = rankdata(data) - 1
    quantiles = ranks / len(data)

    if target_distribution == 'normal':
        return norm.ppf(quantiles)

data = np.array([1, 5, 3, 4, 2], dtype=float)

transformed_data = quantile_transform(data, target_distribution='normal')

print("Original Data:", data)
print("Quantile Transform:", transformed_data)
