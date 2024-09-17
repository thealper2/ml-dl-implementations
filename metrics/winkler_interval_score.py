import numpy as np

def winkler_interval_score(y_true, y_lower, y_upper):
    interval_width = y_upper - y_lower
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    score = (1 - coverage) + interval_width.mean()
    return score

y_true = np.array([2.5, 0.0, 2, 8])
y_lower = np.array([2.0, -0.5, 1.5, 7.0])
y_upper = np.array([3.0, 0.5, 2.5, 9.0])

winkler_score = winkler_interval_score(y_true, y_lower, y_upper)
print(f"Winkler Interval Score: {winkler_score:.2f}")
