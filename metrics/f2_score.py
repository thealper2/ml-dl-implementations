import numpy as np

def f2_score(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true != 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred != 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    beta = 2
    f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0
    
    return f2

y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1])
y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1])

f2 = f2_score(y_true, y_pred)
print(f"F2-Score: {100 * f2:.2f}%")