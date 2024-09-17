import numpy as np

def f_beta_score(y_true, y_pred, beta):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true != 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred != 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0
    
    return f_beta

y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1])
y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1])

beta = 4 # F-4 Score
f_beta = f_beta_score(y_true, y_pred, beta)
print(f"F{beta}-Score: {100 * f_beta:.2f}%")