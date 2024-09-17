import numpy as np

def precision_score(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true != 1) & (y_pred == 1))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision

y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1])
y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1])

prec = precision_score(y_true, y_pred)
print(f"Precision Score: {100 * prec:.2f}%")