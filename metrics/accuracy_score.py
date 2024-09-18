import numpy as np

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred, axis=0) / len(y_true)

y_true = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0])
y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1])

acc = (y_true == y_pred).mean()
# acc = accuracy_score(y_true, y_pred)
print(f"Accuracy Score: {100 * acc:.02f}%")