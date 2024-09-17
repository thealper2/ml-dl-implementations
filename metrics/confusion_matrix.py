import numpy as np

def confusion_matrix(y_true, y_pred, labels):
    num_labels = len(labels)
    label_index = {label: i for i, label in enumerate(labels)}

    matrix = np.zeros((num_labels, num_labels), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_index[true_label]
        pred_idx = label_index[pred_label]
        matrix[true_idx, pred_idx] += 1

    return matrix

y_true = ['cat', 'dog', 'dog', 'cat', 'cat', 'dog']
y_pred = ['cat', 'dog', 'cat', 'cat', 'cat', 'dog']
labels = ['cat', 'dog']

cm = confusion_matrix(y_true, y_pred, labels)
print("Confusion Matrix:\n", cm)
# [3 0]
# [1 2]