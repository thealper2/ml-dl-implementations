import numpy as np

def balanced_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    num_classes = len(classes)

    recalls = []

    for cls in classes:
        TP = np.sum((y_true == cls) & (y_pred == cls))
        FN = np.sum((y_true == cls) & (y_pred != cls))
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        recalls.append(recall)

    balanced_acc = np.mean(recalls)
    return balanced_acc

y_true = np.array([1, 0, 1, 1, 2, 0, 0, 1, 0])
y_pred = np.array([1, 1, 1, 0, 2, 2, 0, 1, 1])

balanced_acc = balanced_accuracy(y_true, y_pred)
print(f"Balanced Accuracy Score: {100 * balanced_acc:.02f}%")