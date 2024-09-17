import numpy as np

def top_k_accuracy_score(y_true, y_scores, k):
    top_k_preds = np.argsort(y_scores, axis=1)[:, -k:]

    correct = 0
    for i in range(len(y_true)):
        if y_true[i] in top_k_preds[i]:
            correct += 1

    top_k_acc = correct / len(y_true)
    return top_k_acc

k = 2
y_true = np.array([0, 1, 2, 2, 1])
y_scores = np.array([[0.2, 0.5, 0.3],
                     [0.1, 0.6, 0.3],
                     [0.3, 0.3, 0.4],
                     [0.4, 0.4, 0.2],
                     [0.1, 0.7, 0.2]])

top_k_acc = top_k_accuracy_score(y_true, y_scores, k)
print(f"Top-K Accuracy Score: {100 * top_k_acc:.02f}%")