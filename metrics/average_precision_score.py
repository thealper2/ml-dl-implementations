import numpy as np

def average_precision(y_true, y_scores):
    order = np.argsort(y_scores)[::-1]
    y_true = y_true[order]
    
    precisions = []
    recalls = []
    tp = 0
    fp = 0
    total_positives = np.sum(y_true)
    
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        
        precision = tp / (tp + fp)
        recall = tp / total_positives
        
        precisions.append(precision)
        recalls.append(recall)
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    
    return ap

y_true = np.array([1, 0, 1, 1, 0, 1, 0])
y_scores = np.array([0.8, 0.2, 0.7, 0.6, 0.4, 0.9, 0.3]) 

ap_score = average_precision(y_true, y_scores)
print(f"Average Precision Score: {100 * ap_score:.2f}%")
