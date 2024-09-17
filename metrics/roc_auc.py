import numpy as np
import matplotlib.pyplot as plt

def compute_roc_auc(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))
    
    tpr_list = []
    fpr_list = []
    
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        tpr_list.append(TPR)
        fpr_list.append(FPR)
    
    plt.figure()
    plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig("roc-auc.png")

    tpr_list = np.array(tpr_list)
    fpr_list = np.array(fpr_list)
    auc = np.trapz(tpr_list, fpr_list)
    
    return fpr_list, tpr_list, auc

y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
y_scores = np.array([0.2, 0.3, 0.55, 0.6, 0.75, 0.2, 0.9, 0.6, 0.4, 0.1])

fpr, tpr, auc = compute_roc_auc(y_true, y_scores)
print(f"AUC: {auc:.2f}")
