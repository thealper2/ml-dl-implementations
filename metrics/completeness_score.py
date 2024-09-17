import numpy as np

def completeness_score(labels_true, labels_pred):
    unique_true = np.unique(labels_true)
    unique_pred = np.unique(labels_pred)
    
    completeness_scores = []
    
    for true_class in unique_true:
        true_mask = labels_true == true_class
        true_class_size = np.sum(true_mask)
        
        predicted_clusters = np.unique(labels_pred[true_mask])
        
        cluster_scores = []
        for cluster in predicted_clusters:
            cluster_size = np.sum((labels_pred == cluster) & true_mask)
            cluster_scores.append(cluster_size / true_class_size)
        
        completeness_scores.append(np.max(cluster_scores))
    
    completeness_score_value = np.mean(completeness_scores)
    
    return completeness_score_value

labels_true = np.array([0, 0, 1, 1, 2, 2, 1, 0])
labels_pred = np.array([0, 0, 1, 1, 2, 2, 2, 1])

completeness_score_value = completeness_score(labels_true, labels_pred)
print(f"Completeness Score: {completeness_score_value:.2f}")
